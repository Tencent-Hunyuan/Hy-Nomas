"""Batch independent native PixelUMM windows; retain native noise and blending."""

from types import MethodType


def install_cpu_streaming_state(vae):
    """Keep CPU preview tails on CPU while native offload restores latent inputs.

    The pinned native session returns CPU pixels in single-GPU mode as well as
    window-parallel mode. Its retained tail must share that output device.
    """
    create = vae.create_streaming_decode_session

    def move(self, device):
        import torch

        target = torch.device(device)
        if self.buffer is not None and self.buffer.device != target:
            self.buffer = self.buffer.to(device=target, non_blocking=False)
        pixels = torch.device("cpu") if self.to_cpu else target
        if self.pending is not None and self.pending.device != pixels:
            self.pending = self.pending.to(device=pixels, non_blocking=False)

    def session(*, to_cpu=True, retain_history=False):
        value = create(to_cpu=to_cpu, retain_history=retain_history)
        value._move_decode_state = MethodType(move, value)
        return value

    vae.create_streaming_decode_session = session


def install_gpu_output(pipeline):
    vae = pipeline.vae
    create = vae.create_streaming_decode_session

    def session(*, to_cpu=True, retain_history=False):
        return create(to_cpu=False, retain_history=retain_history)

    vae.create_streaming_decode_session = session
    native_postprocess = pipeline._postprocess_vae_video

    def postprocess(video, *, inplace=False):
        import torch

        pixels = native_postprocess(video, inplace=inplace)
        if pixels.shape[0] != 1:
            raise ValueError("GPU RGB8 requires one world per worker")
        rgb = copy_rgb8(pixels[0])
        return torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0)

    pipeline._postprocess_vae_video = postprocess


def copy_rgb8(video):
    import torch

    frames = video.permute(1, 2, 3, 0)
    host = torch.empty(frames.shape, dtype=torch.uint8, pin_memory=True)
    for first in range(0, len(frames), 8):
        rgb = (frames[first : first + 8] * 255).clamp(0, 255).to(torch.uint8)
        host[first : first + 8].copy_(rgb, non_blocking=True)
    torch.cuda.current_stream(video.device).synchronize()
    return host.numpy()


def install_window_batching(vae, batch_size, *, quantized=False):
    if batch_size <= 1:
        return
    original = vae._predict_x0

    def predict(
        self,
        latent,
        xt_source,
        *,
        all_ranks,
        t_starts=None,
        pad_short=False,
        output_cpu=False,
    ):
        import torch
        import torch.distributed as dist
        from hymm.models.vae.pixelumm_decoder.autoencoder import (
            _TemporalBlender,
            _window_starts,
        )

        if latent.shape[0] != 1:
            return original(
                latent,
                xt_source,
                all_ranks=all_ranks,
                t_starts=t_starts,
                pad_short=pad_short,
                output_cpu=output_cpu,
            )
        frames, height, width = latent.shape[2:]
        window = self.temporal_window_latents
        starts = (
            list(t_starts)
            if t_starts is not None
            else _window_starts(frames, window, window - self.temporal_overlap_latents)
        )
        length = min(window, frames)
        ys, xs, tile_h, tile_w = self._tile_grid(height, width)
        if quantized and (length * tile_h * tile_w * 4) % 128:
            # Preserve the native per-window 128-row activation-scale blocks.
            return original(
                latent,
                xt_source,
                all_ranks=all_ranks,
                t_starts=t_starts,
                pad_short=pad_short,
                output_cpu=output_cpu,
            )
        spatial = [(y, x) for y in ys for x in xs]
        tasks = [(ti, y, x) for ti in range(len(starts)) for y, x in spatial]
        pad_noise = None
        if pad_short and frames >= window and starts[-1] + window > frames:
            variance, mean = torch.var_mean(
                latent.detach().float(), dim=(2, 3, 4), correction=0, keepdim=True
            )
            generator = torch.Generator(device=latent.device)
            generator.manual_seed(self.seed * 1_000_003 + frames - 1)
            pad_noise = mean + variance.sqrt() * torch.randn(
                (1, latent.shape[1], window, height, width),
                generator=generator,
                device=latent.device,
                dtype=torch.float32,
            )

        def inputs(index):
            ti, y, x = tasks[index]
            t0 = starts[ti]
            real = latent[:, :, t0 : t0 + length, y : y + tile_h, x : x + tile_w]
            if pad_noise is not None and real.shape[2] < length:
                fill = length - real.shape[2]
                offset = (t0 // (window - self.temporal_overlap_latents)) % (
                    window - fill + 1
                )
                tile = pad_noise[
                    :, :, offset : offset + fill, y : y + tile_h, x : x + tile_w
                ]
                real = torch.cat([real, tile.to(real.dtype)], dim=2)
            return real, xt_source(t0, t0 + length, y, x, tile_h, tile_w)

        group = self._distributed_decode_group
        distributed = self._distributed_decode_mode != "none"
        world = dist.get_world_size(group) if distributed else 1
        rank = dist.get_rank(group) if distributed else 0
        output_rank = self._distributed_decode_output_rank
        receives = not distributed or all_ranks or rank == output_rank
        blender = _TemporalBlender(
            (starts[-1] + length - starts[0]) * 4, output_cpu=output_cpu
        )
        pending, next_window = {}, 0

        def absorb(index, pixels):
            nonlocal next_window
            ti, si = divmod(index, len(spatial))
            pending.setdefault(ti, [None] * len(spatial))[si] = pixels
            while next_window in pending and all(
                x is not None for x in pending[next_window]
            ):
                blender.add(
                    self._merge_tiles(pending.pop(next_window), ys, xs),
                    (starts[next_window] - starts[0]) * 4,
                )
                next_window += 1

        for first in range(0, len(tasks), world * batch_size):
            count = min(batch_size, -(-(len(tasks) - first) // world))
            indices = [first + rank + j * world for j in range(count)]
            values = [inputs(min(index, len(tasks) - 1)) for index in indices]
            local = self.decoder(
                torch.cat([v[0] for v in values]), torch.cat([v[1] for v in values])
            ).contiguous()
            if distributed:
                gathered = (
                    [torch.empty_like(local) for _ in range(world)]
                    if receives
                    else None
                )
                if all_ranks:
                    dist.all_gather(gathered, local, group=group)
                else:
                    dist.gather(
                        local,
                        gather_list=gathered,
                        dst=dist.get_global_rank(group, output_rank),
                        group=group,
                    )
            else:
                gathered = [local]
            if receives:
                for j in range(count):
                    for source in range(world):
                        index = first + source + j * world
                        if index < len(tasks):
                            absorb(index, gathered[source][j : j + 1])
        if not receives:
            return None
        if pending or next_window != len(starts):
            raise RuntimeError("Batched PixelUMM left incomplete native windows")
        return blender.result()

    vae._predict_x0 = MethodType(predict, vae)
