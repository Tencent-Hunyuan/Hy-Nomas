"""Cache per-segment Ulysses lengths; keep the native element permutation."""


class SequenceExchange:
    def __init__(self, packed_qkv=False):
        import hymm.models.dit.nomas_cpt as native

        self.original = native.all_to_all_4D
        self.lengths = {}
        self.packed_qkv = packed_qkv
        self.pending = []
        self.receive = None
        self.stats = {
            "length_queries": 0,
            "fast_calls": 0,
            "fallback_calls": 0,
            "packed_exchanges": 0,
        }
        native.all_to_all_4D = self

    def reset(self):
        self.lengths.clear()
        self.pending.clear()
        self.receive = None

    def begin_step(self):
        # All ranks enter through the same native segment boundary.
        self.lengths.clear()
        if self.pending:
            raise RuntimeError("Incomplete native Q/K/V exchange at segment boundary")

    def __call__(self, input_, group, scatter_dim=2, gather_dim=1):
        import torch
        import torch.distributed as dist

        world = dist.get_world_size(group)
        if world == 1:
            return input_
        batch, length, heads, dim = input_.shape
        key = id(group)
        if scatter_dim == 2 and gather_dim == 1:
            if key not in self.lengths:
                local = torch.tensor([length], dtype=torch.int64, device=input_.device)
                values = torch.empty(world, dtype=torch.int64, device=input_.device)
                dist.all_gather_into_tensor(values, local, group=group)
                self.lengths[key] = (length, tuple(values.cpu().tolist()))
                self.stats["length_queries"] += 1
            local_length, lengths = self.lengths[key]
            if length != local_length:
                raise RuntimeError("Ulysses input length changed inside one segment")
            if len(set(lengths)) != 1:
                self.stats["fallback_calls"] += 1
                return self.original(input_, group, scatter_dim, gather_dim)
            if self.packed_qkv:
                return self._packed(input_, group, world)
            shard_heads = heads // world
            send = (
                input_.reshape(batch, length, world, shard_heads, dim)
                .transpose(0, 2)
                .contiguous()
            )
            receive = torch.empty_like(send)
            dist.all_to_all_single(receive, send, group=group)
            self.stats["fast_calls"] += 1
            return (
                receive.reshape(length * world, batch, shard_heads, dim)
                .transpose(0, 1)
                .contiguous()
            )
        if scatter_dim == 1 and gather_dim == 2 and key in self.lengths:
            local_length, lengths = self.lengths[key]
            if len(set(lengths)) == 1 and length == local_length * world:
                send = (
                    input_.reshape(batch, world, local_length, heads, dim)
                    .transpose(0, 3)
                    .transpose(0, 1)
                    .contiguous()
                )
                receive = torch.empty_like(send)
                dist.all_to_all_single(receive, send, group=group)
                self.stats["fast_calls"] += 1
                return (
                    receive.reshape(heads * world, local_length, batch, dim)
                    .transpose(0, 2)
                    .contiguous()
                )
        self.stats["fallback_calls"] += 1
        return self.original(input_, group, scatter_dim, gather_dim)

    def _packed(self, input_, group, world):
        # The pinned cached-attention function calls Q, K, V consecutively,
        # before reading any result. Return views of one receive allocation;
        # the V call submits the fused pack/collective on the same CUDA stream.
        import torch
        import torch.distributed as dist

        from hy_nomas_fusions import pack_qkv

        batch, length, heads, dim = input_.shape
        slot = len(self.pending)
        if slot == 0:
            self.receive = torch.empty(
                (world, length, batch, heads // world, 3 * dim),
                dtype=input_.dtype,
                device=input_.device,
            )
        elif (
            input_.shape != self.pending[0].shape
            or input_.dtype != self.pending[0].dtype
        ):
            raise RuntimeError("Native Q/K/V exchange shapes differ")
        result = self.receive.reshape(
            length * world, batch, heads // world, 3 * dim
        ).transpose(0, 1)[..., slot * dim : (slot + 1) * dim]
        self.pending.append(input_)
        if slot == 2:
            send = pack_qkv(*self.pending, world)
            dist.all_to_all_single(self.receive, send, group=group)
            self.pending.clear()
            self.receive = None
            self.stats["packed_exchanges"] += 1
        self.stats["fast_calls"] += 1
        return result
