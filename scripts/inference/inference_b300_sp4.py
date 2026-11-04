"""Benchmark GPU-resident video on four B300 ranks (SP4 or TP4; SP4 by default)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

process_start = time.perf_counter()
root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path, help="Pipeline input JSON or original annotation JSON")
parser.add_argument("--output", type=Path, default=root / "outputs/b300_sp4")
parser.add_argument("--gpus", default="0,1,2,3", help="Four distinct physical GPU indices")
parser.add_argument("--segments", type=int, help="Default: all input segments")
parser.add_argument("--validate-only", action="store_true",
                    help="Validate and convert input without loading models or launching workers")
parser.add_argument("--entry-index", type=int, help="Zero-based entry to select from a multi-entry JSON list")
parser.add_argument("--input-base-dir", type=Path,
                    help="Original input directory for resolving sidecars after copying the JSON")
parser.add_argument("--write-input", type=Path,
                    help="Write a new pipeline JSON and exit without launching workers")
cli, overrides = parser.parse_known_args()
input_path, output_dir = cli.input.resolve(), cli.output.resolve()
if not input_path.is_file():
    parser.error(f"Input JSON does not exist: {input_path}")
gpus = cli.gpus.split(",")
if (len(gpus) != 4 or not all(g.isascii() and g.isdigit() for g in gpus)
        or len({int(g) for g in gpus}) != 4):
    parser.error("--gpus requires four distinct numeric GPU indices")
# Convert annotations using the regular entry point's prompt shaping, then
# validate before starting four workers or loading models.
try:
    data = json.loads(input_path.read_text())
except (OSError, ValueError) as exc:
    parser.error(f"Cannot read input JSON: {exc}")
sys.path.insert(0, str(root))
from hymm.utils.b300_input import prepare_b300_input
try:
    data = prepare_b300_input(data, input_path, entry_index=cli.entry_index,
                             overrides=overrides, input_base_dir=cli.input_base_dir)
except (ValueError, OSError, TypeError, KeyError) as exc:
    parser.error(str(exc))
required = {"prompts", "segment_frame_boundaries", "num_frames", "height", "width",
            "guidance_scale", "num_inference_steps", "seed"}
optional = {"camera_guidance_scale", "camera_trajectory", "segment_has_camera",
            "cfg_schedule", "cfg_skip_steps", "trajectory_translation_scale",
            "trajectory_rotation_scale"}
if not isinstance(data, dict) or not required <= data.keys():
    parser.error("Expected pipeline input JSON with keys: " + ", ".join(sorted(required)))
if data.keys() - required - optional:
    parser.error("Unsupported input keys: " + ", ".join(sorted(data.keys() - required - optional))
                 + "; flow_shift and negative_prompt are supplied through Hydra config")
if not isinstance(data["prompts"], list) or not data["prompts"] or not all(
        isinstance(prompt, str) and prompt.strip() for prompt in data["prompts"]):
    parser.error("prompts must be a nonempty list of nonempty strings")
boundaries = data["segment_frame_boundaries"]
if (not isinstance(boundaries, list) or len(boundaries) != len(data["prompts"]) + 1
        or any(type(value) is not int for value in boundaries)
        or boundaries[0] != 0
        or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
    parser.error("segment_frame_boundaries must start at 0 and increase strictly, with one more entry than prompts")
for key in ("num_frames", "height", "width", "num_inference_steps"):
    if type(data[key]) is not int or data[key] <= 0:
        parser.error(f"{key} must be a positive integer")
if data["num_frames"] != boundaries[-1]:
    parser.error("num_frames must match the last segment_frame_boundaries entry")
if type(data["seed"]) is not int:
    parser.error("seed must be an integer")
if type(data["guidance_scale"]) not in (int, float) or not math.isfinite(data["guidance_scale"]):
    parser.error("guidance_scale must be a finite number")
if "segment_has_camera" in data and data["segment_has_camera"] is not None:
    if (not isinstance(data["segment_has_camera"], list)
            or len(data["segment_has_camera"]) != len(data["prompts"])
            or any(type(value) is not bool for value in data["segment_has_camera"])):
        parser.error("segment_has_camera must contain one boolean per prompt")
segment_count = cli.segments if cli.segments is not None else len(data["prompts"])
if not 1 <= segment_count <= len(data["prompts"]):
    parser.error("--segments must be between 1 and the input prompt count")
data["prompts"] = data["prompts"][:segment_count]
data["num_frames"] = boundaries[segment_count]
data["segment_frame_boundaries"] = boundaries[:segment_count + 1]
if data.get("segment_has_camera") is not None:
    data["segment_has_camera"] = data["segment_has_camera"][:segment_count]
if cli.write_input is not None:
    try:
        with cli.write_input.open("x", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except OSError as exc:
        parser.error(f"Cannot create pipeline JSON {cli.write_input}: {exc}")
    print(f"Wrote pipeline input: {cli.write_input}")
if cli.validate_only or cli.write_input is not None:
    print(f"Valid B300 input: {segment_count} segments, {data['num_frames']} frames")
    raise SystemExit(0)

if "RANK" not in os.environ:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_input.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=cli.gpus, HY_KV_ARCHIVE_DEVICE="cuda",
               HY_KV_FAST_HISTORY="1", MAGI_ATTENTION_KERNEL_BACKEND="cutedsl")
    defaults = {
        "CUDA_HOME": str(root / "vendor/cuda"), "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "DG_JIT_CACHE_DIR": str(root / ".cache/deep_gemm"),
        "NCCL_IB_DISABLE": "1", "NCCL_NET_PLUGIN": "none",
        "NCCL_NET": "Socket", "NCCL_DEBUG": "WARN",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=4", str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=root, env=env,
    ))

import contextlib
import gc
import datetime

sys.path.insert(0, str(root))
sys.argv = [sys.argv[0], "+experiment=inference",
    f"dit_weight={root}/weights/Hy-Nomas/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
    f"model.vae.checkpoint={root}/weights/Hy-Nomas/VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors",
    f"model.text_encoder.model_name={root}/weights/Qwen2.5-VL-7B-Instruct",
    "model.vae.deepgemm_fp8=true", "model.vae.cutlass_fp8=false",
    "model.vae.attention_backend=magi", "model.vae.cute_attention=true",
    "model.vae.compile_decoder=true", "model.vae.parallel_mode=window",
    "model.vae.cpu_offload=false", "model.dit.seg_ar_attention_backend=magi",
    "parallelism.sp_size=4", "parallelism.tp_size=1", "parallelism.shard_text_encoder=false",
    "parallelism.text_encoder_cpu_offload=false", "parallelism.dit_cpu_offload=false",
    "parallelism.dit_cpu_offload_for_decode=false", "inference.torch_compile=true",
    "inference.sliding_window.enabled=true",
    "inference.sliding_window.sampling=temporal_context",
    "inference.sliding_window.past_frames=36", "inference.profile_memory=false",
    "inference.inference_mode=true", *overrides]
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from hymm.distributed.parallel_states import ParallelDims, initialize_sequence_parallel_state_v2
from configs.config import parse_args
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
from hymm.utils.inference_metrics import InferenceMetrics, tensor_storage_bytes
from hymm.utils.inference_mode import offline_sample_mode
from hymm.utils.video_output import video_to_uint8_numpy
from hymm.models.dit.modules.nomas_compile_prepost import compilation_snapshot, compilation_delta

torch.set_grad_enabled(False)
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
args = parse_args(mode="inference")
if world_size != 4 or (args.parallelism.sp_size, args.parallelism.tp_size) not in ((4, 1), (1, 4)):
    raise ValueError("B300 entry point requires four ranks and (sp_size, tp_size) of (4, 1) or (1, 4)")
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
dims = ParallelDims(dp_replicate=1, dp_shard=-1,
                    sp=args.parallelism.sp_size, tp=args.parallelism.tp_size,
                    pp=1, world_size=world_size)
world_mesh = dims.build_mesh(device_type="cuda")
initialize_sequence_parallel_state_v2(dims, world_mesh)
assert not args.vae.cpu_offload
assert not args.parallelism.text_encoder_cpu_offload
assert not args.parallelism.dit_cpu_offload
assert not args.parallelism.dit_cpu_offload_for_decode
assert args.vae.attention_backend == "magi" and args.vae.deepgemm_fp8
assert not args.vae.cutlass_fp8
assert args.dit.seg_ar_attention_backend == "magi"
# SP and TP compile local arithmetic; TP collectives stay outside the graphs.
assert args.inference.torch_compile
assert args.vae.compile_decoder and args.vae.cute_attention
assert os.environ.get("HY_KV_ARCHIVE_DEVICE") == "cuda"
output_dir.mkdir(parents=True, exist_ok=True)
load_start = time.perf_counter()
pipeline = CPTCausalInferencePipeline(args, device=device, dit_model_path=args.dit_weight,
                                     world_mesh=world_mesh)
pipeline.vae.decode_output_cpu = False
torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_start
ready_seconds = time.perf_counter() - process_start
def residency():
    modules = dict(dit=pipeline.transformer, text_encoder=pipeline.text_encoder, vae=pipeline.vae)
    result = {name: tensor_storage_bytes(module) for name, module in modules.items()}
    if any(size for devices in result.values() for device, size in devices.items()
           if not device.startswith("cuda")):
        raise RuntimeError(f"Model tensors are not fully GPU-resident: {result}")
    return result

initial_residency = residency()
print(json.dumps({"model_load_seconds": load_seconds, "residency_bytes": initial_residency}), flush=True)


class TopLevelMetrics(InferenceMetrics):
    @contextlib.contextmanager
    def stage(self, name):
        allowed = {
            "te_H2D", "text_encode", "te_D2H", "denoise", "cache_write",
            "dit_D2H", "pixelumm_H2D", "vae_decode", "pixelumm_D2H",
            "postprocess", "dit_H2D", "finite_check", "uint8_conversion",
        }
        nested = {"kv_d2h", "kv_h2d", "kv_cpu_pack", "kv_gpu_pack", "kv_gpu_archive"}
        if (self.stack and name not in nested) or name not in allowed | nested:
            yield
        else:
            with super().stage(name):
                yield


@contextlib.contextmanager
def measured_segment_profile(pipeline, rank, enabled):
    """Attach the existing segment hook only to the post-warmup request."""
    original = pipeline.transformer.forward_with_cache
    profiler = None
    if enabled:
        from hymm.utils.torch_profile_windows import install
        profiler = install(pipeline, rank)
    try:
        yield profiler
    finally:
        if profiler is not None:
            try:
                profiler.finalize()
            finally:
                pipeline.transformer.forward_with_cache = original


results = []
for index in range(2):
    # Drop unreachable request-local KV graphs before allocating another history.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(json.dumps({"run_ready": index,
                      "allocated_gib": torch.cuda.memory_allocated() / 2**30}), flush=True)
    run_dir = output_dir / f"run{index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.pop("HY_INFER_SAVE_FINAL_LATENTS_PATH", None)
    before = compilation_snapshot()
    dist.barrier()
    metrics = TopLevelMetrics(device, detailed=True, jsonl=run_dir / f"stages.rank{rank}.jsonl")
    request_start = time.perf_counter()
    with measured_segment_profile(pipeline, rank, enabled=index == 1) as profiler:
        with offline_sample_mode(True), metrics:
            output = pipeline.inference(
                **data, flow_shift=args.inference.infer_flow_shift_video,
                negative_prompt=args.inference.neg_prompt,
            )
            if os.environ.get("HY_KV_ARCHIVE_DEVICE", "cpu") == "cuda":
                if pipeline.kv_cache is not None:
                    raise RuntimeError("Cannot release history of a persistent request")
                from hymm.models.dit.modules.uniform_history_cache import release_gpu_request_history
                release_gpu_request_history()
            torch.cuda.synchronize()
            dist.barrier()
    summary = metrics.summary()
    rank_summaries = [None] * world_size
    dist.all_gather_object(rank_summaries, summary)
    compile_info = compilation_delta(before)
    residency_info = residency()
    local_info = dict(rank=rank, compilation=compile_info, residency=residency_info,
                      torch_profile_enabled=profiler is not None,
                      gpu_kv_audit=getattr(pipeline, "_last_kv_optimization_audit", {}))
    rank_info = [None] * world_size
    dist.all_gather_object(rank_info, local_info)
    if rank != 0:
        del output
        dist.barrier()
        continue
    if output.device.type != "cuda":
        raise RuntimeError("Pipeline output must remain on GPU")
    gpu_output_shape = list(output.shape)
    finite_start = time.perf_counter()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Non-finite decoder output")
    finite_seconds = time.perf_counter() - finite_start
    conversion_start = time.perf_counter()
    frames = video_to_uint8_numpy(output[0], chunk_frames=16)
    conversion_seconds = time.perf_counter() - conversion_start
    del output
    encode_start = time.perf_counter()
    if index == 1:
        with imageio.get_writer(
            str(run_dir / "video.mp4"), fps=24, codec="libx264",
            quality=8, macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
    encode_seconds = time.perf_counter() - encode_start
    request_seconds = time.perf_counter() - request_start
    if index == 1:
        imageio.imwrite(run_dir / "preview.jpg", frames[len(frames) // 2])
    if len(frames) != data["num_frames"] or frames.max() == frames.min():
        raise RuntimeError("Invalid generated video")
    result = {
        "run": index,
        "warmup": index == 0,
        "torch_profile_enabled": any(info["torch_profile_enabled"] for info in rank_info),
        "prompt_to_gpu_video_seconds": summary["total_seconds"],
        "request_to_mp4_seconds": request_seconds if index == 1 else None,
        "gpu_output_shape": gpu_output_shape,
        "finite_check_seconds": finite_seconds,
        "uint8_conversion_and_D2H_seconds": conversion_seconds,
        "mp4_encode_seconds": encode_seconds if index == 1 else None,
        "process_to_first_mp4_seconds": time.perf_counter() - process_start if index == 1 else None,
        "frames": len(frames), "shape": list(frames.shape[1:]),
        "gpu_e2e_fps": len(frames) / summary["total_seconds"],
        "request_to_mp4_fps": len(frames) / request_seconds if index == 1 else None,
        "pixel_std": float(np.std(frames[::24])),
        "metrics": summary, "compilation": compilation_delta(before),
        "stage_rows": metrics.rows,
        "rank_summaries": rank_summaries,
        "rank_info": rank_info,
        "residency_bytes_after": residency(),
    }
    results.append(result)
    (output_dir / "results.json").write_text(json.dumps({
        "model_load_seconds": load_seconds,
        "startup_to_model_ready_seconds": ready_seconds,
        "torch_compile": args.inference.torch_compile,
        "vae_compile_decoder": args.vae.compile_decoder,
        "vae_cute_attention": args.vae.cute_attention,
        "dit_attention_backend": args.dit.seg_ar_attention_backend,
        "vae_attention_backend": args.vae.attention_backend,
        "vae_gemm_backend": "deepgemm",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sp_size": args.parallelism.sp_size,
        "tp_size": args.parallelism.tp_size,
        "initial_residency_bytes": initial_residency,
        "fast_history": os.environ.get("HY_KV_FAST_HISTORY", "0") == "1",
        "kv_archive_device": os.environ.get("HY_KV_ARCHIVE_DEVICE", "cpu"),
        "segment_count": segment_count,
        "input_path": str(input_path),
        "segment_frame_boundaries": data["segment_frame_boundaries"],
        "runs": results,
        "profiling_note": "When HY_PROFILE_DIR is set, only run1 is traced. Timings include profiling overhead; request-to-MP4 also includes final trace export. Use B300_PROFILE=0 in the shell launcher for unprofiled timing.",
        "notes": "Run0 full requested-segments warmup (no MP4); run1 measured same-process request, fresh KV history. All model weights resident on GPU, output and VAE blend stay GPU. Full-clip VAE decode after all DiT segments. Prompt-to-GPU-video includes text encode, noise preparation, DiT, cache logic and VAE/postprocess, excludes startup, validation, D2H conversion and video export. Existing disk caches retained. SP4 compiles DiT cache pre/post; TP4 compiles local regions around explicit gathers and reduce-scatters.",
    }, indent=2))
    print(json.dumps({k:v for k,v in result.items()
                      if k not in {"stage_rows", "rank_info"}}, indent=2), flush=True)
    del frames
    dist.barrier()
dist.destroy_process_group()

