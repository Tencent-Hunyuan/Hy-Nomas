#!/usr/bin/env python3
"""
Hy-Nomas Video Generation — Gradio Web UI

Provides a browser-based interface for text-to-video generation with:
  - Built-in request queue (configurable max size)
  - Serialized GPU access (one inference at a time via threading lock)
  - Generation controls for resolution, duration, CFG, and seed
  - Multi-prompt support (separate scenes with ||| or paste JSON with events)
  - Batch generation from JSON prompt files (benchmark suites)

Usage:
    # Basic — load an experiment config
    python apps/gradio_app.py --experiment inference

    # Override checkpoint
    python apps/gradio_app.py -e inference \\
        --dit-weight /path/to/ckpt.safetensors

    # Public link + custom port
    python apps/gradio_app.py -e inference --port 8080 --share

    # Multi-GPU with sequence parallelism (sp_size from config)
    torchrun --nproc_per_node=8 apps/gradio_app.py -e inference --port 8080
"""

import os
import sys
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from configs.config import _flatten_cfg

# ---------------------------------------------------------------------------
# Resolution / duration presets
# ---------------------------------------------------------------------------

RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "352 x 640  (360p)": (352, 640),
    "480 x 832  (480p)": (480, 832),
    "720 x 1280 (720p)": (720, 1280),
}

DURATION_PRESETS: Dict[str, int] = {
    " 2 s  (49 frames)": 49,
    " 5 s  (121 frames)": 121,
    "10 s  (241 frames)": 241,
    "15 s  (361 frames)": 361,
    "20 s  (481 frames)": 481,
}


# ---------------------------------------------------------------------------
# Config loading (Hydra compose, then flatten)
# ---------------------------------------------------------------------------


def load_config(
    experiment: str, overrides: Optional[Dict[str, str]] = None
) -> DictConfig:
    """Load a full config from an experiment YAML via Hydra compose + flatten."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()

    override_list = [f"+experiment={experiment}"]
    if overrides:
        for key, val in overrides.items():
            override_list.append(f"{key}={val}")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=override_list)

    OmegaConf.resolve(cfg)
    return _flatten_cfg(cfg)


# ---------------------------------------------------------------------------
# JSON prompt parsing (for Interactive tab)
# ---------------------------------------------------------------------------

JSON_DESC_FALLBACKS = [
    "visual_description_long",
    "visual_description_mid",
    "visual_description_short",
    "label",
]


def _parse_interactive_json(
    prompt_text: str,
    desc_field_first: str,
    desc_field_rest: str,
    max_events: int,
    fps: int,
) -> Tuple[str, Optional[List[int]], int]:
    """Parse a JSON prompt entry pasted into the Interactive tab.

    Accepts either a single JSON object ``{...}`` with an ``"events"`` list,
    or a JSON array ``[{...}, ...]`` (only the first item is used).

    Returns:
        (prompt_string_joined_by_|||, frame_boundaries_or_None, num_frames_or_0)
        num_frames is 0 when no timestamps are available (caller keeps its own).
    """
    data = json.loads(prompt_text)

    if isinstance(data, list):
        if not data:
            return prompt_text, None, 0
        data = data[0]

    if not isinstance(data, dict):
        return prompt_text, None, 0

    events = data.get("events", [])
    if not events:
        return prompt_text, None, 0

    if max_events and max_events > 0 and len(events) > max_events:
        events = events[:max_events]

    from hymm.inference import _parse_timestamp

    prompts: List[str] = []
    frame_boundaries: List[int] = []
    has_timestamps = True

    for i, event in enumerate(events):
        field = desc_field_first if i == 0 else desc_field_rest
        text = (event.get(field) or "").strip()
        if not text:
            for fb in JSON_DESC_FALLBACKS:
                text = (event.get(fb) or "").strip()
                if text:
                    break
        if not text:
            continue

        prompts.append(text)

        st = event.get("start_time")
        et = event.get("end_time")
        if st is not None and et is not None:
            start_sec = _parse_timestamp(st)
            end_sec = _parse_timestamp(et)
            if len(frame_boundaries) == 0:
                frame_boundaries.append(int(start_sec * fps))
            frame_boundaries.append(int(end_sec * fps))
        else:
            has_timestamps = False

    if not prompts:
        return prompt_text, None, 0

    boundaries = (
        frame_boundaries
        if has_timestamps and len(frame_boundaries) == len(prompts) + 1
        else None
    )
    num_frames = frame_boundaries[-1] if boundaries else 0

    return "|||".join(prompts), boundaries, num_frames


# ---------------------------------------------------------------------------
# Inference server
# ---------------------------------------------------------------------------


class InferenceServer:
    """Wraps pipeline loading and thread-safe generation."""

    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        world_mesh=None,
    ):
        self.config = config
        self.device = device
        self.pipeline = None
        self.rollout_type: str = config.inference.rollout_type
        self.rank = rank
        self.world_size = world_size
        self.world_mesh = world_mesh

        self._lock = threading.Lock()
        self._request_counter = 0

        self._load_pipeline()

    # ---- model loading ----

    def _load_pipeline(self):
        from hymm.pipeline.bidirectional import MultiPromptBidirectionalPipeline
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline

        args = self.config
        logger.info(f"Loading pipeline  rollout_type={self.rollout_type} ...")
        t0 = time.time()

        if self.rollout_type == "one_shot":
            self.pipeline = MultiPromptBidirectionalPipeline(
                args=args,
                device=self.device,
                dit_model_path=args.dit_weight,
                world_mesh=self.world_mesh,
            )
        else:
            self.pipeline = CPTCausalInferencePipeline(
                args=args,
                device=self.device,
                dit_model_path=args.dit_weight,
                world_mesh=self.world_mesh,
            )

        logger.info(f"Pipeline ready  ({time.time() - t0:.1f} s)")

    # ---- public generation entry ----

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        cfg_scale: float,
        num_steps: int,
        flow_shift: float,
        seed: int,
        use_dynamic_cfg: bool,
        cfg_min: float,
        cfg_max: float,
        cfg_max_step: int,
        segment_boundaries: str = "",
        desc_field_first: str = "visual_description_long",
        desc_field_rest: str = "description_change",
        max_events_json: int = 4,
    ) -> Tuple[str, str]:
        """Run one generation request (blocks until GPU is free). Rank-0 only."""
        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a prompt.")

        with self._lock:
            self._request_counter += 1
            rid = self._request_counter
            logger.info(f"[#{rid}] Generation start")
            t0 = time.time()

            try:
                seed = int(seed)
                if seed < 0:
                    seed = torch.randint(0, 2**31, (1,)).item()

                # Manual boundaries from text input
                boundaries = None
                if segment_boundaries and segment_boundaries.strip():
                    try:
                        boundaries = [
                            int(x.strip())
                            for x in segment_boundaries.split(",")
                            if x.strip()
                        ]
                    except ValueError:
                        boundaries = None

                # Auto-detect JSON prompt input
                effective_prompt = prompt
                effective_num_frames = int(num_frames)
                stripped = prompt.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        fps = self.config.inference.video_fps
                        parsed_prompt, json_bounds, json_nframes = (
                            _parse_interactive_json(
                                stripped,
                                desc_field_first.strip(),
                                desc_field_rest.strip(),
                                int(max_events_json),
                                fps,
                            )
                        )
                        effective_prompt = parsed_prompt
                        if not boundaries and json_bounds:
                            boundaries = json_bounds
                        if json_nframes > 0:
                            effective_num_frames = json_nframes
                        logger.info(
                            f"[#{rid}] JSON input detected: "
                            f"{len(parsed_prompt.split('|||'))} prompts, "
                            f"boundaries={json_bounds}, "
                            f"num_frames={json_nframes or effective_num_frames}"
                        )
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug(f"[#{rid}] Not valid JSON, using as-is: {exc}")

                params = dict(
                    rid=rid,
                    prompt=effective_prompt,
                    negative_prompt=negative_prompt,
                    height=int(height),
                    width=int(width),
                    num_frames=effective_num_frames,
                    cfg_scale=float(cfg_scale),
                    num_steps=int(num_steps),
                    flow_shift=float(flow_shift),
                    seed=seed,
                    use_dynamic_cfg=bool(use_dynamic_cfg),
                    cfg_min=float(cfg_min),
                    cfg_max=float(cfg_max),
                    cfg_max_step=int(cfg_max_step),
                    segment_boundaries=boundaries,
                )
                return self._collective_inference(params)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise gr.Error(
                    "GPU out of memory. Try reducing resolution, frame count, "
                    "or inference steps."
                )
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                if "out of memory" in str(exc).lower():
                    raise gr.Error(
                        "GPU out of memory. Try reducing resolution, frame count, "
                        "or inference steps."
                    )
                logger.exception(f"[#{rid}] Failed")
                raise gr.Error(f"Generation failed: {exc}")
            except gr.Error:
                raise
            except Exception as exc:
                torch.cuda.empty_cache()
                logger.exception(f"[#{rid}] Failed")
                raise gr.Error(f"Generation failed: {exc}")
            finally:
                torch.cuda.empty_cache()
                logger.info(f"[#{rid}] Done in {time.time() - t0:.1f} s")

    # ---- collective inference (all ranks participate) ----

    def _collective_inference(self, params):
        """Run inference collectively across all ranks.

        Rank 0 provides params (from Gradio); other ranks receive via broadcast.
        Returns (video_path, info_str) on rank 0, None on other ranks.

        Optional params keys:
          - segment_boundaries: List[int] for frame boundaries (AR or one_shot)
          - save_subdir: subdirectory under gradio_outputs/
          - save_prefix: filename prefix (default "video")
        """
        if self.world_size > 1:
            obj_list = [params]
            dist.broadcast_object_list(obj_list, src=0)
            params = obj_list[0]

        if params is None:
            return None

        rid = params["rid"]
        prompt = params["prompt"]
        negative_prompt = params["negative_prompt"]
        height = params["height"]
        width = params["width"]
        num_frames = params["num_frames"]
        cfg_scale = params["cfg_scale"]
        num_steps = params["num_steps"]
        flow_shift = params["flow_shift"]
        seed = params["seed"]
        solver_type = self.config.inference.solver_type
        if solver_type != "euler":
            raise ValueError(
                f"Unsupported inference solver: {solver_type!r}; expected 'euler'"
            )
        use_dynamic_cfg = params["use_dynamic_cfg"]
        cfg_min = params["cfg_min"]
        cfg_max = params["cfg_max"]
        cfg_max_step = params["cfg_max_step"]
        segment_boundaries = params.get("segment_boundaries")

        prompts: List[str] = [p.strip() for p in prompt.split("|||") if p.strip()]
        if not prompts:
            if self.rank == 0:
                raise gr.Error("Prompt is empty after parsing.")
            return None

        if self.rollout_type != "one_shot":
            self.pipeline.kv_cache = None
            self.pipeline.cache_valid_len = 0
            self.pipeline.global_frame_counter = 0
            torch.cuda.empty_cache()

        neg = negative_prompt.strip() if negative_prompt else None

        try:
            if self.rollout_type == "one_shot":
                output = self.pipeline.inference(
                    prompts=prompts,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=cfg_scale,
                    num_inference_steps=num_steps,
                    flow_shift=flow_shift,
                    seed=seed,
                    negative_prompt=neg,
                    prompt_video_boundaries=segment_boundaries,
                    use_dynamic_cfg_scale=use_dynamic_cfg,
                    cfg_scale_min=cfg_min,
                    cfg_scale_max=cfg_max,
                    cfg_scale_max_step=cfg_max_step,
                )
            else:
                output = self.pipeline.inference(
                    prompts=prompts,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=cfg_scale,
                    num_inference_steps=num_steps,
                    flow_shift=flow_shift,
                    seed=seed,
                    negative_prompt=neg,
                    segment_frame_boundaries=segment_boundaries,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if (
                isinstance(exc, RuntimeError)
                and "out of memory" not in str(exc).lower()
            ):
                raise
            logger.error(f"[#{rid}] OOM during inference, cleaning up")
            if self.rollout_type != "one_shot":
                self.pipeline.kv_cache = None
                self.pipeline.cache_valid_len = 0
                self.pipeline.global_frame_counter = 0
            torch.cuda.empty_cache()
            if self.rank == 0:
                raise torch.cuda.OutOfMemoryError(str(exc))
            return None

        if self.rank != 0:
            torch.cuda.empty_cache()
            return None

        video_tensor = output[0]
        video_np = video_tensor.permute(1, 2, 3, 0)
        video_np = (video_np * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        del output, video_tensor
        torch.cuda.empty_cache()

        import imageio

        save_subdir = params.get("save_subdir", "")
        output_dir = PROJECT_ROOT / "gradio_outputs"
        if save_subdir:
            output_dir = output_dir / save_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = params.get("save_prefix", "video")
        safe_prefix = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in save_prefix
        )[:100]
        video_path = str(output_dir / f"{safe_prefix}_{ts}_{seed}.mp4")

        fps = self.config.inference.video_fps
        imageio.mimwrite(
            video_path,
            video_np,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )

        nf = video_np.shape[0]

        metadata = {
            "request_id": rid,
            "timestamp": ts,
            "prompt": prompt,
            "prompts": prompts,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": nf,
            "duration_seconds": round(nf / fps, 2),
            "fps": fps,
            "seed": seed,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "flow_shift": flow_shift,
            "solver_type": solver_type,
            "use_dynamic_cfg": use_dynamic_cfg,
            "cfg_min": cfg_min,
            "cfg_max": cfg_max,
            "cfg_max_step": cfg_max_step,
            "segment_boundaries": segment_boundaries,
            "rollout_type": self.rollout_type,
            "sp_size": int(getattr(self.config.parallelism, "sp_size", 1)),
            "tp_size": int(getattr(self.config.parallelism, "tp_size", 1)),
            "video_path": video_path,
        }
        meta_path = video_path.rsplit(".", 1)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        info_lines = [
            f"Request #{rid}  —  Complete",
            "=" * 36,
            f"Time:        see log",
            f"Frames:      {nf}  ({nf / fps:.1f} s @ {fps} fps)",
            f"Resolution:  {height} x {width}",
            f"Seed:        {seed}",
            f"Steps:       {num_steps}",
            f"CFG Scale:   {cfg_scale}",
            f"Flow Shift:  {flow_shift}",
            f"Solver:      {solver_type}",
            f"Prompts:     {len(prompts)}",
            f"Boundaries:  {segment_boundaries or 'even split'}",
            f"SP Size:     {self.world_size}",
            "=" * 36,
            f"Saved to: {video_path}",
            f"Metadata: {meta_path}",
        ]

        return video_path, "\n".join(info_lines)

    # ---- batch generation ----

    def generate_batch(
        self,
        json_file,
        desc_field_first: str,
        desc_field_rest: str,
        max_events: int,
        height: int,
        width: int,
        num_frames: int,
        cfg_scale: float,
        num_steps: int,
        flow_shift: float,
        seed: int,
    ) -> Tuple[Optional[str], List[str], str]:
        """Run batch inference on all entries from a JSON prompt file.

        Returns (latest_video_path, all_video_paths, summary_text).
        """
        if json_file is None:
            raise gr.Error("Please upload a JSON prompt file.")

        from hymm.inference import _load_from_json, PROMPT_SEPARATOR

        json_path = (
            Path(json_file) if isinstance(json_file, str) else Path(json_file.name)
        )
        entries = _load_from_json(
            json_path,
            max_events=int(max_events) if max_events else None,
            keep_keywords=None,
            fps=self.config.inference.video_fps,
            desc_field_first=(desc_field_first.strip() if desc_field_first else ""),
            desc_field_rest=(desc_field_rest.strip() if desc_field_rest else ""),
        )

        if not entries:
            raise gr.Error("No prompt entries found in the JSON file.")

        seed_val = int(seed)
        if seed_val < 0:
            seed_val = torch.randint(0, 2**31, (1,)).item()

        with self._lock:
            self._request_counter += 1
            batch_rid = self._request_counter
            logger.info(
                f"[Batch #{batch_rid}] Processing {len(entries)} entries "
                f"from {json_path.name}"
            )
            t0 = time.time()

            video_paths: List[str] = []
            info_parts: List[str] = []
            latest_video: Optional[str] = None

            for i, (entry, frame_boundaries, bench_filename) in enumerate(entries):
                try:
                    params = dict(
                        rid=batch_rid * 1000 + i,
                        prompt=entry,
                        negative_prompt="",
                        height=int(height),
                        width=int(width),
                        num_frames=int(num_frames),
                        cfg_scale=float(cfg_scale),
                        num_steps=int(num_steps),
                        flow_shift=float(flow_shift),
                        seed=seed_val,
                        use_dynamic_cfg=False,
                        cfg_min=1.0,
                        cfg_max=8.0,
                        cfg_max_step=16,
                        segment_boundaries=frame_boundaries,
                        save_subdir="batch",
                        save_prefix=bench_filename or f"entry_{i:04d}",
                    )

                    result = self._collective_inference(params)
                    if result:
                        video_path, _ = result
                        video_paths.append(video_path)
                        latest_video = video_path
                        n_prompts = len(
                            [p for p in entry.split(PROMPT_SEPARATOR) if p.strip()]
                        )
                        info_parts.append(
                            f"[{i + 1}/{len(entries)}] {bench_filename or i}: "
                            f"{n_prompts} prompts -> {video_path}"
                        )
                    logger.info(
                        f"[Batch #{batch_rid}] Entry {i + 1}/{len(entries)} done"
                    )
                    torch.cuda.empty_cache()

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    torch.cuda.empty_cache()
                    if (
                        isinstance(exc, RuntimeError)
                        and "out of memory" not in str(exc).lower()
                    ):
                        info_parts.append(f"[{i + 1}/{len(entries)}] ERROR: {exc}")
                        logger.exception(
                            f"[Batch #{batch_rid}] Entry {i + 1}/{len(entries)}: failed"
                        )
                    else:
                        info_parts.append(
                            f"[{i + 1}/{len(entries)}] OOM ERROR - skipped"
                        )
                        logger.error(
                            f"[Batch #{batch_rid}] Entry {i + 1}/{len(entries)}: OOM"
                        )
                except gr.Error:
                    raise
                except Exception as exc:
                    torch.cuda.empty_cache()
                    info_parts.append(f"[{i + 1}/{len(entries)}] ERROR: {exc}")
                    logger.exception(
                        f"[Batch #{batch_rid}] Entry {i + 1}/{len(entries)}: failed"
                    )

            elapsed = time.time() - t0
            summary = (
                f"Batch #{batch_rid} — {len(video_paths)}/{len(entries)} "
                f"videos in {elapsed:.1f}s\n" + "=" * 40 + "\n" + "\n".join(info_parts)
            )
            return latest_video, video_paths, summary

    # ---- worker loop for non-rank-0 processes ----

    def worker_loop(self):
        """Block forever, participating in collective inference when rank 0 requests it."""
        logger.info(f"[Rank {self.rank}] Entering worker loop")
        while True:
            try:
                self._collective_inference(None)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if (
                    isinstance(exc, RuntimeError)
                    and "out of memory" not in str(exc).lower()
                ):
                    raise
                logger.error(f"[Rank {self.rank}] OOM in worker loop, recovering")
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.error(f"[Rank {self.rank}] Error in worker loop: {exc}")
                torch.cuda.empty_cache()

    # ---- helpers ----

    @property
    def model_summary(self) -> str:
        c = self.config
        return (
            f"Model: {c.dit.model_type}  |  "
            f"Rollout: {self.rollout_type}  |  "
            f"VAE: {c.vae_type}  |  "
            f"Text Encoder: {c.text_encoder_type}"
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def _initial_values(cfg: DictConfig) -> dict:
    """Extract sensible initial slider values from the loaded config."""
    vs = cfg.inference.video_size
    if isinstance(vs, (list, tuple, ListConfig)):
        h, w = vs[0], vs[1] if len(vs) > 1 else vs[0]
    else:
        h = w = int(vs)
    return dict(
        height=h,
        width=w,
        num_frames=cfg.inference.num_frames,
        steps=cfg.inference.infer_steps_video,
        cfg_scale=cfg.inference.cfg_scale_video,
        flow_shift=cfg.inference.infer_flow_shift_video,
        seed=cfg.inference.seed or 42,
    )


def _closest_resolution_preset(h: int, w: int) -> str:
    for label, (ph, pw) in RESOLUTION_PRESETS.items():
        if ph == h and pw == w:
            return label
    return "Custom"


def _closest_duration_preset(nf: int) -> str:
    for label, pf in DURATION_PRESETS.items():
        if pf == nf:
            return label
    return "Custom"


def build_app(server: InferenceServer) -> gr.Blocks:
    iv = _initial_values(server.config)

    resolution_choices = list(RESOLUTION_PRESETS.keys()) + ["Custom"]
    duration_choices = list(DURATION_PRESETS.keys()) + ["Custom"]

    with gr.Blocks(
        title="Hy-Nomas Video Generation",
        theme=gr.themes.Soft(),
        css="""
        .gen-btn { min-height: 52px !important; font-size: 15px !important; }
        #info textarea, #batch-info textarea {
            font-family: 'Courier New', Consolas, monospace !important;
            font-size: 13px !important; line-height: 1.5 !important;
        }
        .header-row { margin-bottom: 4px; }
        """,
    ) as app:
        # ---- header ----
        gr.Markdown("# Hy-Nomas — Video Generation")
        gr.Markdown(
            f"**{server.model_summary}**",
            elem_classes="header-row",
        )

        with gr.Tabs():
            # ============================================================
            # Tab 1: Interactive (single prompt / multi-prompt via |||)
            # ============================================================
            with gr.TabItem("Interactive"):
                with gr.Row(equal_height=False):
                    # ======== LEFT: controls ========
                    with gr.Column(scale=1, min_width=380):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder=(
                                "Plain text:  scene 1 ||| scene 2 ||| scene 3\n"
                                'JSON input:  {"events": [{"visual_description_long": "...", '
                                '"start_time": "00:00.000", "end_time": "00:05.000"}, ...]}'
                            ),
                            lines=5,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="(optional) blurry, low quality, distorted ...",
                            lines=2,
                        )

                        with gr.Group():
                            gr.Markdown("### Resolution")
                            res_preset = gr.Dropdown(
                                choices=resolution_choices,
                                value=_closest_resolution_preset(
                                    iv["height"], iv["width"]
                                ),
                                label="Preset",
                            )
                            with gr.Row():
                                sl_h = gr.Slider(
                                    128,
                                    1280,
                                    value=iv["height"],
                                    step=16,
                                    label="Height",
                                    interactive=True,
                                )
                                sl_w = gr.Slider(
                                    128,
                                    1280,
                                    value=iv["width"],
                                    step=16,
                                    label="Width",
                                    interactive=True,
                                )

                        with gr.Group():
                            gr.Markdown("### Duration")
                            dur_preset = gr.Dropdown(
                                choices=duration_choices,
                                value=_closest_duration_preset(iv["num_frames"]),
                                label="Preset",
                            )
                            sl_frames = gr.Slider(
                                1,
                                481,
                                value=iv["num_frames"],
                                step=4,
                                label="Number of Frames",
                                interactive=True,
                            )

                        with gr.Group():
                            gr.Markdown("### Generation Parameters")
                            with gr.Row():
                                sl_cfg = gr.Slider(
                                    1.0,
                                    20.0,
                                    value=iv["cfg_scale"],
                                    step=0.5,
                                    label="CFG Scale",
                                )
                                sl_steps = gr.Slider(
                                    1,
                                    100,
                                    value=iv["steps"],
                                    step=1,
                                    label="Steps",
                                )
                            with gr.Row():
                                sl_shift = gr.Slider(
                                    0.0,
                                    20.0,
                                    value=iv["flow_shift"],
                                    step=0.5,
                                    label="Flow Shift",
                                )
                                num_seed = gr.Number(
                                    value=iv["seed"],
                                    label="Seed  (-1 = random)",
                                    precision=0,
                                )
                        with gr.Accordion("Advanced: Dynamic CFG", open=False):
                            cb_dyncfg = gr.Checkbox(
                                value=False,
                                label="Enable Dynamic CFG",
                            )
                            with gr.Row():
                                sl_cfg_min = gr.Slider(
                                    1.0,
                                    10.0,
                                    value=1.0,
                                    step=0.5,
                                    label="CFG Min",
                                )
                                sl_cfg_max = gr.Slider(
                                    1.0,
                                    20.0,
                                    value=8.0,
                                    step=0.5,
                                    label="CFG Max",
                                )
                                sl_cfg_peak = gr.Slider(
                                    1,
                                    50,
                                    value=16,
                                    step=1,
                                    label="Peak Step",
                                )

                        with gr.Accordion("Advanced: Multi-Prompt & JSON", open=False):
                            gr.Markdown(
                                "**JSON input:** Paste a JSON object/array with "
                                "`events` directly into the Prompt field. "
                                "Timestamps (`start_time`/`end_time`) are auto-"
                                "converted to frame boundaries and `num_frames`."
                            )
                            with gr.Row():
                                json_desc_first = gr.Textbox(
                                    value="visual_description_long",
                                    label="desc_field_first",
                                    info="JSON field for first event",
                                )
                                json_desc_rest = gr.Textbox(
                                    value="description_change",
                                    label="desc_field_rest",
                                    info="JSON field for subsequent events",
                                )
                            json_max_events = gr.Slider(
                                1,
                                16,
                                value=4,
                                step=1,
                                label="Max Events (JSON mode)",
                            )
                            gr.Markdown("---")
                            txt_boundaries = gr.Textbox(
                                label="Manual Segment Boundaries (override)",
                                placeholder=(
                                    "Comma-separated frame numbers, e.g. "
                                    "0,49,121,241  (leave empty for auto)"
                                ),
                                lines=1,
                            )
                            gr.Markdown(
                                "_Manual boundaries override JSON timestamps. "
                                "Count should be `num_prompts + 1`._"
                            )

                        gen_btn = gr.Button(
                            "Generate Video",
                            variant="primary",
                            elem_classes="gen-btn",
                        )

                    # ======== RIGHT: output ========
                    with gr.Column(scale=1, min_width=380):
                        vid_out = gr.Video(label="Generated Video", height=480)
                        info_out = gr.Textbox(
                            label="Generation Info",
                            lines=14,
                            interactive=False,
                            elem_id="info",
                        )
                        gr.Markdown(
                            "_Requests are queued and processed one at a time.  "
                            "Your position in the queue is shown above the button._"
                        )

                # -- Example prompts --
                gr.Examples(
                    examples=[
                        [
                            "A golden retriever playing fetch on a sunny beach, "
                            "waves crashing in the background.",
                            "",
                        ],
                        [
                            "A timelapse of a flower blooming in a garden, "
                            "soft sunlight filtering through leaves.",
                            "blurry, low quality",
                        ],
                        [
                            "A cat sitting on a window sill watching rain fall "
                            "outside, cozy warm interior lighting."
                            "|||"
                            "The cat jumps down and walks across the room, "
                            "stretching lazily on a soft rug.",
                            "",
                        ],
                        [
                            '{"events": ['
                            '{"visual_description_long": "A futuristic cityscape at night with neon lights and flying cars, cinematic atmosphere.", '
                            '"start_time": "00:00.000", "end_time": "00:05.000"}, '
                            '{"description_change": "A flying car lands on a rooftop, doors open and a figure steps out into the rain.", '
                            '"start_time": "00:05.000", "end_time": "00:10.000"}'
                            "]}",
                            "",
                        ],
                    ],
                    inputs=[prompt, negative_prompt],
                    label="Example Prompts  (plain text, ||| multi-prompt, or JSON)",
                )

                # ---- Interactive callbacks ----
                def _on_res_preset(preset):
                    r = RESOLUTION_PRESETS.get(preset)
                    return (r[0], r[1]) if r else (gr.update(), gr.update())

                def _on_dur_preset(preset):
                    v = DURATION_PRESETS.get(preset)
                    return v if v else gr.update()

                res_preset.change(_on_res_preset, [res_preset], [sl_h, sl_w])
                dur_preset.change(_on_dur_preset, [dur_preset], [sl_frames])

                gen_btn.click(
                    fn=server.generate,
                    inputs=[
                        prompt,
                        negative_prompt,
                        sl_h,
                        sl_w,
                        sl_frames,
                        sl_cfg,
                        sl_steps,
                        sl_shift,
                        num_seed,
                        cb_dyncfg,
                        sl_cfg_min,
                        sl_cfg_max,
                        sl_cfg_peak,
                        txt_boundaries,
                        json_desc_first,
                        json_desc_rest,
                        json_max_events,
                    ],
                    outputs=[vid_out, info_out],
                )

            # ============================================================
            # Tab 2: Batch Generation (from JSON prompt file)
            # ============================================================
            with gr.TabItem("Batch Generation"):
                gr.Markdown(
                    "Upload a **JSON prompt file** (same format as benchmark "
                    "suites used by `hymm/inference.py`) to generate videos "
                    "for all entries in one go."
                )

                with gr.Row(equal_height=False):
                    # ======== LEFT: batch controls ========
                    with gr.Column(scale=1, min_width=380):
                        json_upload = gr.File(
                            label="JSON Prompt File",
                            file_types=[".json"],
                            type="filepath",
                        )

                        with gr.Group():
                            gr.Markdown("### Prompt Extraction Fields")
                            with gr.Row():
                                batch_desc_first = gr.Textbox(
                                    value="visual_description_long",
                                    label="desc_field_first",
                                    info="Description field for the first event",
                                )
                                batch_desc_rest = gr.Textbox(
                                    value="description_change",
                                    label="desc_field_rest",
                                    info="Description field for subsequent events",
                                )
                            batch_max_events = gr.Slider(
                                1,
                                16,
                                value=4,
                                step=1,
                                label="Max Events per Entry",
                                info="Truncate to this many events per JSON entry",
                            )

                        with gr.Group():
                            gr.Markdown("### Resolution & Duration")
                            with gr.Row():
                                batch_h = gr.Slider(
                                    128,
                                    1280,
                                    value=iv["height"],
                                    step=16,
                                    label="Height",
                                )
                                batch_w = gr.Slider(
                                    128,
                                    1280,
                                    value=iv["width"],
                                    step=16,
                                    label="Width",
                                )
                            batch_frames = gr.Slider(
                                1,
                                481,
                                value=iv["num_frames"],
                                step=4,
                                label="Number of Frames",
                            )

                        with gr.Group():
                            gr.Markdown("### Generation Parameters")
                            with gr.Row():
                                batch_cfg = gr.Slider(
                                    1.0,
                                    20.0,
                                    value=iv["cfg_scale"],
                                    step=0.5,
                                    label="CFG Scale",
                                )
                                batch_steps = gr.Slider(
                                    1,
                                    100,
                                    value=iv["steps"],
                                    step=1,
                                    label="Steps",
                                )
                            with gr.Row():
                                batch_shift = gr.Slider(
                                    0.0,
                                    20.0,
                                    value=iv["flow_shift"],
                                    step=0.5,
                                    label="Flow Shift",
                                )
                                batch_seed = gr.Number(
                                    value=iv["seed"],
                                    label="Seed (-1 = random)",
                                    precision=0,
                                )
                        batch_btn = gr.Button(
                            "Generate All Videos",
                            variant="primary",
                            elem_classes="gen-btn",
                        )

                    # ======== RIGHT: batch output ========
                    with gr.Column(scale=1, min_width=380):
                        batch_video = gr.Video(
                            label="Latest Generated Video",
                            height=360,
                        )
                        batch_files = gr.File(
                            label="All Generated Videos",
                            file_count="multiple",
                            interactive=False,
                        )
                        batch_info = gr.Textbox(
                            label="Batch Info",
                            lines=14,
                            interactive=False,
                            elem_id="batch-info",
                        )

                # ---- Batch callbacks ----
                batch_btn.click(
                    fn=server.generate_batch,
                    inputs=[
                        json_upload,
                        batch_desc_first,
                        batch_desc_rest,
                        batch_max_events,
                        batch_h,
                        batch_w,
                        batch_frames,
                        batch_cfg,
                        batch_steps,
                        batch_shift,
                        batch_seed,
                    ],
                    outputs=[batch_video, batch_files, batch_info],
                )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hy-Nomas Video Generation — Gradio Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python apps/gradio_app.py -e inference
  python apps/gradio_app.py -e inference --dit-weight /path/to/ckpt.safetensors
  torchrun --nproc_per_node=8 apps/gradio_app.py -e inference --port 8080 --share
""",
    )
    p.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment config name, e.g. inference",
    )
    p.add_argument(
        "--dit-weight",
        default=None,
        help="Override DiT checkpoint path",
    )
    p.add_argument(
        "--text-encoder-model",
        default=None,
        help="Hugging Face model ID or local model/tokenizer directory",
    )
    p.add_argument("--sp-size", type=int)
    p.add_argument("--tp-size", type=int)
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--share", action="store_true", help="Create public link")
    p.add_argument(
        "--max-queue",
        type=int,
        default=10,
        help="Maximum pending requests in queue",
    )
    return p.parse_args()


def main():
    args = parse_cli()

    # ---- config ----
    overrides: Dict[str, str] = {}
    if args.dit_weight:
        overrides["dit_weight"] = args.dit_weight

    if args.text_encoder_model:
        overrides["model.text_encoder.model_name"] = args.text_encoder_model

    if args.sp_size is not None:
        overrides["parallelism.sp_size"] = args.sp_size
    if args.tp_size is not None:
        overrides["parallelism.tp_size"] = args.tp_size
    logger.info(f"Experiment : {args.experiment}")
    logger.info(f"Overrides  : {overrides}")

    cfg = load_config(args.experiment, overrides)

    logger.info(f"Model type : {cfg.dit.model_type}")
    logger.info(f"DiT weight : {cfg.dit_weight}")
    logger.info(f"TE type    : {cfg.text_encoder_type}")
    logger.info(f"VAE type   : {cfg.vae_type}")
    logger.info(f"Rollout    : {cfg.inference.rollout_type}")

    # ---- distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    sp_size = (
        getattr(cfg.parallelism, "sp_size", 1) if hasattr(cfg, "parallelism") else 1
    )

    tp_size = getattr(cfg.parallelism, "tp_size", 1)
    from hymm.distributed.tensor_parallel import validate_parallel_sizes, validate_tensor_parallel_config
    model_parallel_size = validate_parallel_sizes(sp_size, tp_size, world_size)
    validate_tensor_parallel_config(cfg)
    if world_size != model_parallel_size:
        raise ValueError("Gradio hosts one replica: WORLD_SIZE must equal sp_size * tp_size")
    world_mesh = None
    if world_size > 1:
        import datetime as _dt
        from hymm.distributed.parallel_states import (
            ParallelDims,
            initialize_sequence_parallel_state_v2,
        )

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            sp=sp_size,
            tp=tp_size,
            pp=1,
            world_size=world_size,
        )
        dist.init_process_group(
            backend="nccl",
            timeout=_dt.timedelta(seconds=3600),
        )
        torch.cuda.set_device(local_rank)
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        initialize_sequence_parallel_state_v2(parallel_dims, world_mesh)
        logger.info(
            f"Distributed: rank={rank}/{world_size}, "
            f"local_rank={local_rank}, sp_size={sp_size}, tp_size={tp_size}"
        )

    # ---- device ----
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and world_size <= 1:
        torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    logger.info(f"Device     : {device}")

    # ---- pipeline ----
    server = InferenceServer(cfg, device, rank=rank, world_size=world_size, world_mesh=world_mesh)

    if rank == 0:
        # ---- gradio (rank 0 only) ----
        app = build_app(server)
        app.queue(max_size=args.max_queue, default_concurrency_limit=1)

        logger.info(f"Launching on {args.host}:{args.port}  (share={args.share})")
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
    else:
        # ---- worker loop (rank > 0) ----
        server.worker_loop()


if __name__ == "__main__":
    main()
