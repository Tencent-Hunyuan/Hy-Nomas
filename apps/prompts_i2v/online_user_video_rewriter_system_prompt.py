"""Portable text profile and image-grounded I2V request construction."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent
BASE_SYSTEM_PROMPT_PATH = PROMPT_DIR / "online_user_video_rewriter_system_prompt.md"
ICL_EXAMPLES_PATH = PROMPT_DIR / "icl_examples.json"
PUBLIC_VIEWS = {"first-person", "second-person", "third-person", "third-person observer"}


def load_system_prompt(base_path: Path = BASE_SYSTEM_PROMPT_PATH) -> str:
    prompt = Path(base_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"empty system prompt: {base_path}")
    return prompt


def load_icl_examples(icl_path: Path = ICL_EXAMPLES_PATH) -> list[dict]:
    data = json.loads(Path(icl_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "i2v_icl_v1":
        raise ValueError("unsupported ICL schema")
    return data["examples"]


def load_prompt_profile(directory: Path = PROMPT_DIR) -> dict[str, str]:
    """Match the existing system/first-turn/intermediate-turn Markdown profile."""
    directory = Path(directory)
    return {
        "system_prompt": load_system_prompt(directory / BASE_SYSTEM_PROMPT_PATH.name),
        "first_turn_prompt": (directory / "online_user_video_rewriter_first_turn.md").read_text(),
        "intermediate_turn_prompt": (
            directory / "online_user_video_rewriter_intermediate_turn.md"
        ).read_text(),
    }


def _image_block(relative_path: str, root: Path, expected_hash: str | None = None) -> dict:
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("ICL image is outside the prompt directory")
    raw = path.read_bytes()
    if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("ICL image checksum mismatch")
    return _raw_image_block(raw)


def _raw_image_block(raw: bytes) -> dict:
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("expected a JPEG, PNG, or WebP reference image")
    encoded = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _asset_hash(relative_path: str, root: Path) -> str:
    manifest = json.loads((root / "icl_manifest.json").read_text())
    try:
        return manifest["file_sha256"][relative_path]
    except KeyError as error:
        raise ValueError("unregistered ICL image: " + relative_path) from error


def load_icl_conversation(
    example_id: str,
    icl_path: Path = ICL_EXAMPLES_PATH,
    base_path: Path = BASE_SYSTEM_PROMPT_PATH,
) -> list[dict]:
    """Return one independent demonstration with real reference/history images."""
    icl_path = Path(icl_path)
    example = next((e for e in load_icl_examples(icl_path)
                    if e["example_id"] == example_id), None)
    if example is None:
        raise KeyError(example_id)
    rules = load_system_prompt(base_path).split("# In-Context Learning Examples", 1)[0].strip()
    messages = [{"role": "system", "content": rules}]
    for index, turn in enumerate(example["history"] + [example]):
        content = [{"type": "text", "text": json.dumps(turn["user"], ensure_ascii=False)}]
        if index == 0:
            content.append(_image_block(example["reference_image"], icl_path.parent,
                                        example["source"]["first_frame_sha256"]))
        else:
            for segment_index, past in enumerate(example["history"][:index]):
                segment = past["assistant"]["segments"][0]
                content.append({"type": "text", "text": (
                    f"Observed prior segment {segment_index}: "
                    f"{segment['start_time']} to {segment['end_time']}. "
                    "Ordered first, middle, last sampled frames; not continuous video.")})
                for label, frame in zip(("first", "middle", "last"), past["video_frames"]):
                    content.append({"type": "text",
                                    "text": f"Prior segment {segment_index}, {label} sampled frame"})
                    content.append(_image_block(frame, icl_path.parent,
                                                _asset_hash(frame, icl_path.parent)))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant",
                         "content": json.dumps(turn["assistant"], ensure_ascii=False)})
    return messages


def build_first_turn_messages(
    reference_image: Path,
    perspective_view: str,
    user_prompt: str,
    directory: Path = PROMPT_DIR,
) -> list[dict]:
    """Build an actual first-turn request; no legacy fields or placeholder images."""
    if perspective_view not in PUBLIC_VIEWS:
        raise ValueError("unsupported public perspective_view")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be non-empty")
    profile = load_prompt_profile(directory)
    user = {"perspective_view": perspective_view, "user_prompt": user_prompt.strip()}
    return [
        {"role": "system", "content": profile["system_prompt"]},
        {"role": "user", "content": [
            {"type": "text", "text": profile["first_turn_prompt"]},
            {"type": "text", "text": json.dumps(user, ensure_ascii=False)},
            _raw_image_block(Path(reference_image).read_bytes()),
        ]},
    ]


def build_subsequent_turn_messages(
    history: list[dict],
    prior_video_frames: list[list[Path]],
    *,
    subsequent_text_prompt: str | None = None,
    camera_control: dict | None = None,
    directory: Path = PROMPT_DIR,
) -> list[dict]:
    """Build a continuation with complete past conversations and frame evidence."""
    if (subsequent_text_prompt is None) == (camera_control is None):
        raise ValueError("supply exactly one continuation mode")
    if not history or history[-1].get("role") != "assistant":
        raise ValueError("history must end with the preceding assistant segment")
    turns = [json.loads(m["content"]) for m in history if m.get("role") == "assistant"]
    if len(prior_video_frames) != len(turns) or any(not frames for frames in prior_video_frames):
        raise ValueError("provide frame samples from every previous segment")
    if camera_control is None:
        if not isinstance(subsequent_text_prompt, str) or not subsequent_text_prompt.strip():
            raise ValueError("subsequent_text_prompt must be non-empty")
        user = {"subsequent_text_prompt": subsequent_text_prompt.strip()}
    else:
        if (set(camera_control) != {"durationMs", "keyEvents"}
                or type(camera_control["durationMs"]) is not int
                or camera_control["durationMs"] <= 0
                or not isinstance(camera_control["keyEvents"], list)):
            raise ValueError("invalid structured camera_control")
        active, last_t = set(), 0
        for event in camera_control["keyEvents"]:
            if (set(event) != {"t", "code", "pressed"} or type(event["pressed"]) is not bool
                    or type(event["t"]) not in (int, float)
                    or not last_t <= event["t"] <= camera_control["durationMs"]
                    or event["code"] not in {
                        "KeyW", "KeyS", "KeyA", "KeyD", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"
                    }):
                raise ValueError("invalid key event")
            code = event["code"]
            if (code in active) if event["pressed"] else (code not in active):
                raise ValueError("unpaired key event")
            active.add(code) if event["pressed"] else active.remove(code)
            last_t = event["t"]
        if active:
            raise ValueError("unreleased camera key")
        user = {"camera_control": camera_control}
    profile = load_prompt_profile(directory)
    content = [
        {"type": "text", "text": profile["intermediate_turn_prompt"]},
        {"type": "text", "text": json.dumps(user, ensure_ascii=False)},
    ]
    for index, (turn, frames) in enumerate(zip(turns, prior_video_frames)):
        segment = turn["segments"][0]
        content.append({"type": "text", "text": (
            f"Observed prior segment {index}: {segment['start_time']} to "
            f"{segment['end_time']}; ordered chronological samples, not continuous video.")})
        for sample_index, frame in enumerate(frames):
            content.append({"type": "text",
                            "text": f"Prior segment {index}, sample {sample_index}"})
            content.append(_raw_image_block(Path(frame).read_bytes()))
    return [
        {"role": "system", "content": profile["system_prompt"]},
        *[m for m in history if m.get("role") != "system"],
        {"role": "user", "content": content},
    ]


SYSTEM_PROMPT = load_system_prompt()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--perspective-view", choices=sorted(PUBLIC_VIEWS))
    parser.add_argument("--user-prompt")
    args = parser.parse_args()
    if args.example and args.reference_image:
        parser.error("choose either an ICL conversation or a first-turn request")
    if args.example:
        payload = json.dumps(load_icl_conversation(args.example), ensure_ascii=False, indent=2)
    elif args.reference_image:
        if not args.perspective_view or not args.user_prompt:
            parser.error("reference-image requires perspective-view and user-prompt")
        payload = json.dumps(build_first_turn_messages(
            args.reference_image, args.perspective_view, args.user_prompt
        ), ensure_ascii=False, indent=2)
    else:
        if args.perspective_view or args.user_prompt:
            parser.error("perspective-view and user-prompt require reference-image")
        payload = load_system_prompt()
    if args.output is None:
        print(payload)
    else:
        output = args.output.resolve()
        if output.is_relative_to(PROMPT_DIR.resolve()):
            parser.error("write generated requests outside the source prompt folder")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
