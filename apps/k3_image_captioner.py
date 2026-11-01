"""Reference image captioning with local Nomas-PE or the K3 API."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

import requests
from PIL import Image
from apps.reference_image import REFERENCE_SECONDS, REFERENCE_VISIBLE_SECONDS

from apps.kimi_k3_prompt_enhancer import (
    ALLOWED_STYLES, USER_PERSPECTIVES, EnhancedPrompt, PromptEnhancementError,
    _extract_json, _retime_segment, validate_response,
)


CAPTION_INSTRUCTIONS = """Treat the uploaded image as a representative frame from
the first five-second video segment of an interactive world. Write a natural
video caption in the present tense. Infer the subject, environment, visual
style from the visual evidence. Use the user's selected camera perspective
exactly; never infer or replace it from the image. Describe the subject's
appearance, pose, expression, surroundings, lighting, and composition faithfully.
Do not describe the source as a photograph, a still image, a frozen frame, or
a repeated image. Do not add words or phrases such as static, motionless,
stationary, fixed camera, held still, remains unchanged, or no movement.
Do not make claims about the absence of motion or invent unseen events,
camera trajectories, or identities. Describe what the video shows.
Return a JSON object with exactly these keys:
{
  "perspective_view": "the user's selected perspective, copied exactly",
  "Style": "one of real-world footage, animation/anime, CG, video game, software/web/app/computer screen recording, slideshow",
  "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "short name", "subject_attribute": "detailed visible appearance"}],
  "segments": [{"start_time": "00:00.000", "end_time": "00:05.000", "visual_description_long": "detailed video scene, environment, lighting, subject pose, placement and composition", "description_change": null, "contained_actions": []}]
}
subject_list must contain exactly one subject, with subject_id <|Subject1|>.
Choose the single primary person or object by visual prominence and composition.
If there is no person, choose one primary visible object. Do not combine separate
people or objects into a group subject. Describe other visible people and objects
only as scene context in visual_description_long, without additional subject IDs.
Leave contained_actions empty; put the visual description in
visual_description_long. Output English captions and raw JSON only, without
Markdown fences.
"""


class K3ImageCaptioner:
    display_name = "K3"
    system_prompt = CAPTION_INSTRUCTIONS

    def _user_prompt(self, perspective):
        return (
            f"Describe the first {REFERENCE_SECONDS}-second video segment using this frame as visual evidence. "
            f"User-selected perspective_view: {perspective}. Copy this perspective exactly."
        )

    def __init__(self, *, session=None):
        self.api_url = os.environ.get("K3_API_URL", "")
        self.api_key = os.environ.get("K3_API_KEY") or os.environ.get("AIHUB_API_KEY", "")
        self.provider = os.environ.get("K3_API_PROVIDER", "moonshot" if self.api_key else "ali")
        self.model = os.environ.get("K3_MODEL", "kimi-k3")
        self.timeout = int(os.environ.get("K3_API_TIMEOUT", "300"))
        self.session = session or requests.Session()
        self.session.trust_env = False

    @property
    def configured(self):
        def valid(value):
            return bool(value and value.isascii() and not any(c.isspace() for c in value))
        return bool(self.api_url and (valid(self.api_key) if self.api_key else
                    valid(os.environ.get("APP_ID")) and valid(os.environ.get("APP_KEY"))))

    def _headers(self, image_id):
        if not self.configured:
            raise PromptEnhancementError("K3 image captioning needs K3_API_URL and valid APP_ID/APP_KEY or K3_API_KEY credentials")
        if self.api_key:
            auth = f"Bearer {self.api_key}?provider={self.provider}"
        else:
            auth = (
                f"Bearer {os.environ['APP_ID']}:{os.environ['APP_KEY']}"
                f"?provider={self.provider}&model={self.model}&timeout={self.timeout}"
                f"&max_retry=2&request_id={uuid.uuid4()}&cache_task_id=image-caption-{image_id}"
            )
        return {"Authorization": auth, "Content-Type": "application/json"}

    def _payload(self, messages):
        payload = {"model": self.model, "messages": messages, "reasoning_effort": "none", "max_tokens": 4096, "stream": True}
        if self.api_key:
            payload.pop("max_tokens")
            payload.update(max_completion_tokens=4096, response_format={"type": "json_object"}, temperature=1.0, top_p=0.95, stream=False)
        return payload

    def caption(self, image_path: Path, *, fps=24, perspective="third-person follow"):
        perspective = selected_perspective(perspective)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": self._user_prompt(perspective)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        payload = self._payload(messages)
        try:
            with self.session.post(self.api_url, headers=self._headers(image_path.stem), json=payload,
                                   timeout=(20, self.timeout + 30), stream=payload["stream"]) as response:
                if response.status_code != 200:
                    # Do not echo upstream bodies or request headers containing credentials.
                    if response.status_code in {401, 403}:
                        raise PromptEnhancementError(f"{self.display_name} rejected the API credentials; configure a valid key for the captioning endpoint")
                    raise PromptEnhancementError(f"{self.display_name} image captioning returned HTTP {response.status_code}")
                request_id = response.headers.get("X-Request-Id")
                if "text/event-stream" in response.headers.get("Content-Type", "").lower():
                    parts = []
                    for line in response.iter_lines(decode_unicode=True):
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if not line or not line.startswith("data:"):
                            continue
                        event = line[5:].strip()
                        if event == "[DONE]":
                            break
                        chunk = json.loads(event)
                        request_id = request_id or chunk.get("id")
                        for choice in chunk.get("choices") or []:
                            content = (choice.get("delta") or {}).get("content")
                            if isinstance(content, str):
                                parts.append(content)
                    raw = "".join(parts)
                else:
                    data = response.json()
                    request_id = request_id or data.get("id")
                    raw = data["choices"][0]["message"]["content"]
            parsed = _extract_json(raw)
            return normalize_caption(parsed, fps=fps, model=self.model, request_id=request_id, perspective=perspective)
        except requests.RequestException as exc:
            raise PromptEnhancementError(f"{self.display_name} image captioning connection failed ({type(exc).__name__})") from None
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PromptEnhancementError(f"{self.display_name} image captioning returned an invalid response") from exc


class NomasImageCaptioner(K3ImageCaptioner):
    """Use the local OpenAI-compatible PE endpoint, without K3 credentials."""

    display_name = "Nomas-PE"
    system_prompt = (
        Path(__file__).resolve().parent / "prompts_i2v" / "online_user_video_rewriter_system_prompt.md"
    ).read_text(encoding="utf-8").strip()

    def _user_prompt(self, perspective):
        return json.dumps({
            "perspective_view": perspective,
            "user_prompt": (
                f"Establish the initial {REFERENCE_SECONDS}-second segment from the reference image. "
                "Preserve its visible subject appearance and environment. "
                "No subject action, environment change, or camera motion is requested. "
                f"Use start_time 00:00.000 and end_time 00:0{REFERENCE_SECONDS}.000."
            ),
        })

    def __init__(self, *, session=None):
        self.api_url = os.environ.get("PE_API_URL", "http://127.0.0.1:30000/v1/chat/completions")
        self.api_key = os.environ.get("PE_API_KEY", "")
        self.model = os.environ.get("PE_MODEL", "Nomas-PE")
        self.timeout = int(os.environ.get("PE_API_TIMEOUT", "300"))
        self.session = session or requests.Session()
        self.session.trust_env = False

    @property
    def configured(self):
        return bool(self.api_url and self.model)

    def _headers(self, image_id):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages):
        return {
            "model": self.model, "messages": messages, "max_tokens": 4096,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }


def selected_perspective(value):
    if value not in USER_PERSPECTIVES:
        raise PromptEnhancementError("Choose a supported perspective")
    return "third-person follow" if value == "third-person" else value


def normalize_caption(parsed, *, fps=24, model="kimi-k3", request_id=None, perspective=None):
    if perspective is not None and isinstance(parsed, dict):
        parsed = {**parsed, "perspective_view": selected_perspective(perspective)}
    if not isinstance(parsed, dict) or parsed.get("perspective_view") not in USER_PERSPECTIVES:
        raise PromptEnhancementError("Image caption must identify a supported perspective")
    if parsed.get("Style") not in ALLOWED_STYLES:
        raise PromptEnhancementError("Image caption must identify a supported visual style")
    errors = validate_response(parsed, first=True, perspective=parsed["perspective_view"],
                               style=parsed["Style"], previous_end=None)
    subjects = parsed.get("subject_list")
    if not isinstance(subjects, list) or len(subjects) != 1:
        raise PromptEnhancementError("Image caption must contain exactly one primary subject (<|Subject1|>)")
    if any(not isinstance(s.get("display_name"), str) or not s["display_name"].strip()
           or not isinstance(s.get("subject_attribute"), str) or not s["subject_attribute"].strip() for s in subjects if isinstance(s, dict)):
        errors.append("Every subject needs a name and appearance description")
    if errors:
        raise PromptEnhancementError("Invalid image caption: " + "; ".join(errors))
    segment = _retime_segment(parsed["segments"][0], start_ms=0, duration_frames=fps * REFERENCE_SECONDS, fps=fps)
    if not segment["visual_description_long"].strip():
        raise PromptEnhancementError("Image caption is empty")
    segment["contained_actions"] = []
    parsed = {**parsed, "segments": [segment]}
    return {
        "caption_status": "ready", "caption_model": model, "request_id": request_id,
        "perspective": parsed["perspective_view"],
        "subject_description": subjects[0]["subject_attribute"],
        "environment_description": segment["visual_description_long"],
        "prompt": parsed, "num_frames": fps * REFERENCE_SECONDS, "fps": fps,
    }


def caption_to_enhanced(record, *, session_id, fps=24, perspective=None):
    record = normalize_caption(record["prompt"], fps=fps, model=record.get("caption_model", "kimi-k3"), request_id=record.get("request_id"), perspective=perspective)
    prompt = json.dumps(record["prompt"], ensure_ascii=False)
    user_message = {"role": "user", "content": json.dumps({
        "perspective_view": record["perspective"], "subject_description": record["subject_description"],
        "environment_description": record["environment_description"],
    }, ensure_ascii=False)}
    session = {
        "session_id": session_id, "requested_perspective": record["perspective"],
        "perspective": record["perspective"], "style": record["prompt"]["Style"],
        "subject": record["subject_description"], "environment_description": record["environment_description"],
        "subject_ids": [s["subject_id"] for s in record["prompt"]["subject_list"]],
        "previous_end": record["prompt"]["segments"][0]["end_time"],
        "segment_frame_lengths": [max(1, round(fps * REFERENCE_VISIBLE_SECONDS))],
        "reference_hidden_frames": fps * REFERENCE_SECONDS - max(1, round(fps * REFERENCE_VISIBLE_SECONDS)),
        "dit_prompts": [prompt], "messages": [user_message, {"role": "assistant", "content": prompt}],
        "video_url": None, "last_result": None, "result_history": [],
        "branch_revision": 0, "rollback_pending": False, "updated_at": time.time(),
    }
    return EnhancedPrompt(prompt=prompt, num_frames=fps * REFERENCE_SECONDS, segment_boundaries=[0, fps * REFERENCE_SECONDS],
                          session=session, raw_response=prompt, request_id=record.get("request_id"), user_message=user_message)
