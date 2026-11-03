"""Validate an I2V profile without API calls or access to its source dataset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SUBJECT = "<|Subject1|>"
VIEWS = {"first-person", "second-person", "third-person", "third-person observer"}
STYLES = {"real-world footage", "animation/anime", "CG", "video game"}
FALLBACK = "Compared to the previous segment, the visible scene and composition transition continuously."
SEGMENT_KEYS = {"start_time", "end_time", "visual_description_long", "description_change", "contained_actions"}
FIRST_TURN_PROMPT = (Path(__file__).resolve().parent / "online_user_video_rewriter_first_turn.md").read_text()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{2}:[0-5]\d\.\d{3}", value),
            "timestamp_format")
    return int(value[:2]) * 60000 + int(value[3:5]) * 1000 + int(value[6:])


def check_action(action, index, start, end):
    require(set(action) == {"start_time", "end_time", "action_description"}, "action_schema")
    a_start, a_end = timestamp(action["start_time"]), timestamp(action["end_time"])
    require(start <= a_start < a_end <= end, "action_bounds")
    require(a_end - a_start >= 2000, "action_minimum_duration")
    description = action["action_description"]
    require(isinstance(description, str), "action_description_type")
    body = re.fullmatch(rf"<Action{index}>(.+)</Action{index}>[.!?]", description)
    require(body is not None, "action_wrapper")
    require(SUBJECT in description, "action_subject_binding")
    require(re.findall(r"<\|Subject\d+\|>", description)
            == [SUBJECT] * description.count(SUBJECT), "additional_subject")
    require("<Camera" not in description, "action_camera_leak")
    require(re.findall(r"</?Action\d+>", description)
            == [f"<Action{index}>", f"</Action{index}>"], "nested_action")
    require(not re.search(r"\b(?:he|she|his|her|him|hers|himself|herself)\b", description, re.I),
            "subject_pronoun")
    require(not re.search(r"\b(?:camera|viewpoint)\s+(?:moves|circles|pans|tilts|follows|tracks)",
                          description, re.I), "action_camera_prose")
    return a_end


def validate_turn(turn, first, previous_end):
    user, target = turn["user"], turn["assistant"]
    if first:
        require(set(user) == {"perspective_view", "user_prompt"}, "first_input_schema")
        require(user["perspective_view"] in VIEWS, "public_view_enum")
        require(isinstance(user["user_prompt"], str) and user["user_prompt"].strip(), "empty_user_prompt")
        require(set(target) == {"perspective_view", "Style", "subject_list", "segments"}, "first_output_schema")
        expected = "third-person follow" if user["perspective_view"] == "third-person" else user["perspective_view"]
        require(target["perspective_view"] == expected, "view_mapping")
        require(target["Style"] in STYLES, "style_enum")
        require(len(target["subject_list"]) == 1, "single_subject")
        subject = target["subject_list"][0]
        require(set(subject) == {"subject_id", "display_name", "subject_attribute"}, "subject_schema")
        require(subject["subject_id"] == SUBJECT, "subject_id")
        require(isinstance(subject["display_name"], str) and "<|" not in subject["display_name"], "display_name")
        attribute = subject["subject_attribute"]
        require(isinstance(attribute, str), "subject_attribute_type")
        declaration = re.search(
            rf"- if `perspective_view` is `{re.escape(expected)}`, append `([^`]+)`",
            FIRST_TURN_PROMPT,
        )
        require(declaration is not None, "viewpoint_contract")
        require(attribute.endswith(declaration.group(1).replace("display_name", subject["display_name"])),
                "viewpoint_declaration")
    else:
        require(set(user) in ({"subsequent_text_prompt"}, {"camera_control"}), "followup_input_schema")
        if "subsequent_text_prompt" in user:
            require(isinstance(user["subsequent_text_prompt"], str) and user["subsequent_text_prompt"].strip(),
                    "empty_followup_prompt")
        require(set(target) == {"segments"}, "followup_output_schema")
    require(isinstance(target["segments"], list) and len(target["segments"]) == 1, "single_segment")
    segment = target["segments"][0]
    require(set(segment) == SEGMENT_KEYS, "segment_schema")
    require(segment["start_time"] == previous_end, "segment_continuity")
    start, end = timestamp(segment["start_time"]), timestamp(segment["end_time"])
    require(start < end, "positive_duration")
    if first:
        require(segment["description_change"] is None, "first_delta_null")
        scene = segment["visual_description_long"]
        require(isinstance(scene, str) and 80 <= len(scene.split()) <= 200, "first_scene_length")
    else:
        require(segment["visual_description_long"] is None, "followup_visual_null")
        scene = segment["description_change"]
        require(isinstance(scene, str) and scene.startswith("Compared to the previous segment, "), "delta_prefix")
    require("<|Subject" not in scene and "<Action" not in scene, "scene_channel_isolation")
    require(not re.search(r"\b(?:he|she|his|her|him|hers)\b", scene, re.I), "scene_personal_pronoun")
    camera_ids = re.findall(r"<Camera(\d+)>", scene)
    require(camera_ids == [str(i+1) for i in range(len(camera_ids))], "camera_numbering")
    require(re.findall(r"</Camera(\d+)>", scene) == camera_ids, "camera_closing_tags")
    for camera_id in camera_ids:
        require(re.search(rf"<Camera{camera_id}>[^<>]+</Camera{camera_id}>[.!?]", scene), "camera_wrapper")
    if not first:
        require("<Camera" not in scene.split(".", 1)[0], "first_delta_scene_sentence")
    actions = segment["contained_actions"]
    require(isinstance(actions, list), "action_list_type")
    last_end = start
    for index, action in enumerate(actions, 1):
        last_end = check_action(action, index, last_end, end)
        if first and user["perspective_view"] == "first-person":
            require(not re.search(r"<\|Subject1\|>\s+(?:enters|walks|runs|stands|sits|approaches)\b",
                                  action["action_description"]), "external_first_person")
    if "camera_control" in user:
        control = user["camera_control"]
        require(isinstance(control, dict) and set(control) == {"durationMs", "keyEvents"}, "camera_input_schema")
        duration = control["durationMs"]
        require(type(duration) is int and duration > 0 and end - start == duration, "camera_duration")
        require(not camera_ids and "<Camera" not in scene, "camera_control_tags")
        require(not re.search(r"\b(?:camera|viewpoint|lens)\b", scene, re.I), "camera_control_prose")
        require(isinstance(control["keyEvents"], list), "key_events_type")
        active, last_t = set(), 0
        for event in control["keyEvents"]:
            require(set(event) == {"t", "code", "pressed"}, "key_event_schema")
            require(type(event["t"]) in (int, float) and last_t <= event["t"] <= duration, "key_event_time")
            require(event["code"] in {
                "KeyW", "KeyS", "KeyA", "KeyD", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"
            }, "key_event_code")
            require(type(event["pressed"]) is bool, "key_event_pressed")
            code = event["code"]
            require((code not in active) if event["pressed"] else (code in active), "key_pairing")
            active.add(code) if event["pressed"] else active.remove(code)
            last_t = event["t"]
        require(not active, "key_unreleased")
        review = turn.get("camera_review", {})
        require(review.get("status") == "reviewed_conservative", "camera_review")
        if review.get("decision") == "insufficient_evidence_fallback":
            require(scene == FALLBACK, "fallback_exact")
    return segment["end_time"]


def validate_bundle(bundle, directory):
    directory = Path(directory)
    require(bundle.get("schema_version") == "i2v_icl_v1", "bundle_schema")
    examples = bundle["examples"]
    require(len({e["example_id"] for e in examples}) == len(examples), "duplicate_example")
    require(Counter(e["example_id"][0] for e in examples)
            == {"A": 4, "B": 2, "C": 3, "D": 4, "E": 9, "F": 9}, "category_distribution")
    require(Counter(e["assistant"]["Style"] for e in examples if not e["history"])
            == {"real-world footage": 10, "video game": 3}, "style_distribution")
    turns = 0
    for example in examples:
        reference = (directory / example["reference_image"]).resolve()
        require(reference.is_relative_to(directory.resolve()) and reference.is_file(), "reference_path")
        require(hashlib.sha256(reference.read_bytes()).hexdigest()
                == example["source"]["first_frame_sha256"], "reference_hash")
        previous_end = "00:00.000"
        for index, turn in enumerate(example["history"] + [example]):
            try:
                previous_end = validate_turn(turn, index == 0, previous_end)
            except ValueError as error:
                raise ValueError(f"{example['example_id']} turn {index}: {error}") from error
            turns += 1
        for turn in example["history"]:
            require(len(turn["video_frames"]) == 3, "three_history_samples")
            for relative in turn["video_frames"]:
                frame = (directory / relative).resolve()
                require(frame.is_relative_to(directory.resolve()) and frame.is_file(), "history_image_path")
    return turns


def parse_example_blocks(text):
    result = {}
    headings = list(re.finditer(r"^### Example ([A-F]\d+): (.+)$", text, re.M))
    for index, heading in enumerate(headings):
        stop = headings[index+1].start() if index+1 < len(headings) else len(text)
        block = text[heading.end():stop]
        objects = [json.loads(m) for m in re.findall(r"```json\s*\n(.*?)\n```", block, re.S)]
        result[heading.group(1)] = objects
    return result


def validate_folder(directory):
    directory = Path(directory)
    required = {
        "online_user_video_rewriter_system_prompt.md", "online_user_video_rewriter_system_prompt.py",
        "online_user_video_rewriter_first_turn.md", "online_user_video_rewriter_intermediate_turn.md",
        "icl_user_dense_caption.md", "icl_action.md", "icl_examples.json", "icl_manifest.json",
        "validate_prompt_folder.py", "readme.md",
    }
    require(all((directory / name).is_file() for name in required), "required_profile_files")
    bundle = json.loads((directory / "icl_examples.json").read_text())
    turns = validate_bundle(bundle, directory)
    manifest = json.loads((directory / "icl_manifest.json").read_text())
    preserved = manifest["preserved_prompts"]
    original_parts = {
        "system_rules_sha256": (directory / "online_user_video_rewriter_system_prompt.md").read_bytes()
            .split(b"# In-Context Learning Examples", 1)[0],
        "first_turn_sha256": (directory / "online_user_video_rewriter_first_turn.md").read_bytes(),
        "intermediate_turn_sha256": (directory / "online_user_video_rewriter_intermediate_turn.md").read_bytes(),
    }
    for name, content in original_parts.items():
        require(hashlib.sha256(content).hexdigest() == preserved[name], "original_prompt_preserved:" + name)
    for relative, expected in manifest["file_sha256"].items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()) and path.is_file(), "manifest_path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "file_hash:" + relative)
    for name in ("online_user_video_rewriter_system_prompt.md", "icl_user_dense_caption.md"):
        parsed = parse_example_blocks((directory / name).read_text())
        expected = bundle["examples"] if name.startswith("online") else [
            e for e in bundle["examples"] if e["example_id"] in parsed
        ]
        require(len(parsed) == (31 if name.startswith("online") else 10), "markdown_example_count")
        for example in expected:
            objects = [obj for turn in example["history"] + [example]
                       for obj in (turn["user"], turn["assistant"])]
            require(parsed[example["example_id"]] == objects, "markdown_bundle_parity")
    action_text = (directory / "icl_action.md").read_text()
    action_objects = [json.loads(m) for m in re.findall(r"```json\s*\n(.*?)\n```", action_text, re.S)]
    require(len(action_objects) == 16, "action_excerpt_count")
    all_actions = [a for e in bundle["examples"] if not e["history"]
                   for a in e["assistant"]["segments"][0]["contained_actions"]]
    for user, action in zip(action_objects[::2], action_objects[1::2]):
        require(set(user) == {"perspective_view", "user_prompt"}, "action_excerpt_input")
        require(action in all_actions, "action_excerpt_source")
        index = int(re.match(r"<Action(\d+)>", action["action_description"]).group(1))
        check_action(action, index, 0, timestamp(action["end_time"]))
    images = list((directory / "icl_images").glob("*.jpg"))
    require(all(p.read_bytes().startswith(b"\xff\xd8\xff") for p in images), "jpeg_assets")
    return {
        "status": "pass", "version": manifest["version"], "examples": len(bundle["examples"]),
        "base_prompt_rules_preserved": True,
        "turns_checked": turns, "standalone_caption_examples": 10, "action_excerpts": 8,
        "reference_images": len({e["reference_image"] for e in bundle["examples"]}),
        "image_assets": len(images), "checksummed_files": len(manifest["file_sha256"]),
        "categories": dict(Counter(e["example_id"][0] for e in bundle["examples"])),
        "first_styles": dict(Counter(e["assistant"]["Style"] for e in bundle["examples"] if not e["history"])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(validate_folder(args.directory), indent=2))
