"""Portable I2V profile contracts and actual multimodal request construction."""

import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "apps/prompts_i2v"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def loader():
    return module("portable_i2v_loader", PROFILE / "online_user_video_rewriter_system_prompt.py")


@pytest.fixture(scope="module")
def validator():
    return module("portable_i2v_validator", PROFILE / "validate_prompt_folder.py")


@pytest.fixture(scope="module")
def bundle():
    return json.loads((PROFILE / "icl_examples.json").read_text())


def test_full_portable_profile(validator):
    report = validator.validate_folder(PROFILE)
    assert report["examples"] == 31
    assert report["turns_checked"] == 50
    assert report["reference_images"] == 22
    assert report["action_excerpts"] == 8


def test_original_prompt_prose_preserved():
    # Pinned to 37ed096 with only the reference_image JSON field removed.
    system = (PROFILE / "online_user_video_rewriter_system_prompt.md").read_bytes()
    prefix = system.split(b"# In-Context Learning Examples", 1)[0]
    assert hashlib.sha256(prefix).hexdigest() == (
        "381a040690b2237836e35a18436aad54ff9382658531910ec25108fe407151bf"
    )
    hashes = {
        "online_user_video_rewriter_first_turn.md":
            "4acc30e964acb59b72a6d43807392637574f0a81f75622d8fac69f89666a0a60",
        "online_user_video_rewriter_intermediate_turn.md":
            "440bdd1be38805b1626c5d5e09cbd7c26f860db3b61a8541440632d679d940de",
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((PROFILE / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("example_id", [
    "A2", "A3", "A4", "A5", "B1", "B2", "C1", "C2", "C3", "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E5", "E6", "E7", "E8", "E9", "E10",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
])
def test_every_conversation_has_real_past_images(loader, bundle, example_id):
    example = next(e for e in bundle["examples"] if e["example_id"] == example_id)
    messages = loader.load_icl_conversation(example_id)
    assert "# In-Context Learning Examples" not in messages[0]["content"]
    users = [m for m in messages if m["role"] == "user"]
    assert len(users) == 1 + len(example["history"])
    for index, user in enumerate(users):
        images = [part for part in user["content"] if part["type"] == "image_url"]
        assert len(images) == (1 if index == 0 else 3 * index)
        expected_paths = ([example["reference_image"]] if index == 0 else
                          [frame for past in example["history"][:index] for frame in past["video_frames"]])
        for image, relative in zip(images, expected_paths):
            raw = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
            assert raw == (PROFILE / relative).read_bytes()
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.verify()
    assert json.loads(messages[-1]["content"]) == example["assistant"]


@pytest.mark.parametrize("mutation", [
    "legacy_input", "input_enum", "subject2", "short_action", "pronoun",
    "scene_subject", "camera_tag", "key_event", "continuity", "fallback",
])
def test_invalid_contract_rejected(validator, bundle, mutation):
    changed = copy.deepcopy(bundle)
    first = changed["examples"][0]
    camera = next(e for e in changed["examples"] if e["example_id"] == "F1")
    if mutation == "legacy_input":
        first["user"]["initial"] = {}
    elif mutation == "input_enum":
        first["user"]["perspective_view"] = "third-person follow"
    elif mutation == "subject2":
        first["assistant"]["subject_list"].append({"subject_id": "<|Subject2|>"})
    elif mutation == "short_action":
        first["assistant"]["segments"][0]["contained_actions"][0]["end_time"] = "00:01.000"
    elif mutation == "pronoun":
        action = first["assistant"]["segments"][0]["contained_actions"][0]
        action["action_description"] = "<Action1><|Subject1|> lifts her hand</Action1>."
    elif mutation == "scene_subject":
        first["assistant"]["segments"][0]["visual_description_long"] += " <|Subject1|>"
    elif mutation == "camera_tag":
        camera["assistant"]["segments"][0]["description_change"] += " <Camera1>The lens pans</Camera1>."
    elif mutation == "key_event":
        camera["user"]["camera_control"]["keyEvents"][0]["t"] = 999999
    elif mutation == "continuity":
        camera["assistant"]["segments"][0]["start_time"] = "00:00.000"
    elif mutation == "fallback":
        camera["assistant"]["segments"][0]["description_change"] += " New trees appear."
    with pytest.raises(ValueError):
        validator.validate_bundle(changed, PROFILE)


def test_live_first_turn_request(loader):
    messages = loader.build_first_turn_messages(PROFILE / "icl_images/C1_reference.jpg",
                                               "third-person", "Walk forward.")
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert json.loads(content[1]["text"]) == {
        "perspective_view": "third-person", "user_prompt": "Walk forward."
    }
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError):
        loader.build_first_turn_messages(PROFILE / "icl_images/C1_reference.jpg",
                                         "third-person follow", "Walk.")
    with pytest.raises(ValueError):
        loader.build_first_turn_messages(PROFILE / "icl_images/C1_reference.jpg",
                                         "third-person", " ")


def test_live_continuation_has_all_past_frames(loader):
    history = loader.load_icl_conversation("E7")
    frames = [
        [PROFILE / f"B2_missing_{index}"] for index in range(3)
    ]
    example = next(e for e in loader.load_icl_examples() if e["example_id"] == "E7")
    frames[:2] = [[PROFILE / p for p in past["video_frames"]] for past in example["history"]]
    frames[2] = frames[1]
    messages = loader.build_subsequent_turn_messages(
        history, frames, subsequent_text_prompt="Keep walking."
    )
    assert json.loads(messages[-1]["content"][1]["text"]) == {"subsequent_text_prompt": "Keep walking."}
    assert len([p for p in messages[-1]["content"] if p["type"] == "image_url"]) == 9
    with pytest.raises(ValueError):
        loader.build_subsequent_turn_messages(history, frames[:1], subsequent_text_prompt="Continue.")
    with pytest.raises(ValueError):
        loader.build_subsequent_turn_messages(
            history, frames, subsequent_text_prompt="Continue.",
            camera_control={"durationMs": 2000, "keyEvents": []}
        )
    messages = loader.build_subsequent_turn_messages(
        history, frames, camera_control={"durationMs": 542, "keyEvents": []}
    )
    assert json.loads(messages[-1]["content"][1]["text"])["camera_control"]["durationMs"] == 542


def test_portable_copy_and_tamper_detection(tmp_path, validator):
    destination = tmp_path / "renamed_profile"
    shutil.copytree(PROFILE, destination, ignore=shutil.ignore_patterns("__pycache__"))
    relocated = module("relocated_i2v", destination / "online_user_video_rewriter_system_prompt.py")
    assert relocated.load_icl_conversation("E7")[-1]["role"] == "assistant"
    assert validator.validate_folder(destination)["status"] == "pass"
    sample = next(destination.glob("icl_images/*_segment_*.jpg"))
    sample.write_bytes(sample.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="file_hash"):
        validator.validate_folder(destination)
    key = next(e["example_id"] for e in relocated.load_icl_examples()
               if any(str(sample.relative_to(destination)) in t["video_frames"] for t in e["history"]))
    with pytest.raises(ValueError, match="checksum"):
        relocated.load_icl_conversation(key)


def test_loader_rejects_path_escape_and_unknown_example(loader):
    with pytest.raises(ValueError):
        loader._image_block("../outside.jpg", PROFILE)
    with pytest.raises(KeyError):
        loader.load_icl_conversation("missing")


def test_first_inputs_and_controls_preserved(bundle):
    inputs = [
        {"example_id": e["example_id"], "users": [t["user"] for t in e["history"] + [e]]}
        for e in bundle["examples"]
    ]
    raw = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(raw).hexdigest() == (
        "c271dca6180c22edb448a389a0b04649d8b7191f625110366f7d99d1eb25f3a2"
    )
