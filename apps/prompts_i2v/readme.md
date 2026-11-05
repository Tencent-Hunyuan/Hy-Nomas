# I2V Prompt Profile 20261103 (Minimal Changes)

This is a portable prompt folder, with 31 independent demonstrations selected
from accepted v20 I2V rewrites. All demonstrations use actual reference images;
continuations also include sampled frames from every required prior segment.
The folder requires no API keys, remote media URLs, or source-dataset mounts.

The rules retain the `i2v_pe` branch wording at commit `37ed096`, including its
subsequent subject-selection clarifications. The only input-schema adjustment
removes the textual `reference_image` field: the actual image is supplied as a
visual attachment beside `perspective_view` and `user_prompt`. The intermediate-
turn prompt is unchanged. Original A-F category headings and notes are retained,
with only history references adjusted to the bundled conversations. Example
viewpoint declarations use the original first-turn wording. No policy exception
for second-person glances is added; example answers resolve conflicting gaze
instructions according to the original perspective contract.

## Files

- `online_user_video_rewriter_system_prompt.md`: common I2V contract and 31 examples.
- `online_user_video_rewriter_first_turn.md`: first-turn image/input instructions.
- `online_user_video_rewriter_intermediate_turn.md`: text and camera continuations.
- `online_user_video_rewriter_system_prompt.py`: compatible text loader and actual
  multimodal example/first-turn request builders.
- `icl_examples.json`: canonical examples, prior conversations, and source lineage.
- `icl_user_dense_caption.md`: ten complete conversations from the same bundle.
- `icl_action.md`: eight source-grounded action excerpts, all at least two seconds.
- `icl_images/`: 22 reference images and ordered historical frame samples.
- `icl_manifest.json`: file checksums, distribution, source groups, and camera reviews.
- `validate_prompt_folder.py`: standalone schema, media, and contract validation.
- `validation_report.json`: validation results for this version.

## Contract

First turn: attach the reference image beside JSON containing exactly
`perspective_view` and `user_prompt`. Public `third-person` maps to output
`third-person follow`. Other public values remain unchanged.
Follow-up: exactly `subsequent_text_prompt`, or structured `camera_control`
containing `durationMs` and `keyEvents`. There is always one controlled Subject1.

Category counts are A=4, B=2, C=3, D=4, E=9, F=9. The initial examples retain
ten real-world and three video-game styles. Eligible image-recoverable data
cannot match every old object and duration exactly: the cart is replaced by a
male game traversal, engine fitting by body-panel fitting, and the resting seal
is in water. These are data-grounded replacements, not recreated reference images.

## Use

Existing callers can read the three Markdown files as their system/first-turn/
intermediate-turn profile. Callers implementing `PE_PROMPT_PROFILE_DIR` can point
that setting to this folder. This setting selects instructions only; the caller
must still send the reference image and correct first-turn JSON. It does not
convert legacy request-building code or reload a running process automatically.

Markdown image paths sent as text do not attach images. For a real image-grounded
ICL conversation:

```python
from apps.prompts_i2v.online_user_video_rewriter_system_prompt import (
    load_icl_conversation,
    build_first_turn_messages,
    build_subsequent_turn_messages,
)

example_messages = load_icl_conversation("E7")
request_messages = build_first_turn_messages(
    reference_image="reference.jpg",
    perspective_view="third-person",
    user_prompt="Walk forward and turn left.",
)
```

Pass either returned message list to your existing image-capable chat API.
Each example is an independent world; do not concatenate their histories into
one continuous dialogue. The demonstration is a complete labeled conversation,
not a live user request. `build_first_turn_messages` constructs the live request
with actual JPEG/PNG/WebP image data and the folder's turn instructions.
`build_subsequent_turn_messages` accepts the complete prior message history,
chronological frame paths for every past segment, and exactly one continuation
mode. It attaches actual past images and uses the intermediate-turn instructions.

Command-line validation and request export:

```bash
python validate_prompt_folder.py
python online_user_video_rewriter_system_prompt.py --example E7 --output /tmp/i2v_icl_E7.json
python online_user_video_rewriter_system_prompt.py --reference-image /path/reference.jpg --perspective-view third-person --user-prompt "Walk forward." --output /tmp/i2v_request.json
```

## Evidence And Limits

First user prompts and current camera events are unchanged accepted v20 data.
Assistant labels are repaired for image grounding, single-subject semantics,
viewpoint declarations, channel separation, and exact JSON/timestamp constraints.
Camera labels use current controls and past evidence only. No physical camera
speed or rotation calibration is assumed; insufficient evidence uses the exact
fallback. Current requested output frames are never supplied as input evidence.

Historical images are first/middle/last samples, not full-motion observations.
Contract and loading validation is not an end-to-end model-quality guarantee.
Reserve every source group listed in the manifest from evaluation to avoid
demonstration leakage. This folder does not change the running demo.
