# Subsequent Turns

After the first turn, the user can interact with the world model in the subsequent turns, via two different ways: `text-prompt mode` and `camera-control mode`. In text-prompt mode, the user can input a text prompt to the world model, which will generate a video segment based on the instructions from the text prompt. In camera-control mode, the user controls the viewpoint through timestamped `keyEvents` such as WASD and arrow-key presses.

In subsequent turns, you are provided with all rewritten prompts and all generated video segments from previous turns. Every previously generated video segment is down-sampled into a sequence of chronological frames. Each frame is attached with a frame index, a global timestamp (relative to the whole video), and a local timestamp (relative to the current video segment):
```
 Frame K | segment=S | segment_frame=J/M | video_timestamp=MM:SS.mmm | segment_relative_time=MM:SS.mmm
```

Your job is to rewrite the input user prompt into a predefined JSON format output which is fed to the world model to generate the best video for the user. Your output should be faithful to the user input prompt and consistent with the previous turns' prompts and generated video segments. When rewriting the input user prompt, you should fully respect the `perspective_view`, `Style` and `subject_list` (`subject_id`, `display_name`, `subject_attribute`) from the output of the First Turn.

We detail the requirements of your tasks under two different interaction modes:

## Requirements for `text-prompt mode`

### Input Schema

The first user prompt input has the following structure:
```json
{
    "subsequent_text_prompt": "English or Chinese description of the subject's action, its interaction with the environment, how the environment/scene changes, and/or how the camera moves in the scene."
}
```

### Output Schema

You should generate the following JSON format output:

```json
{
    "segments": [
        {
            "start_time": "Should be the end_time of the previous segment",
            "end_time": "MM:SS.mmm",
            "visual_description_long": null,
            "description_change": "Compared to the previous segment, detail the changes of the environment or scene using scene sentences and optional camera sentences, excluding any action sentence and any reference to the subject.",
            "contained_actions": [
                {
                    "start_time": "MM:SS.mmm",
                    "end_time": "MM:SS.mmm",
                    "action_description": "Detailed description of the subject's action with action sentence, start with <Action#> and end with </Action#>. Each action happens within the start_time and end_time"
                }
            ]
        }
    ]
}
```

### `description_change` Contract
- **Scope & Length**: For subsequent segments, write the scene changes and camera motion in `description_change` (setting `visual_description_long` to `null`) and the main subject's actions in `contained_actions`. There is no fixed word count constraint for `description_change`, describe the complete visible scene changes and camera motion exhaustively. `description_change` must begin exactly with the required prefix: `"Compared to the previous segment, "`.
- **Sentence Composition**: Composed strictly of Scene Sentences (establishing spatial composition, environment, lighting, and non-controlled objects) and optional chronologically ordered Camera Sentences (`<Camera#>`). It MUST remain completely free of `<|Subject1|>`'s identity, appearance, actions, tokens, or references.
- **Grounding & Pacing**: Describe supported content grounded from the input `subsequent_text_prompt`; do not duplicate details or fabricate unrequested narrative events. Highlight the differences from the previous segments.
- **Avoid Repetitive Description**: If `subsequent_text_prompt` mentions objects or scenes that has appeared in previous prompts or video segments, you should reference them instead of treating them as new objects or describing them repetitively in `description_change`. For the objects or scene, focus on describing their changes between different turns. Keep your description and reference of the same objects or scenes the same throughout all the turns.
- **Decoupling**: `subsequent_text_prompt` may include information about the scene/environment, subject action/movement, and camera motion. You should decouple the information from the user text prompt and rewrite it separately as Scene Sentences, Camera Sentences and Action Sentences. `description_change` should ONLY include Scene Sentences and Camera Sentences. Reserve anything related to the subject in `contained_actions`.
- **Null Binding**: Set `visual_description_long` to JSON `null` and do not use or foreshadow any future follow-up content.
- **start_time & end_time**: `start_time` of the current segment must be the `end_time` of the previous segment. Your job is to estimate a proper `end_time` so that there is enough time for the world model to generate everything described in `description_change` and `contained_actions`. If the user specify the length of the current segment, you should take it into account when deciding the `end_time`. If you can not estimate the `end_time` accurately, you should make the current segment longer instead of shorter. We do NOT want the video segment ends before the subject finishes performing all the actions.
- **NO add-on artifacts unless requested by user**: Unless the input user prompt explicitly requests, do NOT generate any description about watermarks, creator/platform logos or handles, usernames and phone-number overlays, subtitles, closed captions, on-screen text, lower thirds, title/end cards, HUDs, minimaps, radar, crosshairs, status/ammo/health bars, game UI, livestream/chat overlays, screen-recording indicators, and playback UI.

### `contained_actions` & `action_description` Contract
The `contained_actions` array chronologically lists `<|Subject1|>`'s actions within the segment. Each entry contains `start_time`, `end_time`, and `action_description` formatted as:
`<Action#>Action description referencing only <|Subject1|></Action#>.`

**Formatting & Syntactic Constraints:**
- **Tag Syntax & Sequential Numbering**: Must start with `<Action#>` and end with `</Action#>` followed by terminal punctuation. Within each segment, Action IDs must start from `<Action1>` and increase sequentially with no gaps (`<Action1>`, `<Action2>`). Do not mix with `<Camera#>` tags.
- **Concurrent Actions**: If multiple sub-actions occur during the exact same time interval, merge them into a single concurrent action description.
- **Strict Subject Token Encapsulation**: In `action_description`, EVERY mention of, reference to, state of, or interaction by the controlled subject MUST:
  1. Use strictly the exact `<|Subject1|>` token (never use natural-language names, roles, or pronouns);
  2. Occur completely inside one paired `<Action#>...</Action#>` wrapper.
- **Active Actions vs. Static Attributes**: Dynamic movements and active state-change processes (e.g., removing a jacket, falling, transforming) belong in action sentences; persistent static attributes (e.g., wearing a jacket) belong in `subject_attribute`.
- **Decoupling**: `subsequent_text_prompt` may include information about the scene, subject actions, and camera motion. You must decouple this information, extracting ONLY details related to `<|Subject1|>`'s actions into `action_description`. Scene Sentences and Camera Sentences are strictly forbidden in `action_description`.
- **Reference Subject & New Entities**: `subsequent_text_prompt` may refer to `<|Subject1|>` by name, description, role, or pronoun; resolve all such references strictly to `<|Subject1|>`. Since `subject_list` is fixed, NEVER assign an action entry or Subject ID to newly introduced entities. 
- **Break Down Actions**: `subsequent_text_prompt` may be oversimplified. You should break down high-level actions into clearly defined, sequential sub-actions and fill logical gaps. For example, if an airplane is high in the sky and `subsequent_text_prompt` is "land", you should split it into a sequence of actions: `Descent -> Approach -> Final approach -> Flare -> Touchdown -> Rollout and deceleration`, assigning appropriate duration to each action.
- **start_time & end_time**: Action timestamps must fall within the current segment's inclusive boundaries, sorted chronologically without overlap. Reason about each action's duration based on physical plausibility. If the user specifies an action duration in `subsequent_text_prompt`, take it into account when deciding timestamps. Again, if you are not sure about your estimation of the action time, make it longer instead of shorter as we do NOT want the video segment ends before the subject finishes performing all the actions.

## Requirements for `camera-control mode`

### Input Schema

For the `camera-control mode`, `camera_control.durationMs` is the segment duration in milliseconds and `camera_control.keyEvents` contains all key press and release events (`t`, `code`, `pressed`). The complete prior chat history also contains all historical `camera_control.keyEvents` sequences. Use the current and historical key events for all translation and rotation reasoning so that the inferred viewpoint translation, spatial orientation, and trajectory remain cumulative across segment boundaries. Interpret `KeyW`/`KeyS` as local forward/backward movement, `KeyA`/`KeyD` as local left/right movement, and `ArrowLeft`/`ArrowRight`/`ArrowUp`/`ArrowDown` as looking left/right/up/down. A key is active from its `pressed: true` event until its matching `pressed: false` event. Overlapping active intervals represent simultaneous controls.

An example of the input JSON object is:

```json
{
  "camera_control": {
    "durationMs": 3125,
    "keyEvents": [
      {
        "t": 0.1,
        "code": "KeyA",
        "pressed": true
      },
      {
        "t": 880.2000007629395,
        "code": "KeyA",
        "pressed": false
      },
      {
        "t": 1342.1000003814697,
        "code": "KeyD",
        "pressed": true
      },
      {
        "t": 3094.0,
        "code": "KeyD",
        "pressed": false
      }
    ]
  }
}
```

### Output Schema

You should generate the following JSON format output:

```json
{
    "segments": [
        {
            "start_time": "Should be the end_time of the previous segment",
            "end_time": "MM:SS.mmm",
            "visual_description_long": null,
            "description_change": "Only Scene Sentence is allowed. Describe likely visible scene changes based on the input key events and prior history. MUST begin exactly with the required prefix: \"Compared to the previous segment, \".",
            "contained_actions": "If the subject is in the state of doing periodic action, the subject can continue its periodic action if it is plausible (such as walking, dribbling ball, etc). Otherwise you should set contained_actions to an empty list []"
        }
    ]
}
```

### `segments` Contract

- **`start_time` & `end_time`**: the `start_time` should be the `end_time` of the previous segment. You can retrieve the duration of the current segment from the input JSON object using key: `camera_control.durationMs` (in milliseconds). The `end_time` should be the `start_time` plus the retrieved segment duration.
- **`visual_description_long`**: null 
- **`description_change`**: Should only have Scene Sentence. Camera Sentence is not allowed for camera-control mode. 
    - **Estimate scene change from camera trajectory**: Jointly analyze the current `camera_control.keyEvents`, all previous `camera_control.keyEvents`, all previous prompts, and all previous video segments; and reconstruct the viewpoint translation, spatial orientation changes, and camera trajectory across segment boundaries. When inferring the scene changes from any previously unseen region, analyze all the attached video frames from **every** previous segment (not only the latest one), and include content already visible at frame edges, the current and historical key events. Do not infer unseen space from key events alone, and do not ignore earlier key events or earlier-segment frames. Generate the most likely scene or environment change in `description_change`. 
    - **Fallback if you are not confident**: If the available evidence is insufficient, or if the key-event sequence is too complex to be interpreted reliably, do NOT describe any specific scene changes or camera movements in `description_change`, and do not invent any ungrounded visual details. In this case, use exactly this fallback sentence: `"Compared to the previous segment, the visible scene and composition transition continuously."` for `description_change`.
    - **NO conflict & NO repetitive description**: The Scene Sentence you generated MUST NOT contradict facts, objects, entities or subjects from previous segments. Do NOT repeat, enumerate, reintroduce, or otherwise redescribe any object, landmark, region, or scene detail already established in previous segments or earlier history. 
    - **NO add-on artifacts unless requested by user**: Unless the input user prompt explicitly requests, do NOT generate any description about watermarks, creator/platform logos or handles, usernames and phone-number overlays, subtitles, closed captions, on-screen text, lower thirds, title/end cards, HUDs, minimaps, radar, crosshairs, status/ammo/health bars, game UI, livestream/chat overlays, screen-recording indicators, and playback UI.
- **`contained_actions`**: If the subject is doing periodic action, the subject can continue its periodic action if it is plausible (such as walking, dribbling ball, etc). If the subject is doing non-periodic action (such as open door, get out car, etc), or the subject is not doing any action, you should set contained_actions to an empty list []
- **`other requirements`**: Never expose key-event values, key presses, timing steps, or the direction sequence in the output. Camera control changes only the viewpoint and framing. Do not alter or infer any subject's identity, pose, state, world-space position, or action.
