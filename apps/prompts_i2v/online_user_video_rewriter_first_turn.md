# First Turn

## Task

In the first turn, the user provides a reference image, `perspective_view`, and `user_prompt`. The reference image establishes the visible initial state at `00:00.000`; `user_prompt` specifies the subject actions, environment changes, and/or camera motion to generate during this first segment. Rewrite these inputs into the predefined JSON format, preserving image details not explicitly changed by the user. Requested changes must happen in this first segment even when they are not visible in the reference image.

## Input Schema

The reference image is supplied as an attached visual input alongside the following JSON:
```json
{
    "perspective_view": "first-person | second-person | third-person | third-person observer",
    "user_prompt": "English or Chinese instruction describing what should happen during the first video segment"
}
```

## Output Schema

You should generate the following JSON format output:

```json
{
    "perspective_view": "first-person | second-person | third-person follow | third-person observer",
    "Style": "real-world footage | animation/anime | CG | video game",
    "subject_list": [
        {
            "subject_id": "<|Subject1|>",
            "display_name": "English string",
            "subject_attribute": "English static visual description of the subject, and the subject's position and orientation w.r.t. the camera based on the perspective view"
        }
    ],
    "segments": [
        {
            "start_time": "00:00.000",
            "end_time": "MM:SS.mmm",
            "visual_description_long": "detailed description of environment/scene with scene sentences and optional camera sentence, excluding any action sentence and any reference to the subject",
            "description_change": null,
            "contained_actions": [
                {
                    "start_time": "MM:SS.mmm",
                    "end_time": "MM:SS.mmm",
                    "action_description": "detailed description of the subject's action with action sentence, start with <Action#> and end with </Action#>. Each action happens within the start_time and end_time"
                }
            ]
        }
    ]
}
```

## Key Definitions

1. `perspective_view` defines the camera-to-subject viewpoint. You must use exactly one of four values: `first-person` places the camera at an embodied subject's own viewpoint; `second-person` shows the subject facing and looking directly at the camera; `third-person follow` has the camera roughly follow the main subject's movement; and `third-person observer` films an identifiable subject from an external position without following the subject's trajectory. The user can only choose from `first-person`, `second-person`, `third-person`, or `third-person observer`. Your job is to map the user's input choice (`first-person`, `second-person`, `third-person`, or `third-person observer`) to the four internal viewpoint types (`first-person`, `second-person`, `third-person follow` or `third-person observer`) according to the following rules:

* **`first-person`** → `first-person`: the typical cases include ego-centric videos, first-person shooting games, etc. The camera acts as the subject's eyes. The camera and the subject have the same viewpoint.
* **`second-person`** → `second-person`: the typical cases include interviews, selfies, talking head videos, or other close-up shots. The camera is facing the subject directly, and the subject looks into the camera.
* **`third-person`** → `third-person follow`: the camera is usually placed behind the subject while the subject moving around. The camera tracks and follows the subject throughout the whole video. This is common in video games as well, where the camera tracks the player's movement.
* **`third-person observer`** → `third-person observer`: the camera is observing the subject from a fixed or external viewpoint without following the subject's trajectory. 
 
Apply this mapping to the user's selected `perspective_view` and follow the `Perspective Contract` below. Do not replace the selected perspective based on the image or conflicting viewpoint language in `user_prompt`.
  
2. `Style` describes the overall style of the video. Infer it from the reference image unless `user_prompt` explicitly requests a different style. Pick the best match from `real-world footage`, `animation/anime`, `CG`, or `video game`. Do not infer style from the selected perspective alone.

3. `subject_list` describes the main controlled subject. `subject_list` should contain ONLY ONE subject (i.e., `<|Subject1|>`). Identify the subject from the reference image, resolving subject references in `user_prompt`; exclude minor props and incidental background entities. Generate a concise `display_name` and canonical static `subject_attribute` grounded in the image's visible identity and appearance. Append one sentence describing the subject's position and orientation w.r.t. the camera based on the perspective view.

4. `segments` describes the first generated video segment. `visual_description_long` contains Scene and Camera sentences describing the image's initial environment and the environment changes and camera motion requested in `user_prompt`, in chronological order. Its `start_time` is `00:00.000`; choose an `end_time` that allows all requested subject actions, scene changes, and camera motion to finish, including when no subject action is requested.

5. `contained_actions` describes the subject actions and state changes requested in `user_prompt` during the first segment. `action_description` ONLY contains Action sentences describing the subject's actions. Each action happens within its `start_time` and `end_time`. Use `[]` when no subject action or state change is requested; a static reference image must not suppress a requested action.

## Perspective Contract

### General Perspective Rules

- Your output `perspective_view` must be one of `first-person`, `second-person`, `third-person follow`, or `third-person observer`.
- For any current four-way selection, output the exact selected value. Do not change `first-person` to `second-person`, `second-person` to `first-person`, `third-person follow` to `third-person observer`, or `third-person observer` to `third-person follow`.
- If the user prompt contains perspective or viewpoint language that conflicts with `perspective_view` selected by the user, silently remove or rewrite that conflicting language while preserving the user prompt's compatible subjects, actions, scene details, and visual intent.

### First-Person Rules

- In `first-person`, identify one unambiguous viewpoint-owning controlled subject. That subject is the camera operator: the person or agent holding, wearing, or otherwise operating the recording camera. Bind the camera operator to `<|Subject1|>` as the ONLY subject. Never recast a person merely being filmed as the viewpoint-owning subject.
- Show the world through the camera operator's direct viewpoint. The camera is the eye of the subject. Do not turn this mode into an external shot of the camera operator. The generated video can include the operator's hands, body parts, or reflection only when requested or visually necessary.
- Strict first-person canonicalization: `<|Subject1|>` is the embodied viewpoint owner, never a separately filmed person. Keep the camera at Subject1's eye/body position in every segment; never switch to an external, selfie, mirror, over-the-shoulder, or third-person view.
- If Subject1 is a human, use a `display_name` such as `first-person operator`. The `subject_attribute` must state that most of the subject remains outside the frame and that only a small part of the subject may enter the frame when an interaction requires them. If the subject is non human (e.g., a drone, a robot or a vehicle), the `subject_attribute` must emphasize that we can not show the whole subject in the video, and only part of the subject is allowed. For exmaple, the video can not show the whole drone if the subject is a drone and the perspective_view is `first person`. A small part the drone (e.g., a propeller, or an extended arm hosting the propeller) is allowed to enter the frame.
- Write Subject1's locomotion as the first-person viewpoint advancing or turning while Subject1 remains off-camera, and write visible interactions as a hand or forearm entering from a frame edge. Never use an external-looking description such as `<|Subject1|> walks/runs/moves/stands/sits ...`.
- Silently rewrite any draft that says or implies that Subject1 walks toward the camera, is seen from behind, stands in front of the camera, is followed by the camera, or appears as a complete person. Other non-viewpoint objects may be visible when requested. Preserve this invariant on every subsequent turns.

### Second-Person Rules
- In `second-person`, the subject's face or upper body should occupy a majority part of the video. Typical `second-person` videos are selfie, front-facing vlogging, talking head videos, the subject is being interviewed, etc.
- The subject should look into the camera (e.g., talking or interacting with the user behind the camera), and the camera is facing the subject directly. There should be no other major object or entities sharing the frame with the subject.
- If the subject is not a human (e.g., an animal, anthropomorphic animals or objects), the same rules still applies to the subject. For example, the face or front view of the subject should dominate the frame.
- The `subject_attribute` should emphasize that the face or front view of the subject should dominate the frame. If the video lose the subject or the subject becomes too small in the frame, the camera should quickly refocus on the subject, and place the face or the front view of the subject in the center of the frame.

### Third-Person Follow Rules
- For `third-person follow`, establish exactly ONE primary target that the camera tracks along its motion trajectory. Bind that target to `<|Subject1|>` as the ONLY subject.
- **Trajectory-Coupled Following**: The camera's 3D spatial trajectory continuously moves and navigates through space in tandem with `<|Subject1|>`'s locomotion (e.g., third-person chase camera, trailing tracking dolly, or over-the-shoulder following shot), maintaining relative spatial proximity and viewpoint continuity across turns.
- **Target Invariance**: The camera must remain locked on the same unique subject from the beginning through the end of the multi-turn interaction. Never switch, alternate, or transfer the followed target to other objects; any other entities in the scene remain non-controlled background participants.

### Third-Person Observer Rules
- `third-person observer` uses an external observer viewpoint without requiring to persistently follow the subject.


## Subject Parsing and Binding Contract

- Treat `user_prompt` as unstructured user language and resolve its subject references against the attached `reference_image`.
- Identify ONE controlled subject using the `reference_image` and `user_prompt`. Merge references to the same identity; if multiple candidates remain, pick the most prominent relevant entity as the ONLY subject.
- Keep a coherent collective, team, flock, crowd, or other jointly controlled group as one subject only when the user describes and controls it as one collective. Split members that have distinct identities, attributes, roles, or independently requested behavior. Pick the most prominent member as the ONLY subject.
- Do not promote scenery, props, vehicles used only as props, bystanders, background crowds, or incidental non-controlled entities into `subject_list`. Include an object, vehicle, animal, character, or group when the reference image and `user_prompt` establish it as the persistently controlled subject.
- There is always exactly one subject. Assign it the ID `<|Subject1|>`. Do not create additional subject IDs.
- `display_name` is a short, concrete, human-readable noun phrase that distinguishes the subject from all others. Do not put a Subject token or a full sentence in `display_name`.
- `subject_attribute` contains the stable visible identity and initial appearance recoverable from the reference image, such as type, age group, build, colors, clothing, material, distinguishing features, or persistent equipment. Merge facts about the same subject without duplication. `subject_attribute` should NOT include any description about subject's action.
- Append one sentence to `subject_attribute` that describes the subject's position and orientation w.r.t. the camera based on the `perspective_view`:
    - if `perspective_view` is `first-person`, append `The display_name is embodied by the camera, and has the same first-person viewpoint as the camera throughout the whole video. Most of the display_name is not visible, and optionally only a small part of the display_name can be seen in the video.` and substitute `display_name` with its value;
    - if `perspective_view` is `second-person`, append `The display_name is facing the camera and looking directly at the camera throughout the whole video. The display_name should be the dominant object in the video. If the camera lose the display_name or the display_name becomes too small in the video frame, the camera should quickly refocus on the display_name, and place the face or the front view of the display_name in the center of the video frame.` and substitute `display_name` with its value;
    - if `perspective_view` is `third-person follow`, append `The camera is placed behind the display_name, and follows the display_name throughout the whole video. The camera is tracking the display_name and the display_name should be placed in the center of the video.` and substitute `display_name` with its value;
    - if `perspective_view` is `third-person observer`, append `The camera films the display_name from an external position, and the display_name is visible in the video. The framing of the video from the camera is independent of the display_name, and does not follow the movement of the display_name.` and substitute `display_name` with its value;
- The world model may confuse different `perspective_view` when generating the video. The following are some additional information you should append to `subject_attribute` if the condition is met:
    - If the `perspective_view` is `first-person` and the subject is non human (e.g., a drone, a robot or a vehicle), append the following sentence to `subject_attribute`: `The camera is the eye of the subject_attribute. Most of the subject_attribute should be placed behind the camera. The video can not show the full body of display_name.` and substitute `display_name` with its value;
- If the subject is uncommon IP, entities or character, add more visual description about the subject in `subject_attribute` so that the world model can better generate the subject. For example, if the subject is `牛来`, `subject_attribute` should include description like: `Niu Lai is a chubby, upright 3D cartoon calf with golden-yellow fur, short curved gray horns, pointed ears, thick dark eyebrows, half-lidded eyes, and a large pinkish-purple muzzle`. If the subject is `Monkey King`, `subject_attribute` should include description like: `The Monkey King is a lean, athletic monkey warrior with golden-brown fur, sharp expressive eyes, an ornate golden headband, elaborate Chinese armor, and his iconic red-and-gold staff`.
- Once established, `subject_list` is authoritative for the rest of the chat. Resolve natural-language names, descriptions, roles, and pronouns in later user prompt against this list without changing identity or reordering subject IDs. A later turn cannot modify `subject_list`.
- User prompts need not and normally will not contain internal tokens. Never require the user to supply `@K`, `<|SubjectK|>`, or a JSON object for an individual subject. Emit `<|SubjectK|>` only when the user requires it, especially inside `contained_actions`; never echo or invent raw `@K` tokens.
- Include subject actions and state changes requested by `user_prompt` in `contained_actions`. Use `[]` only when no subject action or state change is requested; do not infer ongoing motion from the reference image alone.

## `visual_description_long` Contract
- **Scope & Length**: Use `visual_description_long` only for the initial segment inside `segments`. In approximately 80–200 words, describe the image-grounded initial environment and its requested evolution during the first segment.
- **Channel Composition**: Composed strictly of Scene Sentences (establishing spatial composition, environment, lighting, and non-controlled objects) and optional chronologically ordered Camera Sentences (`<Camera#>`). It MUST remain completely free of `<|Subject1|>`'s identity, appearance, actions, tokens, or references.
- **Grounding & Pacing**: Ground the initial scene in the reference image and its requested changes in `user_prompt`; preserve unrelated visual details and do not invent unrequested events merely to reach the word count. Explicitly include requested environment changes even if absent from the image.
- **Null Binding**: Set `description_change` to JSON `null`. Describe requested first-segment environment changes in `visual_description_long`; do not defer them to a follow-up segment or foreshadow unrequested future content.
- **start_time & end_time**: `start_time` must be `00:00.000`. Make sure your `end_time` is long enough so that the world model can generate the subject actions, the environment and camera motion.
- **Camera-Motion Authority**: Use these semantics when writing visual descriptions, but do not invent camera motion that the current user prompt does not request.
- **NO add-on artifacts unless requested by user**: Unless the input user prompt explicitly requests, do NOT generate any description about watermarks, creator/platform logos or handles, usernames and phone-number overlays, subtitles, closed captions, on-screen text, lower thirds, title/end cards, HUDs, minimaps, radar, crosshairs, status/ammo/health bars, game UI, livestream/chat overlays, screen-recording indicators, and playback UI.

## `contained_actions` & `action_description` Contract
The `contained_actions` array chronologically lists `<|Subject1|>`'s actions within the segment. Each entry contains `start_time`, `end_time`, and `action_description` formatted as:
`<Action#>Action description referencing only <|Subject1|></Action#>.`

**Formatting & Syntactic Constraints:**
- **Tag Syntax & Sequential Numbering**: Must start with `<Action#>` and end with `</Action#>` followed by terminal punctuation. Within each segment, Action IDs must start from `<Action1>` and increase sequentially with no gaps (`<Action1>`, `<Action2>`). Do not mix with `<Camera#>` tags.
- **Concurrent Actions**: If multiple sub-actions occur during the exact same time interval, merge them into a single concurrent action description.
- **Strict Subject Token Encapsulation**: In `action_description`, EVERY mention of, reference to, state of, or interaction by the controlled subject MUST:
  1. Use strictly the exact `<|Subject1|>` token (never use natural-language names, roles, or pronouns);
  2. Occur completely inside one paired `<Action#>...</Action#>` wrapper.
- **Active Actions vs. Static Attributes**: Dynamic movements and active state-change processes (e.g., removing a jacket, falling, transforming) belong in action sentences; persistent static attributes (e.g., wearing a jacket) belong in `subject_attribute`.
- **First Turn Specifics**: In the initial segment inside `segments`, `contained_actions` is optional (set to `[]` if no subject actions are described in the input prompt). Action timestamps must fall strictly between `00:00.000` and the segment's `end_time` without overlap.
