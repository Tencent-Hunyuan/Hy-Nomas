<p align="left">
  <a href="README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>

<h1 align="center">Hy-Nomas IDM</h1>

**Hy-Nomas IDM** estimates camera motion from video. Built on the VPT visual backbone, it produces temporal action probabilities, zoom probabilities, velocity predictions, subject motion quality scores (`motion_score`), and clip-level camera motion predictions. The package includes model weights and an independent inference API for use with [Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas).

**Coming soon: a dedicated IDM for games, built on the same architecture.**

## Contents

- [Model Overview](#model-overview)
- [Camera Pose Estimation](#camera-pose-estimation)
- [Environment Setup](#environment-setup)
- [Inference](#inference)
- [Inputs and Outputs](#inputs-and-outputs)
- [Reproducing Metrics](#reproducing-metrics)
- [Package Layout](#package-layout)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Model Overview

| Property | Description |
|---|---|
| Backbone | VPT |
| Parameters | Approximately 0.474B (473,986,521) |
| Input | 144 RGB frames, 96 × 128 pixels (height × width) |
| Temporal output | 24 bins per window, one bin every 0.25 seconds |
| Predictions | 18 action probabilities, 2 zoom probabilities, 6 velocity values per bin |
| Motion Score | Predicts the subject's motion quality in the video using a trained regression head (`motion_score`) |
| Clip-level predictions | 15 camera motion classes for the entire input video (`clip_camera_motion`) |
| Clip-level head | 15 linear classifiers with 13 features each, 210 parameters, blend weight 0.3 |
| Model assets | `weights/idm.pt`, `weights/idm.json`, and `weights/clip_head.json` |

On CameraBench, temporal aggregation of frame-level predictions achieves **88.68% mAP**. Freezing the model, training a lightweight clip-level head on the CameraBench training set, and blending its predictions with the original predictions improves mAP to **89.19%**. Both results are clip-level macro mAP, evaluated with uniform sampling across each video over 969 videos, 15 classes, and 7,679 known labels. Both methods are included; enable them with `--sampling uniform` and `--sampling uniform --clip-head`, respectively.

### CameraBench Results by Category

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Params</th>
<th rowspan="2">Overall</th>
<th colspan="6">Translation (Dolly / Pedestal / Truck)</th>
<th colspan="2">Zooming</th>
<th colspan="6">Rotation (Pan / Tilt / Roll)</th>
<th rowspan="2">Static</th>
</tr>
<tr>
<th>In</th>
<th>Out</th>
<th>Up</th>
<th>Down</th>
<th>Right</th>
<th>Left</th>
<th>In</th>
<th>Out</th>
<th>Right</th>
<th>Left</th>
<th>Up</th>
<th>Down</th>
<th>CW</th>
<th>CCW</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Hy-Nomas IDM</strong></td>
<td align="center">0.474B</td>
<td align="center"><strong>88.68</strong></td>
<td align="center">95.46</td>
<td align="center">80.48</td>
<td align="center">70.49</td>
<td align="center">84.60</td>
<td align="center">88.54</td>
<td align="center">92.75</td>
<td align="center">80.10</td>
<td align="center">87.21</td>
<td align="center">94.77</td>
<td align="center">97.20</td>
<td align="center">92.12</td>
<td align="center">85.84</td>
<td align="center">95.95</td>
<td align="center">95.77</td>
<td align="center">88.84</td>
</tr>
<tr>
<td><a href="https://3dagentworld.github.io/CameraVLM/">CameraVLM</a></td>
<td align="center">9B</td>
<td align="center">88.20</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><a href="https://arxiv.org/html/2610.17242v1#S4">GPT-6 Astra (high)</a></td>
<td align="center">—</td>
<td align="center">82.50</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td><a href="https://linzhiqiu.github.io/papers/camerabench/">Qwen2.5-VL-72B (SFT)</a></td>
<td align="center">72B</td>
<td align="center">62.8</td>
<td align="center">86.8</td>
<td align="center">41.3</td>
<td align="center">30.5</td>
<td align="center">50.6</td>
<td align="center">70.7</td>
<td align="center">52.6</td>
<td align="center">57.1</td>
<td align="center">78.5</td>
<td align="center">81.9</td>
<td align="center">86.6</td>
<td align="center">79.1</td>
<td align="center">70.4</td>
<td align="center">35.0</td>
<td align="center">43.8</td>
<td align="center">76.6</td>
</tr>
<tr>
<td><a href="https://linzhiqiu.github.io/papers/camerabench/">Qwen2.5-VL-7B (SFT)</a></td>
<td align="center">7B</td>
<td align="center">60.0</td>
<td align="center">83.9</td>
<td align="center">38.6</td>
<td align="center">27.8</td>
<td align="center">47.8</td>
<td align="center">67.9</td>
<td align="center">50.0</td>
<td align="center">54.5</td>
<td align="center">75.8</td>
<td align="center">79.2</td>
<td align="center">83.8</td>
<td align="center">76.3</td>
<td align="center">67.6</td>
<td align="center">32.3</td>
<td align="center">41.0</td>
<td align="center">73.6</td>
</tr>
<tr>
<td><a href="https://linzhiqiu.github.io/papers/camerabench/">MegaSAM</a></td>
<td align="center">—</td>
<td align="center">58.2</td>
<td align="center">87.0</td>
<td align="center">58.3</td>
<td align="center">43.0</td>
<td align="center">48.4</td>
<td align="center">59.1</td>
<td align="center">58.0</td>
<td align="center">11.1</td>
<td align="center">10.2</td>
<td align="center">77.9</td>
<td align="center">82.4</td>
<td align="center">75.6</td>
<td align="center">57.7</td>
<td align="center">67.4</td>
<td align="center">76.9</td>
<td align="center">60.1</td>
</tr>
<tr>
<td><a href="https://linzhiqiu.github.io/papers/camerabench/">CUT3R</a></td>
<td align="center">—</td>
<td align="center">50.9</td>
<td align="center">88.0</td>
<td align="center">65.5</td>
<td align="center">38.7</td>
<td align="center">54.6</td>
<td align="center">42.5</td>
<td align="center">36.5</td>
<td align="center">15.9</td>
<td align="center">21.3</td>
<td align="center">59.1</td>
<td align="center">65.0</td>
<td align="center">65.0</td>
<td align="center">47.5</td>
<td align="center">60.7</td>
<td align="center">66.2</td>
<td align="center">37.6</td>
</tr>
<tr>
<td><a href="https://linzhiqiu.github.io/papers/camerabench/">MASt3R</a></td>
<td align="center">0.689B</td>
<td align="center">49.5</td>
<td align="center">65.4</td>
<td align="center">34.3</td>
<td align="center">35.1</td>
<td align="center">59.6</td>
<td align="center">43.7</td>
<td align="center">38.1</td>
<td align="center">42.2</td>
<td align="center">46.6</td>
<td align="center">66.6</td>
<td align="center">58.0</td>
<td align="center">63.2</td>
<td align="center">40.3</td>
<td align="center">50.4</td>
<td align="center">53.5</td>
<td align="center">45.6</td>
</tr>
</tbody>
</table>

Baseline data sources: [CameraBench leaderboard](https://sy77777en.github.io/CameraBench/leaderboard/table.html) and [CameraVLM, Table 1](https://arxiv.org/html/2610.17242v1#S4).

### Camera Pose Estimation

Hy-Nomas IDM reconstructs camera trajectories by integrating its predicted translational and angular velocities over six-second windows. ATE and translation RPE are measured in meters; rotation RPE is measured in degrees. Lower is better, and bold values mark the lowest result in each column within the Offline and Streaming groups.

<!-- BEGIN CAMERA POSE RESULTS -->
<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">ScanNet</th>
<th colspan="3">TUM RGB-D (dynamic)</th>
<th colspan="3">Sintel</th>
</tr>
<tr>
<th>ATE (m) ↓</th>
<th>RPE<sub>t</sub> (m) ↓</th>
<th>RPE<sub>r</sub> (°) ↓</th>
<th>ATE (m) ↓</th>
<th>RPE<sub>t</sub> (m) ↓</th>
<th>RPE<sub>r</sub> (°) ↓</th>
<th>ATE (m) ↓</th>
<th>RPE<sub>t</sub> (m) ↓</th>
<th>RPE<sub>r</sub> (°) ↓</th>
</tr>
</thead>
<tbody>
<tr><th colspan="10" align="left">Offline</th></tr>
<tr>
<td>DUSt3R-GA</td>
<td align="center">0.081</td>
<td align="center">0.028</td>
<td align="center">0.784</td>
<td align="center">0.083</td>
<td align="center">0.017</td>
<td align="center">3.567</td>
<td align="center">0.417</td>
<td align="center">0.250</td>
<td align="center">5.796</td>
</tr>
<tr>
<td>MASt3R-GA</td>
<td align="center">0.078</td>
<td align="center">0.020</td>
<td align="center">0.475</td>
<td align="center">0.038</td>
<td align="center">0.012</td>
<td align="center">0.448</td>
<td align="center">0.185</td>
<td align="center">0.060</td>
<td align="center">1.496</td>
</tr>
<tr>
<td>MonST3R-GA</td>
<td align="center">0.077</td>
<td align="center">0.018</td>
<td align="center">0.529</td>
<td align="center">0.098</td>
<td align="center">0.019</td>
<td align="center">0.935</td>
<td align="center">0.111</td>
<td align="center">0.044</td>
<td align="center">0.869</td>
</tr>
<tr>
<td>Fast3R</td>
<td align="center">0.155</td>
<td align="center">0.123</td>
<td align="center">3.491</td>
<td align="center">0.090</td>
<td align="center">0.101</td>
<td align="center">1.425</td>
<td align="center">0.371</td>
<td align="center">0.298</td>
<td align="center">13.750</td>
</tr>
<tr>
<td>FLARE</td>
<td align="center">0.064</td>
<td align="center">0.023</td>
<td align="center">0.971</td>
<td align="center">0.026</td>
<td align="center">0.013</td>
<td align="center">0.475</td>
<td align="center">0.207</td>
<td align="center">0.090</td>
<td align="center">3.015</td>
</tr>
<tr>
<td>VGGT</td>
<td align="center">0.035</td>
<td align="center">0.015</td>
<td align="center">0.380</td>
<td align="center"><strong>0.009</strong></td>
<td align="center"><strong>0.008</strong></td>
<td align="center">0.350</td>
<td align="center">0.172</td>
<td align="center">0.061</td>
<td align="center">0.470</td>
</tr>
<tr>
<td>π³</td>
<td align="center"><strong>0.031</strong></td>
<td align="center"><strong>0.013</strong></td>
<td align="center"><strong>0.347</strong></td>
<td align="center">0.014</td>
<td align="center">0.009</td>
<td align="center"><strong>0.312</strong></td>
<td align="center"><strong>0.074</strong></td>
<td align="center"><strong>0.040</strong></td>
<td align="center"><strong>0.282</strong></td>
</tr>
<tr>
<td>MapAnything</td>
<td align="center">0.052</td>
<td align="center">0.025</td>
<td align="center">0.720</td>
<td align="center">0.029</td>
<td align="center">0.023</td>
<td align="center">0.370</td>
<td align="center">0.226</td>
<td align="center">0.077</td>
<td align="center">0.640</td>
</tr>
<tr><th colspan="10" align="left">Streaming</th></tr>
<tr>
<td>StreamVGGT</td>
<td align="center">0.127</td>
<td align="center">0.041</td>
<td align="center">1.880</td>
<td align="center">0.062</td>
<td align="center">0.030</td>
<td align="center">0.690</td>
<td align="center">0.273</td>
<td align="center">0.109</td>
<td align="center">0.850</td>
</tr>
<tr>
<td>CUT3R</td>
<td align="center">0.096</td>
<td align="center"><strong>0.022</strong></td>
<td align="center"><strong>0.590</strong></td>
<td align="center"><strong>0.045</strong></td>
<td align="center">0.015</td>
<td align="center"><strong>0.440</strong></td>
<td align="center">0.215</td>
<td align="center">0.070</td>
<td align="center">0.630</td>
</tr>
<tr>
<td>Point3R</td>
<td align="center">0.097</td>
<td align="center">0.035</td>
<td align="center">2.791</td>
<td align="center">0.058</td>
<td align="center">0.031</td>
<td align="center">0.758</td>
<td align="center">0.442</td>
<td align="center">0.154</td>
<td align="center">1.897</td>
</tr>
<tr>
<td>Spann3R</td>
<td align="center">0.096</td>
<td align="center">0.023</td>
<td align="center">0.661</td>
<td align="center">0.056</td>
<td align="center">0.021</td>
<td align="center">0.591</td>
<td align="center">0.329</td>
<td align="center">0.110</td>
<td align="center">4.471</td>
</tr>
<tr>
<td>G2VLM</td>
<td align="center">0.148</td>
<td align="center">0.048</td>
<td align="center">1.220</td>
<td align="center">0.129</td>
<td align="center">0.044</td>
<td align="center">0.700</td>
<td align="center">0.301</td>
<td align="center">0.135</td>
<td align="center">1.450</td>
</tr>
<tr>
<td>Cambrian-P</td>
<td align="center"><strong>0.078</strong></td>
<td align="center">0.023</td>
<td align="center">0.880</td>
<td align="center">0.046</td>
<td align="center">0.020</td>
<td align="center">0.580</td>
<td align="center">0.239</td>
<td align="center">0.081</td>
<td align="center">2.440</td>
</tr>
<tr>
<td>CameraVLM</td>
<td align="center">0.124</td>
<td align="center">0.044</td>
<td align="center">2.177</td>
<td align="center">0.049</td>
<td align="center">0.022</td>
<td align="center">1.440</td>
<td align="center">0.148</td>
<td align="center">0.064</td>
<td align="center">1.420</td>
</tr>
<tr>
<td><strong>Hy-Nomas IDM</strong></td>
<td align="center">0.1673</td>
<td align="center">0.0236</td>
<td align="center">0.9824</td>
<td align="center">0.1110</td>
<td align="center"><strong>0.0126</strong></td>
<td align="center">0.9550</td>
<td align="center"><strong>0.0923</strong></td>
<td align="center"><strong>0.0393</strong></td>
<td align="center"><strong>0.5149</strong></td>
</tr>
</tbody>
</table>
<!-- END CAMERA POSE RESULTS -->

Baseline results and their Offline/Streaming groups are reported in [CameraVLM, Table 2](https://arxiv.org/html/2610.17242v1#S4.T2). The Hy-Nomas IDM row is measured with the released weights using velocity integration.

Evaluation follows the public [MonST3R / Cambrian-P protocol](https://github.com/cambrian-mllm/cambrian-p#pose-estimation-performance).

## Environment Setup

This operator is compatible with the [Hy-Nomas runtime environment](https://github.com/Tencent-Hunyuan/Hy-Nomas#environment-setup) and can reuse its Python, PyTorch, and CUDA installation.

For standalone use, the environment can be reduced to this package's `requirements.txt`. Use Linux, Python 3.10–3.13, and an NVIDIA GPU with BF16 support; install a CUDA-compatible PyTorch build (>=2.3), then install the dependencies below. Run the same command in an existing Hy-Nomas environment to add any missing dependencies:

```bash
python -m pip install -r requirements.txt
```

Only the five direct dependencies above are required; the action-space type definitions used by VPT are bundled. For `--hud`, also make FFmpeg available on `PATH`.

Keep `weights/`, `idm/`, and `vendor/` alongside `infer.py`. Paths are resolved relative to the package, so it can be moved without editing the code. Model weights occupy approximately 1.9 GB. They are configured for Git LFS; a source-only checkout must also obtain the corresponding LFS files.

## Inference

### Command Line

Run these commands from the package directory:

```bash
# Single video
python infer.py --video /path/to/input.mp4 --output outputs/input.json

# All MP4 files directly inside a directory
python infer.py --video /path/to/videos --output outputs/videos.json

# Explicit model and device
python infer.py --video /path/to/input.mp4 --output outputs/input.json \
  --weights weights/idm.pt --device cuda:0 --precision bf16

# Enable the clip-level head
python infer.py --video /path/to/input.mp4 --output outputs/input_clip.json \
  --sampling uniform --clip-head

# CPU inference
python infer.py --video /path/to/input.mp4 --output outputs/cpu.json \
  --device cpu --precision fp32
```

The default is `--sampling windows`: consecutive six-second windows sampled at 24 FPS, using CUDA BF16 autocast and FP32 model weights. Directory inputs are processed in filename order. The JSON output is updated atomically after each video; an existing output file is overwritten. Invalid inputs and incompatible weights raise errors. CPU inference can be slow; FP32 and different hardware or library versions can produce numerical differences.

### Prediction HUD

Add `--hud` to save both prediction JSON and an MP4 with the predictions overlaid:

```bash
python infer.py --video assets/case_01.mp4 --output outputs/case_01.json --hud --shake-threshold 0.7
```

This creates `outputs/case_01.json` and `outputs/case_01_hud.mp4`. Directory inputs
produce one HUD video per input, alongside the JSON. Install FFmpeg and ensure
`ffmpeg` is on `PATH`. OpenCV is included in the Python requirements.

The HUD displays action keys, eight mouse directions, zoom, and model-native
velocity values. SHAKE is active when its probability is **>= 0.7**; use
`--shake-threshold` to adjust both the JSON decision and HUD highlight.
`--rotation-threshold 0.5` controls exclusive rotation decisions and their HUD
highlight. `--hud-threshold 0.5` controls the other keys and zoom. Raw probabilities
and video scores are unchanged. The default HUD updates every 0.25 seconds.
SHAKE shows its predicted probability next to the highlight.
The rotation dial uses 20-degree cardinal sectors (center +/-10 degrees) and
70-degree diagonal sectors (center +/-35 degrees). Only the highest-probability
direction is highlighted when it reaches the rotation threshold; otherwise all sectors
remain inactive. Ties select the first direction in channel order.
Output uses H.264 MP4 at the source resolution and average FPS, with audio retained
when present (AAC). Variable-frame-rate input is rendered at constant average FPS.
An odd frame dimension is padded by one pixel for H.264 compatibility.

### Examples

<!-- BEGIN CASE VIDEOS -->

<table>
<tr><th>Case 01</th><th>Case 02</th></tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/7ca29e6e-ba03-4a29-8d43-e5baad75ed95

</td>
<td width="50%">

https://github.com/user-attachments/assets/8781ddb6-43eb-459e-8e27-3d8cba45732e

</td>
</tr>
<tr><th>Case 03</th><th>Case 04</th></tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/70bc84e1-8d63-4f94-ab0d-94be54967cee

</td>
<td width="50%">

https://github.com/user-attachments/assets/7edf6c02-4ee7-4b43-8ae6-911687afe797

</td>
</tr>
<tr><th>Case 05</th><th>Case 06</th></tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/bc08f78c-c4f8-4b44-9a90-09f0b6cfdb51

</td>
<td width="50%">

https://github.com/user-attachments/assets/058724b2-c65c-41d4-b618-84895b2ca2aa

</td>
</tr>
</table>

<!-- END CASE VIDEOS -->

### Python API

Make the package directory available on the Python import path:

```python
from idm.runtime import load_model, infer_video

model, metadata = load_model(device="cuda:0")
result = infer_video(model, "/path/to/input.mp4", shake_threshold=0.7, rotation_threshold=0.5)
print(result["clip_camera_motion"])
```

To use the clip-level head:

```python
from idm.clip_head import ClipHead

head = ClipHead()  # Loads weights/clip_head.json by default.
result = head.apply(infer_video(model, "/path/to/input.mp4", sampling="uniform"))
print(result["clip_camera_motion"])
```

## Inputs and Outputs

By default, the model processes consecutive **six-second windows at 24 FPS**. Decord decodes RGB frames at 96 × 128 pixels without cropping. Every six encoded frames produce one 0.25-second output bin. The final short window repeats the last frame for padding; only bins within the actual video duration are returned.

For video-level benchmark scoring, `--sampling uniform` samples 144 frames across the entire video using `rint(linspace(0, frame_count - 1, 144))`. This mode returns 24 bins for the whole video, with variable time spacing. The same mode is available through `infer_video(..., sampling="uniform")`.

Translation velocities use the model's native scale and camera-local coordinates; angular velocity targets use degrees per second. These regression outputs are not calibrated physical measurements. `sampling` records frame indices and, in window mode, window boundaries and bucket start times.

`clip_camera_motion` aggregates camera motion predictions across all valid windows of the entire input video; `window_motion_scores` records subject motion quality separately for each window.

Output JSON contains `model` metadata and a `samples` list. N is `ceil(duration_seconds × 4)` in window mode, or 24 in uniform mode:

| Field | Shape / meaning |
|---|---|
| `motion_prob` | N × 18 action probabilities; order in `model.label_names` |
| `zoom_prob` | N × 2 probabilities: zoom in, zoom out |
| `velocity6` | N × 6: right, down, forward, yaw right, pitch down, roll clockwise |
| `motion_score` | Subject motion quality score; averaged by valid duration in window mode |
| `window_motion_scores` | Subject motion quality scores for individual windows (window mode) |
| `shake_prob` | N values from action channel 15 (zero-based) |
| `shake_active` | N booleans: `shake_prob >= shake_threshold` |
| `shake_threshold` | SHAKE decision threshold, default 0.7 |
| `rotation_sector` | N exclusive direction indices: 0 right, 1 up-right, 2 up, 3 up-left, 4 left, 5 down-left, 6 down, 7 down-right; -1 inactive |
| `rotation_threshold` | Rotation decision threshold, default 0.5 |
| `clip_camera_motion` | Predictions for 15 camera motion classes across the entire video; blended predictions when the head is enabled |
| `clip_camera_motion_base` | Original temporal aggregation predictions when the head is enabled |
| `clip_camera_motion_head` | Standalone classifier predictions when the head is enabled |
| `clip_head` | Head weight checksum and blending metadata when enabled |
| `sampling` | Source FPS, frame count, duration, and sampled frame indices |

The 18 action channels are `w`, `a`, `s`, `d`; eight mouse directions (right, up-right, up, up-left, left, down-left, down, down-right); `special`, `up`, `down`, `shake`, `roll_clockwise`, and `roll_counterclockwise`.

Action and zoom logits are converted to probabilities before any aggregation. Pan and tilt use the maximum probability within their direction groups; a diagonal direction may support both. Motion and zoom scores use the maximum across time. Static is computed as `1 - max(motion directions, zoom, shake)` per bin, followed by the minimum across time. The `special` channel is not used as a Static score. These continuous scores are intended for ranking and analysis. The default temporal aggregation applies no velocity fusion or temporal smoothing. `shake_active` and `rotation_sector` are separate threshold-based decisions and do not alter the scores. The rotation decision selects only the most probable direction above or at the threshold, with the first channel winning ties; it is identical in JSON and the HUD.

## Reproducing Metrics

Run these commands from the package directory after setting up the environment.

### CameraBench: AP / mAP

Use the [benchmark videos](https://huggingface.co/datasets/syCen/Videos4CameraBench/tree/1ec8d55f8c380abbb6ea7087e64778f0ca76a68f) and [15 official label splits](https://github.com/linzhiqiu/t2v_metrics/tree/6ecb74f92028f42c7e64546d3a71e98c8c73068f/camerabench/data/binary_classification). The following commands reproduce the released **88.68% mAP** pipeline:

```bash
python infer.py --video /path/to/CameraBench/videos --sampling uniform \
  --precision bf16 --output outputs/camerabench_predictions.json
python tools/evaluate_camerabench.py \
  --predictions outputs/camerabench_predictions.json \
  --labels /path/to/t2v_metrics/camerabench/data/binary_classification \
  --output outputs/camerabench_metrics.json
```

Per-class AP and macro mAP are saved in `outputs/camerabench_metrics.json`. To recompute metrics from existing predictions, run only the second command.

**89.19% mAP:** apply the included clip-level head to those predictions, then score them without rerunning the video model:

```bash
python tools/apply_clip_head.py --predictions outputs/camerabench_predictions.json \
  --output outputs/camerabench_clip_predictions.json
python tools/evaluate_camerabench.py \
  --predictions outputs/camerabench_clip_predictions.json \
  --labels /path/to/t2v_metrics/camerabench/data/binary_classification \
  --output outputs/camerabench_clip_metrics.json
```

The head weights are stored in `weights/clip_head.json`; final predictions are `0.7 × temporal aggregation + 0.3 × head predictions`. Use `--sampling uniform --clip-head` during initial inference to produce these blended predictions directly.

### ScanNet / TUM-dynamic / Sintel: ATE / RPE

Prepare the datasets with the [MonST3R data scripts](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md). `--data-root` points to the directory containing `scannetv2/`, `tum/`, and `sintel/`.

```bash
python -m pip install -r requirements-eval.txt

# Predict trajectories and evaluate all three datasets
python tools/evaluate_pose.py --data-root /path/to/data \
  --datasets scannet tum sintel --methods velocity_only --output outputs/pose

# Recompute metrics from saved predictions without running the model
python tools/evaluate_pose.py --data-root /path/to/data \
  --metrics-only --predictions-root outputs/pose --output outputs/pose_metrics
```

For a single dataset, use `--datasets scannet`, `--datasets tum`, or `--datasets sintel`. For the classification-plus-velocity comparison, use `--methods classification_velocity`. Dataset metrics are saved in `outputs/pose/<dataset>/report.json`; the table results are under `summary.velocity_only.full_eligible_set_mean`.

## Package Layout

```text
README.md / README_CN.md    English / Chinese documentation
infer.py                    Command-line entry point
idm/                        Model loading, sampling, and score mapping
weights/                    Backbone, clip-level head weights, and metadata
vendor/video-pre-training/  VPT runtime and license
assets/                     Example videos, predictions, and HUD outputs
requirements.txt            Runtime dependencies
requirements-eval.txt       Optional pose evaluation dependencies
tests/                      Output, sampling, HUD, and trajectory tests
tools/apply_clip_head.py     Apply the clip-level head to saved predictions
tools/evaluate_camerabench.py CameraBench AP / mAP evaluation
tools/evaluate_pose.py      ScanNet, TUM-dynamic, and Sintel evaluation
tools/verify_package.py     Package checksum verification
```

## Acknowledgements

This package uses the [OpenAI VPT](https://github.com/openai/Video-Pre-Training) runtime. Its MIT license is preserved in `vendor/video-pre-training/LICENSE`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and distribution notes.

We also thank the [CameraVLM](https://3dagentworld.github.io/CameraVLM/) team for their valuable contributions to the community.

## Citation

If you find Hy-Nomas useful in your research, please cite our technical report:

```bibtex
@article{chen2026hy,
  title={Hy-Nomas: Towards Free-form Interactive World Generation},
  author={Chen, Qiaoling and Cheng, Jiaxiang and Deng, Xueqing and Hang, Tiankai and Hou, Ji and Li, Jiaqi and Liu, Yifan and Lu, Tao and Tang, Junshu and Tang, Yu and others},
  journal={arXiv preprint arXiv:2610.36729},
  year={2026}
}
```
