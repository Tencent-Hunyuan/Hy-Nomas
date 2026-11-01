# Hy-Nomas web demo

[参考图初始化中文说明](IMAGE_REFERENCE_CN.md)

Generate and extend videos with text or camera paths. Run commands from the repository root.

## 1. Prepare the environment and models

Follow the [Python environment setup](../README.md#environment-setup) (also available in [Chinese](../README_CN.md#环境配置)) and activate your environment. Complete the Python installation before installing frontend dependencies. You also need Node.js 22.x (≥22.12), npm, and compatible **FlashAttention 2, FlashAttention 3, and MagiAttention** builds. Keep the CUDA Toolkit and compiler available in the launch environment for JIT compilation.

The demo uses the FFmpeg executable provided by `imageio-ffmpeg`. A separate system FFmpeg installation is optional if this executable works; `ffprobe` is needed by some acceptance tests, not by the basic launch flow. Check it with:

```bash
node --version
npm --version
python -c "import imageio_ffmpeg, subprocess; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], check=True)"
```

Install frontend dependencies:

```bash
npm --prefix apps/frontend ci
```

On shared storage, removing or installing `node_modules` can take much longer than on local disk. You can install using copies of `package.json` and `package-lock.json` in a fresh local directory, then symlink `apps/frontend/node_modules` to the resulting directory. Preserve any existing directory before replacing it. Run the frontend build and launcher from this repository. Use a reachable npm registry if package downloads time out.

Download the model repository as described in [Model Assets](../README.md#model-assets). Set `Hy_Nomas_Model_Path` to that download directory once; the launch commands below reuse its value.

| Model | Files | Path setting |
|---|---|---|
| Hy-Nomas | Distilled DiT `.safetensors` | `--dit-weight` |
| PixelUMM decoder | Matching `.safetensors` | `model.vae.checkpoint` |
| Text encoder | Complete Qwen model and tokenizer | `model.text_encoder.model_name` (model ID or local directory) |

Check the downloaded model files:

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

test -s "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
test -s "${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
```

Transformers loads `Qwen/Qwen2.5-VL-7B-Instruct` by default, downloading files as needed and reusing the local cache. No text encoder path is required for normal startup.

## 2. Start the demo

### 2.1. Deploy and connect online prompt enhancement (required)

Online prompt enhancement (PE) is required for the demo. Start the Nomas-PE service before the gateway and GPU workers. The gateway must be able to reach it throughout the session.

On the Nomas-PE server, use an official SGLang Docker image that supports Qwen3.8 inference. Run the following commands inside the container. The provided launcher uses one node with eight GPUs by default:

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

export PE_MODEL_PATH="${Hy_Nomas_Model_Path}/PE"
export PE_MODEL=Nomas-PE
bash apps/serve_nomas_pe.sh
```

Nomas-PE uses its own GPU resources; account for these separately from the video-generation workers. Set `PE_TP_SIZE` and `PE_PORT` if your deployment requires different values. If using an existing service, confirm it is running and exposes `Nomas-PE`.

On the demo host, export the connection settings in the shell used to start the demo:

```bash
export PE_API_URL=http://<your-pe-server>:30000/v1/chat/completions
export PE_MODEL=Nomas-PE
# If the PE server requires authentication:
# export PE_API_KEY=<your-api-key>
```

The service must expose the model ID `Nomas-PE`, matching `PE_MODEL`. Check model discovery from the demo host:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${PE_API_KEY:-EMPTY}" \
  "${PE_API_URL%/chat/completions}/models"
```

Confirm that the response lists `Nomas-PE` before continuing. For the provided server launcher, enable authentication with `PE_API_KEY` and set the matching `PE_API_KEY` on the demo host.

### 2.2. Configure runtime paths

Keep the queue and logs in the same runtime directory. Reuse these settings when stopping the demo:

```bash
export RUNTIME_DIR="$PWD/.runtime/seg_ar_demo"
export JOB_DB="$RUNTIME_DIR/jobs.json"
export WORKER_LOCK="$RUNTIME_DIR/gpu_worker.lock"
export PORT=8081
```

The gateway also reads `PE_CREDENTIAL_FILE`, defaulting to `$RUNTIME_DIR/pe_gateway.env`. If that file exists, its values override the exported PE settings; keep it consistent with the service configured above. `PE_API_TIMEOUT` defaults to 300 seconds and `PE_API_RETRIES` to 3.

### 2.3. Launch the gateway and GPU workers

With `Hy_Nomas_Model_Path` from step 1 and the PE settings above exported, run:

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

bash scripts/inference/start_seg_ar_demo.sh \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  --sp-size 8 \
  --save-path ./outputs/seg_ar_demo \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
```

The launcher builds the frontend and starts services in the background using `configs/experiment/inference.yaml`. Open `http://localhost:8081` or `http://<server-ip>:8081` once the workers are ready.

For a separate instance, choose an unused `PORT` and separate runtime paths and output directory. The launcher reuses an existing gateway on the selected port; stop the previous instance before applying new model or PE settings.

### 2.4. Wait for readiness

```bash
curl "http://localhost:${PORT}/health"
```

Wait for `"worker_online": true` and `"worker_status": "ready"`. `"status": "ok"` only indicates that the gateway is running. Model loading and the first MagiAttention JIT compilation can take several minutes:

```bash
tail -f "$RUNTIME_DIR/worker_0.log" "$RUNTIME_DIR/gateway.log"
```

### 2.5. Single-GPU low-memory mode

To use one GPU for video generation, use the INT8 DiT and FP8 PixelUMM weights. Set `CUTLASS_PATH` to a CUTLASS checkout containing the Ada blockwise GEMM example (example 94), as described in [low-memory inference](../README.md#inference-with-low-mem-gpu), then replace the launch command in step 2.3 with:

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

export CUTLASS_PATH=/path/to/cutlass

bash scripts/inference/start_seg_ar_demo.sh \
  --single-gpu-low-mem-mode \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors" \
  --save-path ./outputs/seg_ar_demo \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors"
```

PE remains required. This selects `configs/experiment/single_gpu_low_mem_inference.yaml` for the gateway and workers, taking precedence over `EXPERIMENT` and explicit `experiment=...` selection. Other launch overrides still apply.

Inspect the selected configuration and topology without starting services:

```bash
NOMAS_PRINT_TOPOLOGY_ONLY=true bash scripts/inference/start_seg_ar_demo.sh \
  --single-gpu-low-mem-mode
```

## 3. Generate and continue a video

1. Enter **Subject Description**, **Environment Description**, and a perspective. Click **Enter world**.
2. Continue with a text instruction or **Camera path**.
3. Click **Download** to save the video, or **New world** to start over.

Continue in the same world. After restarting the GPU worker, create a new world. Validate at least one text-generated segment and one camera continuation, then download and play the cumulative MP4. This exercises FA3 KV-cache attention, MagiAttention, the PixelUMM decoder, and video encoding that an import or health check alone does not cover.

### Start with a reference image

**New world** clears the previous image, its caption, and world prompts, while
preserving perspective, model, thinking level, inference steps, and seed settings.
New world and browser refresh use the same empty world configuration.
Both cancel frontend generation/rollback polling and request release of the
old session; late responses cannot update the newly opened world. A page restored
from the browser back/forward cache also returns to empty setup. Browser-exit
release uses a keepalive request; server-side expiry remains the fallback when
the browser or network cannot deliver it.

Configure `model.vae.encoder_checkpoint` with the DCAE-V encoder weights when
launching the gateway and workers:

```bash
model.vae.encoder_checkpoint="${Hy_Nomas_Model_Path}/VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt"
```

Image captioning uses the local Nomas-PE endpoint configured with `PE_API_URL`
and `PE_MODEL` (default `Nomas-PE`). Set `PE_API_KEY` only if that service
requires authentication. K3 credentials are not needed for image uploads.
Image captioning uses the bundled I2V system prompt in
`apps/prompts_i2v/online_user_video_rewriter_system_prompt.md`.
The attached image is paired with `perspective_view` and a `user_prompt`
requesting the five-second initial state. Only its last 0.5 seconds is shown as
**T0-Reference Image**; the complete reference remains in the model cache. The **User prompt** instruction is
still submitted after the reference segment completes.

Optionally configure the K3 prompt-enhancement API in the gateway environment with
`K3_API_URL`, `K3_MODEL=kimi-k3`, and either `APP_ID`/`APP_KEY` or `K3_API_KEY`
(`AIHUB_API_KEY` is also accepted). `K3_API_PROVIDER` defaults to `ali` for app
credentials and `moonshot` for API-key authentication. The gateway can load
these settings from `$RUNTIME_DIR/k3_gateway.env` or `K3_CREDENTIAL_FILE`.
Keep credentials out of source control. The API URL must be supplied by your
deployment. This optional adapter uses your deployment's app/API-key authentication contract.

These K3 settings also add **Kimi-K3** to the **Prompt enhancer** selector in
the setup and world screens. Nomas-PE is the default, with thinking set to None.
You can select Kimi-K3 and change the thinking level for text and camera
prompt enhancement, including continuation with video history. Nomas-PE
remains available through `PE_API_URL`; the selection determines which service
receives each request. Image uploads use Nomas-PE captioning, and
**Enter world** reuses their saved caption without another PE call.
Both PE choices use the original video-rewriter system and turn prompts in
`apps/prompts/`. The image-captioning system prompt applies only to uploads.

Click **Upload image** beside **Perspective** and select an image to enter
image-reference mode automatically. The subject/environment fields are replaced
by the second-segment prompt while the image is uploaded and captioned. The
selected perspective stays editable. **Upload image** also replaces an existing
image. **Cancel upload** or **Remove image** returns to text-only mode and restores
your manual drafts. Cancelling the file picker does not change modes.

In image-reference mode, select your perspective and upload one JPEG, PNG, or WebP
(up to 10 MB). Image mode replaces the subject/environment fields with one
**User prompt** input. Nomas-PE derives the initial subject and environment
from the image in the background. Enter what should happen next, then choose
**Enter world** after captioning finishes. The demo first creates the five-second
image segment from the saved caption, then sends your instruction and that
segment's video history to the selected PE to generate segment 2. The second
request starts only after the reference segment succeeds. The perspective
selector stays editable, and its current value is authoritative when entering.
Image captions identify exactly one primary subject (`<|Subject1|>`); other
visible people and objects are described only as scene context.
Nomas-PE treats the upload as a representative video frame and writes a present-tense
video caption without describing it as static, frozen, or motionless.
Switching to text-only mode restores the manual subject/environment fields. The image
initializes a five-second segment (120 frames at 24 FPS). The encoder
preserves the complete image: uploads are resized proportionally into a
1280×720 frame with centered black padding where the aspect ratio differs.
The same fit-and-pad operation adapts the saved image to the encoder's frame
size, without cropping content or stretching it. At the default resolution,
DCAE-V uses 1280×704 internally because its spatial compression requires
multiples of 32. The prepared frame is duplicated into
120 pixel frames. The full five-second video is passed through DCAE-V once,
producing 30 clean latent frames (sigma `0`). The cache-write pass mixes them
with Gaussian noise once: `z_cache = (1 - sigma) * z + sigma * noise`, where
`inference.cache_noise_sigma` defaults to `0.206897`. One DiT forward at
timestep `206.897` saves the image-and-text KV cache. No Euler denoising step
runs for the image reference. PixelUMM decodes the original clean latents and
initializes the streaming timeline. The last 12 decoded frames (0.5 seconds)
are published as **T0-Reference Image**. The UI shows **Loading image reference into KV cache…**
while the complete five-second segment is encoded and cached. The user-prompt
segment follows as **T1 · TEXT**, starting at playback time 0.5 seconds.
Decoder correction frames can replace T0's visible tail without reintroducing
the hidden prefix. PE sees the reference evidence with its original model
timestamps, followed by continuation history. Later generated segments retain their
normal denoising schedule before the same cache-noising operation.

The encoder loads only for reference-image requests, on rank 0, and releases
its GPU memory afterward. Rank 0 temporarily moves the idle DiT and decoder
weights to CPU during encoding and restores their original devices afterward.
The first segment uses the saved Nomas-PE caption directly;
subsequent turns use the normal PE service and the saved session history.
The gateway stores normalized uploads and their caption JSON under
`outputs/seg_ar_demo/reference_images`. References can only initialize new
interactive worlds; they cannot replace the image in an existing session.

## Settings, logs, and shutdown

Append overrides to the launch command from step 2.3 or 2.5. The defaults below come from [inference.yaml](../configs/experiment/inference.yaml); the single-GPU preset inherits these four settings.

| Setting | Default | Example override | Effect |
|---|---|---|---|
| Resolution (height × width) | 720 × 1280 | `'inference.video_size=[480,832]'` | Generate a smaller preview with fewer spatial tokens. |
| Generation steps | 8 | `inference.infer_steps_video=6` | Use fewer denoising steps per segment. |
| Translation strength | 1.0 | `inference.trajectory_translation_scale=0.5` | Halve camera translation relative to the default. |
| Rotation strength | 2.5 | `inference.trajectory_rotation_scale=1.5` | Use gentler camera rotation. |

For example, start with a smaller preview resolution and six denoising steps, using the model paths and PE settings prepared above:

```bash
bash scripts/inference/start_seg_ar_demo.sh \
  --dit-weight "$DIT_WEIGHT" \
  --sp-size 8 \
  --save-path ./outputs/seg_ar_demo \
  model.vae.checkpoint="$PIXELUMM_CHECKPOINT" \
  'inference.video_size=[480,832]' \
  inference.infer_steps_video=6
```

These command-line values override the selected YAML preset and are forwarded to the gateway and GPU workers. Stop the existing instance before restarting with changed settings. The browser's generation-step and camera-strength selections override those defaults for each request. Worker-fixed settings are listed in the runtime reference below.

For a different port, set `PORT` before starting:

```bash
export PORT=8082
```

The example saves videos to `./outputs/seg_ar_demo`. View logs:

```bash
tail -f "$RUNTIME_DIR/worker_0.log" "$RUNTIME_DIR/gateway.log"
```

Stop:

```bash
bash scripts/inference/stop_seg_ar_demo.sh
```

Use the same `RUNTIME_DIR` for startup, logs, and shutdown.

## Optional: use multiple GPU machines

Complete the required PE setup in [step 2.1](#21-deploy-and-connect-online-prompt-enhancement-required) on the launch machine first. All hosts need the same repository path, model access, compatible environments, and SSH access without interactive login. Ensure remote `python` uses the correct environment; workers launch through `python -m torch.distributed.run`. Share runtime state, the job queue, and outputs across hosts.

Set `--sp-size` to host count × `NPROC_PER_NODE`. Example: two hosts with eight GPUs each. Replace the hosts and shared paths, including the model paths from step 1. Run once on the launch machine:

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

export NOMAS_WORKER_HOSTS=<host-0>,<host-1>
export NPROC_PER_NODE=8
export MASTER_ADDR=<host-0>
export RUNTIME_DIR=/shared/nomas/runtime
export JOB_DB="$RUNTIME_DIR/jobs.json"
unset WORKER_LOCK  # use the worker's host-local lock default

bash scripts/inference/start_seg_ar_demo.sh \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  --sp-size 16 \
  --save-path /shared/nomas/outputs \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
```

Stop from the launch machine using the shutdown command above.

## Runtime reference

<details>
<summary>Decoder, streaming, and cache behavior</summary>

The single-GPU flag selects the preset; it does not change worker-only streaming policy. PixelUMM CPU offload supports the online worker's streaming settings. Each streaming decode operation restores PixelUMM on the execution GPU, then returns its weights, latent buffer, and pending pixels to CPU, including when decoding fails. Preview tail replacement and rollback preserve the same decode timeline. The preset still requires valid model paths; available GPU memory and transfer latency depend on the selected resolution and session length.

Segment-wise VAE decoding, PixelUMM stable-prefix streaming, and interactive noise and KV-cache runtime settings are fixed in `scripts/inference/run_seg_ar_gpu_worker.sh`, outside the shared YAML; launch arguments and environment variables do not override those settings.

The shared inference preset disables both decoding options for offline use; the online worker always enables them. PixelUMM is selected by `model.vae.name=pixelumm`, with decoder settings in `configs/experiment/inference.yaml`; there is no separate `pixelumm_decoder.enabled` switch. The training VAE settings are declared directly in `configs/experiment/sft.yaml`.

The online worker enables `inference.seg_ar_segment_vae_decode=true` and
`model.vae.stable_prefix_streaming=true`. Streaming requires `model.vae.num_steps=1`.
`model.vae.tail_hold_latents=1` pads intermediate decode windows and holds back their
last four pixel frames until the next segment corrects them; the final segment is
published in full. Set this value to `0` to disable the tail fix.

`model.vae.cpu_offload=true` enables decoder CPU residency between decode operations.
The optional acceleration flags `model.vae.deepgemm_fp8`, `model.vae.cutlass_fp8`,
`model.vae.cute_attention`, and `model.vae.compile_decoder` default to `false`.
DeepGEMM requires its package and SM90; CUTLASS FP8 requires `CUTLASS_PATH` and SM89;
CuTe requires `nvidia-cutlass-dsl==4.8.0` and SM90. Select at most one FP8 backend.

Stable-prefix streaming controls decoding within a request independently of `preserve_kv_cache`, which retains KV and the PixelUMM timeline between session turns.

Streaming sessions using `temporal_context`, `uniform`, or `recent_mix_uniform` support undoing the latest segment. Rollback removes that segment's CPU KV archive, reselects the remaining history, and restores KV values, INT8 scales, masks, frame counters, noise state, and the PixelUMM timeline. Existing GPU capacity is reused; the valid history length can shrink. No old KV is requantized. `removed_cache_tokens` reports the removed full archived segment per layer, not the change in the sampled GPU history length. Undoing the only remaining segment releases the online session. Offline inference does not retain undo state. The `recent` policy has no full CPU archive and does not support this rollback mechanism.

The dry-run endpoint is optional: set `inference.prompt` or `DRY_RUN_SOURCE` to a JSON prompt file to enable it. It is not required for online generation.

`DIT_WEIGHT`, `SP_SIZE`, `SAVE_PATH`, and `DRY_RUN_SOURCE` environment overrides remain supported; explicit command-line overrides win.

</details>
