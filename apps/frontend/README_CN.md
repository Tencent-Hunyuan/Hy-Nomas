# Hy-Nomas 前端

该目录包含 Hy-Nomas 的 React + TypeScript 网页前端。

页面目前通过同步 HTTP 接口调用：

```text
POST /v1/generate
```

后端由 `apps/seg_ar_web_server.py` 提供。启动参数 `--rollout-type` 支持：

- `seg_ar`：加载并常驻 `CPTCausalInferencePipeline`，模型文本来自网页提交的 prompt。
- `oneshot`：加载并常驻 `MultiPromptBidirectionalPipeline`，网页中的 `subject` 必须填写服务器可访问的 `.txt` 文件绝对路径；后端读取文件内容作为模型文本，不使用网页生成的 JSON/text prompt。

提交请求后，页面等待生成完成，再使用后端返回的 `video_url` 播放 MP4。

## 功能范围

当前页面支持：

- 选择视角和视觉风格
- 输入世界主体及环境描述
- 输入后续动作或场景指令
- 调用 `seg_ar` 后端生成视频
- 显示生成中、完成和失败状态
- 播放后端返回的视频

当前实现是同步的一次性视频生成：

```text
提交 prompt
  -> 等待 GPU 推理
  -> 后端编码 MP4
  -> 返回 video_url
  -> 浏览器播放
```

它还不是持续保留世界状态和 KV cache 的实时交互 loop。每次网页请求都会执行一次独立的 `seg_ar` 推理。

## 部署前端和后端

先按[模型资源](../../README_CN.md#模型资源)下载权重，并将 `Hy_Nomas_Model_Path` 设置为下载目录；下方启动命令会复用该值。

推荐在同一台 GPU 机器上部署前端和后端：

```text
浏览器
   |
   | HTTP
   v
seg_ar_web_server.py:8082
   |
   +-- /                 apps/frontend/dist 静态页面
   +-- /v1/generate      seg_ar 推理 API
   +-- /outputs/         生成的视频
   +-- /health           健康检查
```

`seg_ar_web_server.py` 已经包含静态网页托管能力。连接模型时，**不需要也不能同时在 8082 端口运行 `python -m http.server`**。

### 独立 Gateway + GPU Worker

Demo 统一通过以下入口启动 Gateway 和 GPU worker：

```bash
bash scripts/inference/start_seg_ar_demo.sh
```

启动前需在 `configs/experiment/inference.yaml` 中配置自己的模型文件，
也可以通过启动参数覆盖；详见 [Demo 部署说明](../README.md)。
Benchmark prompt 仅用于可选的 dry-run，不是正常在线生成的必需项。

- `apps/seg_ar_gateway.py`：默认监听 8081，可通过 `PORT` 覆盖，托管前端、API、输出文件和持久任务队列。
- `apps/seg_ar_web_server.py --worker-only`：常驻 GPU 模型，从队列领取任务；GPU 数由 `parallelism.sp_size` 和启动拓扑决定。
- 队列默认保存在 `/tmp/nomas_seg_ar_jobs.json`；GPU worker 重启时 Gateway 仍在线，未完成任务会重新排队。
- 每个 GPU worker slot 使用文件锁避免重复启动。

### 可选 Dry Run

通过同一个 demo 入口指定 benchmark prompt，即可启用 **Dry run** 按钮：

```bash
bash scripts/inference/start_seg_ar_demo.sh \
  inference.prompt=/path/to/benchmark.json
```

也可以在 `configs/experiment/inference.yaml` 中设置 `inference.prompt`，
或通过 `--dry-run-source /path/to/benchmark.json`、`DRY_RUN_SOURCE` 环境变量
指定。Worker 会从该文件随机选择样本，并在 segment 完成后更新视频和
caption；无需单独启动 HTTP 推理服务。

### Natural-language action PE

普通用户输入不再直接拼接成自然语言 DiT prompt。Gateway 会通过
选中的 PE 模型 API 将 Setup 中的 perspective、`Subject Description` 和
`Environment Description` 转换为训练同格式的
`perspective_view + Style + subject_list + segments + contained_actions`
JSON，再提交给 GPU worker。PE 保留用户选择的视角，并从描述中推断
Style。每个非首轮 PE 请求都会从所有已完成的 segment 中分别采样最多
40 张帧作为视觉连续性条件。每张帧保持原始宽高比，将短边下采样到 480
像素；短边已经不超过 480 像素时不会放大。完整的多模态 user message
会写入同一 session 的 conversation history，因此后续请求会回放此前各轮
积累的 segment 视频帧和 turn prompt，而不只是文本 JSON。请求内容按照
大规模 caption pipeline 的 multi-image 协议排列：先发送当前用户指令，
随后为每张图片发送对应的绝对 `video_timestamp` 与
`segment_relative_time` 文本，再发送图片。默认使用 Nomas-PE 和 None thinking，
用户可在选项中选择已配置的 Kimi-K3 和其他思考等级；
历史 assistant turn 只保留最终 JSON，不回放 thinking block。
Camera-control 轮会把为 PE 清理后的 trajectory JSON 作为
`camera_control` 值发给 K3，包括全部 pose frames、timestamps、position、
quaternion/Euler、duration、coordinate/scale metadata 和 keyEvents；
原始 trajectory 仍由 DiT 使用，不再使用 `<camera-control-stream>` 占位符。

Gateway 使用已启动的 SGLang OpenAI 兼容接口：

```bash
export PE_API_URL=http://localhost:30000/v1/chat/completions
export PE_MODEL=Nomas-PE
# 仅当服务需要 bearer token 时设置：
export PE_API_KEY=...
```

未设置 `PE_API_KEY` 时不会发送 Authorization 请求头。`PE_MODEL` 对应
SGLang 提供的首选默认模型 ID。Gateway 会向同一 API 地址的 `/v1/models`
查询可用模型，并在 Setup 和 Studio 中保留 **Prompt enhancer** 选择框，
即使只有一个模型也会显示。如果首选默认模型不在列表中，则默认使用列表
中的第一个模型。Thinking mode 和 LLM random seed 仍可逐请求调节，
请求失败不会回退到另一个模型。模型列表缓存 60 秒，查询失败后 5 秒可重试。
API 地址也支持服务根地址或 `/v1` 地址。
可通过 `PE_API_TIMEOUT` 和 `PE_API_RETRIES`
覆盖请求超时（默认 300 秒）和重试次数（默认 3 次）。

默认使用 `apps/prompts/` 下的 system、first-turn 和 intermediate-turn
Markdown prompt，并将每轮完整的多模态 user message 保留在后续 conversation
history 中。

SGLang 服务需单独启动；demo 不会将凭据写入队列、日志或 Git。

停止两个进程：

```bash
bash scripts/inference/stop_seg_ar_demo.sh
```

### 完整部署流程

#### 1. 构建前端

```bash
cd apps/frontend

# 第一次部署或 package-lock.json 变化时执行
npm ci --cache /tmp/nomas-frontend-npm-cache --prefer-offline --no-audit --no-fund

npm run build
```

检查构建产物：

```bash
test -f dist/index.html && echo "frontend build OK"
find dist -maxdepth 2 -type f
```

前端代码修改后，如果 `package-lock.json` 没有变化，不需要再次运行 `npm ci`，只需：

```bash
npm run build
```

#### 2. 准备后端环境

回到项目根目录，并进入可以正常运行 `hymm/inference.py` 的 Python 环境：

```bash
cd ../..
```

检查依赖：

```bash
python -c "
import torch
import hydra
import omegaconf
import fastapi
import uvicorn
import imageio
print('backend dependencies OK')
"
```

文本编码器和分词器默认通过 Transformers 加载以下模型：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

#### 3. 启动后端和前端

实验配置通过与 `hymm/inference.py` 相同的 Hydra 语法传入：

```text
+experiment=<相对于 configs/experiment 的配置路径>
```

单 GPU、`sp_size: 1`：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

python apps/seg_ar_web_server.py \
  +experiment=<experiment-config> \
  --rollout-type seg_ar \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.num_frames=49 \
  --host 0.0.0.0 \
  --port 8082
```

8 GPU、`sp_size: 8`：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --rollout-type seg_ar \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.num_frames=49 \
  --host 0.0.0.0 \
  --port 8082
```

这里显式设置：

```text
inference.num_frames=49
```

是因为 `configs/config.yaml` 的默认值为 1。如果前端请求没有显式携带 `num_frames`，不覆盖该配置会只生成 1 帧。

常用帧数（24 FPS）：

| 视频时长 | 建议帧数 |
| --- | ---: |
| 约 2 秒 | 49 |
| 约 5 秒 | 121 |
| 约 10 秒 | 241 |

如果需要显式指定文本编码器：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --rollout-type seg_ar \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.num_frames=49 \
  --host 0.0.0.0 \
  --port 8082
```

后端默认托管：

```text
<项目根目录>/apps/frontend/dist
```

因此可以从项目根目录启动，不要求当前目录为 `frontend/`。如需显式指定：

```bash
--frontend-dir apps/frontend/dist
```

#### 4. 等待服务开始监听

后端会先加载文本编码器、DiT 和 VAE。加载完成前，8082 端口可能尚未开放。

等待日志出现：

```text
Serving frontend from .../apps/frontend/dist
Starting seg_ar web server on http://0.0.0.0:8082
```

然后在另一个终端验证：

```bash
curl http://127.0.0.1:8082/health
curl -I http://127.0.0.1:8082/
curl http://127.0.0.1:8082/v1/config
```

`/v1/config` 中应确认：

```json
{
  "rollout_type": "seg_ar",
  "num_frames": 49
}
```

#### 5. 从浏览器访问

在服务所在机器的浏览器中访问：

```text
http://127.0.0.1:8082/
```

#### 6. 更新部署

仅修改前端时：

```bash
cd apps/frontend
npm run build
```

然后重启 `seg_ar_web_server.py`，确保浏览器使用最新构建。必要时在浏览器中强制刷新：

```text
Ctrl+Shift+R
```

修改后端时，也需要按“正确退出服务”章节停止现有进程，再重新启动。

### oneshot 文件 Prompt 模式

#### 模式说明

`oneshot` 是 Web Server 对外暴露的 rollout 名称，内部使用：

```text
MultiPromptBidirectionalPipeline
```

并映射到现有推理配置中的：

```text
one_shot
```

两种 Web rollout 的 prompt 来源不同：

| Web rollout | Pipeline | 模型文本来源 |
| --- | --- | --- |
| `seg_ar` | `CPTCausalInferencePipeline` | 网页请求的 `prompt` |
| `oneshot` | `MultiPromptBidirectionalPipeline` | 网页 `subject` 指向的 `.txt` 文件 |

一个服务进程只加载一种 pipeline。切换模式必须先正确退出服务，再使用新的 `--rollout-type` 重启。

#### 启动 oneshot 服务

8 GPU 示例：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --rollout-type oneshot \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.video_fps=24 \
  --host 0.0.0.0 \
  --port 8082
```

`oneshot` 不需要在启动命令中固定：

```text
inference.num_frames
```

请求未显式提供 `num_frames` 时，后端会读取 prompt 文件中的时间戳，并按 `inference.video_fps=24` 自动推导总帧数。

#### 网页端操作

1. 在 Setup 页面选择第一人称或第三人称。`oneshot` 后端不会使用该选项构造网页 prompt，但前端仍会显示它。
2. 在 `Prompt file` 输入框中填写后端服务器可访问的 `.txt` 绝对路径，例如：

   ```text
   /path/to/prompt.txt
   ```

3. 点击 `Enter world` 进入 Studio。
4. 当前 `Enter world` 只切换页面，不会立即发起生成。
5. 在右侧 `What should happen next?` 输入任意非空文本，例如：

   ```text
   Generate this video
   ```

6. 按 Enter 或点击发送按钮，前端才会调用：

   ```text
   POST /v1/generate
   ```

网页请求类似：

```json
{
  "prompt": "网页构造的文本；oneshot 后端不会使用它",
  "subject": "/path/to/prompt.txt"
}
```

当前前端不发送 `num_frames`。这正是 oneshot 自动时长模式所需的请求形式。

#### Prompt 文件要求

后端会对 `subject` 执行以下检查：

- 必须是以 `/` 开头的绝对路径。
- 必须是运行后端的服务器或容器可访问的文件。
- 文件必须存在并以 `.txt` 结尾。
- 文件必须为 UTF-8。
- 文件不能为空。
- 文件大小不能超过 2 MiB。

不要在 Prompt file 中添加引号或说明文字：

```text
# 正确
/path/to/prompt.txt

# 错误
"/path/to/prompt.txt"

# 错误
Prompt file: /path/to/prompt.txt
```

文件可以包含一个 JSON prompt，也可以使用 `|||` 分隔多个 JSON prompt：

```text
{"segments": [...]}|||{"segments": [...]}|||{"segments": [...]}
```

后端读取文件后执行：

```python
prompts = [
    part.strip()
    for part in file_content.split("|||")
    if part.strip()
]
```

读取文件发生在 rank 0。rank 0 随后将解析后的 prompt 和推理参数广播给其他序列并行 rank。

#### 自动帧数计算

固定 24 FPS 只能完成“时长到帧数”的换算，视频时长仍来自 prompt 文件中的时间戳。

对于每个 `|||` prompt，后端按以下顺序查找时间戳：

1. `segments[*].contained_actions[*].start_time/end_time`
2. `segments[*].start_time/end_time`

每段时长计算为：

```text
segment_duration =
    该段最大 end_time - 该段最小 start_time
```

随后按配置 FPS 累加：

```text
boundary_i = int(累计时长 × FPS)
num_frames = 最后一个 boundary
```

例如示例文件有三段：

```text
15.00 秒
15.00 秒
13.88 秒
```

在 24 FPS 下自动得到：

```text
pixel-frame boundaries = [0, 360, 720, 1053]
num_frames = 1053
总播放时长 = 1053 / 24 = 43.875 秒
```

后端将这些 pixel-frame boundaries 传给 `MultiPromptBidirectionalPipeline`。Pipeline 会再根据 VAE temporal ratio 和模型 temporal patch size 转换为内部 temporal-token boundaries。

如果请求显式携带：

```json
{
  "num_frames": 361
}
```

则显式值优先，不再从 prompt 时间戳推导总帧数。正常网页操作不需要传该字段。

如果 prompt 不是合法 JSON，或某一段没有可用的 `end_time`，后端会返回 HTTP 400。此时应修复 prompt 文件，或通过 API 显式提供 `num_frames`。

#### 检查服务配置

```bash
curl http://127.0.0.1:8082/v1/config
```

oneshot 模式预期返回：

```json
{
  "rollout_type": "oneshot",
  "prompt_source": "subject_file",
  "num_frames": null,
  "num_frames_source": "prompt_timestamps",
  "fps": 24
}
```

如果返回：

```json
{
  "rollout_type": "seg_ar"
}
```

说明服务没有使用 `--rollout-type oneshot` 启动，需要正确退出并重启。

#### API 测试

使用自动帧数：

```bash
curl -X POST http://127.0.0.1:8082/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "/path/to/prompt.txt"
  }'
```

显式覆盖帧数：

```bash
curl -X POST http://127.0.0.1:8082/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "/path/to/prompt.txt",
    "num_frames": 361
  }'
```

生成后的 metadata 会记录：

```json
{
  "rollout_type": "oneshot",
  "prompt_file": "/path/to/prompt.txt",
  "timing_source": "prompt_timestamps",
  "segment_durations_seconds": [15.0, 15.0, 13.88],
  "num_frames": 1053,
  "video_fps": 24
}
```

#### 常见错误

`oneshot subject path must be absolute`

: Subject 不是以 `/` 开头的绝对路径，或路径被引号/说明文字包裹。

`oneshot prompt file does not exist`

: 文件在后端容器中不可见，或者路径拼写错误。

`oneshot automatic frame selection requires every '|||' prompt to be JSON`

: 请求未传 `num_frames`，但文件中的某个 prompt 不是合法 JSON。

`has no usable end_time`

: 某个 prompt 没有 action 或 segment 的结束时间，无法自动推导时长。

`oneshot has ... temporal tokens; increase num_frames`

: 显式提供的 `num_frames` 太短，无法为所有 prompt segment 分配有效 temporal token。

## 单机托管快速开始

在一台机器上有两种托管方式：

| 方式 | 用途 | 服务命令 |
| --- | --- | --- |
| 仅托管静态前端 | 查看 UI，不运行模型 | `python3 -m http.server` |
| 前端和模型一体化托管 | 页面可调用 `seg_ar` 推理 | `apps/seg_ar_web_server.py` |

如果需要网页真正生成视频，应选择“前端和模型一体化托管”。

### 方式一：仅托管前端页面

该方式只用于查看前端设计。页面可以打开，但提交 prompt 时 `/v1/generate` 不可用。

先构建：

```bash
cd apps/frontend
npm ci
npm run build
```

在 `frontend/` 目录启动静态服务器：

```bash
python3 -m http.server 8082 \
  --bind 0.0.0.0 \
  --directory dist
```

服务器本机检查：

```bash
curl -I http://127.0.0.1:8082/
```

### 方式二：前端和模型一体化托管

该方式由同一个 `8082` 端口提供：

```text
/                 React 前端
/health           健康检查
/v1/config        默认推理参数
/v1/generate      seg_ar 生成接口
/v1/shutdown      优雅退出接口
/outputs/         生成的视频和元数据
```

第一步，构建前端：

```bash
cd apps/frontend
npm ci
npm run build
cd ../..
```

第二步，进入能够正常运行 `hymm/inference.py` 的模型环境，并设置模型目录：

```bash
```

第三步，根据实验的 `parallelism.sp_size` 启动。

单 GPU（`sp_size: 1`）：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

python apps/seg_ar_web_server.py \
  +experiment=<experiment-config> \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

8 GPU（`sp_size: 8`）：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

第四步，等待模型加载完成。只有日志出现以下内容后，HTTP 服务才开始监听：

```text
Starting seg_ar web server on http://0.0.0.0:8082
```

然后检查：

```bash
curl http://127.0.0.1:8082/health
ss -ltnp | grep ':8082'
```

如果机器端口不能直接访问，但可以 SSH 登录，可在本地电脑建立隧道：

```bash
ssh -N \
  -L 8082:127.0.0.1:8082 \
  <用户名>@<服务器地址>
```

然后在本地浏览器打开：

```text
http://127.0.0.1:8082/
```

### 后台保持运行

推荐使用 `tmux`：

```bash
tmux new -s nomas-web
```

在 tmux 中执行启动命令。按以下组合键只会 detach，不会停止服务：

```text
Ctrl+B
D
```

重新进入：

```bash
tmux attach -t nomas-web
```

停止服务时，请按照后文“正确退出服务”章节操作，不要逐个杀死分布式 rank。

## 环境要求

### Node.js

构建前端需要：

```text
Node.js 22
npm 10
```

检查：

```bash
node --version
npm --version
```

### Python

启动后端需要使用能够正常运行 `hymm/inference.py` 的完整模型环境，并包含：

```text
torch
hydra-core
omegaconf
fastapi
uvicorn
imageio
imageio-ffmpeg
```

检查：

```bash
python -c "
import torch
import hydra
import omegaconf
import fastapi
import uvicorn
import imageio
print('dependencies OK')
"
```

如果推理环境只缺 Web 和视频编码依赖，可以安装：

```bash
pip install \
  fastapi==0.115.3 \
  uvicorn==0.32.0 \
  imageio==2.36.0 \
  imageio-ffmpeg==0.5.1
```

## 构建前端

在项目根目录执行：

```bash
cd apps/frontend
npm ci
npm run build
cd ../..
```

构建产物位于：

```text
apps/frontend/dist/
```

Vite 使用相对资源路径，因此构建结果可以部署在域名根路径或反向代理子路径下。

前端代码修改后，需要重新执行：

```bash
cd apps/frontend
npm run build
```

## 启动 seg_ar 后端

### 实验配置参数

实验配置的选择方式与 `hymm/inference.py` 完全一致，使用 Hydra 参数：

```bash
+experiment=<相对于 configs/experiment 的配置路径>
```

例如：

```bash
+experiment=inference
```

也支持完整配置路径：

```bash
+experiment=configs/experiment/inference.yaml
```

配置解析器会自动移除 `configs/experiment/` 前缀和 `.yaml` 后缀。

不要使用：

```text
--experiment
-e
```

### DiT 权重

可以使用服务参数：

```bash
--dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
```

也可以使用与 `hymm/inference.py` 相同的 Hydra override：

```bash
dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
```

### 文本编码器路径

模型 ID 或本地目录与编码器类型分别配置：

```yaml
model:
  text_encoder:
    name: qwen-2.5vl-7b
    model_name: Qwen/Qwen2.5-VL-7B-Instruct
```

Transformers 按默认模型 ID 加载模型和分词器，按需自动下载并复用本地缓存。常规启动无需覆盖文本编码器参数。

## 单 GPU 启动

单 GPU 运行要求实验配置中：

```yaml
parallelism:
  sp_size: 1
```

启动：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

python apps/seg_ar_web_server.py \
  +experiment=<experiment-config> \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

## 多 GPU 序列并行启动

如果实验配置中：

```yaml
parallelism:
  sp_size: 8
```

启动：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=<experiment-config> \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

例如：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

当前服务只启动一个完整的序列并行模型副本，因此要求：

```text
WORLD_SIZE == parallelism.sp_size
```

rank 0 负责：

- 启动 FastAPI/uvicorn
- 接收网页请求
- 将请求广播给其他 rank
- 编码并返回视频

非零 rank 会进入常驻 worker loop，等待 rank 0 广播每个请求。

## Hydra 推理参数覆盖

服务自身不识别的参数会原样传给 Hydra，因此可以像 `hymm/inference.py` 一样追加覆盖项：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.video_size='[352,640]' \
  inference.num_frames=49 \
  inference.infer_steps_video=20 \
  inference.cfg_scale_video=5.0 \
  --host 0.0.0.0 \
  --port 8082
```

无论 YAML 中如何设置，服务都会强制使用：

```yaml
inference:
  rollout_type: seg_ar
  inference_chunk_size: 0
```

### 健康检查

在服务器上执行：

```bash
curl http://127.0.0.1:8082/health
```

正常响应示例：

```json
{
  "status": "ok",
  "busy": false,
  "rollout_type": "seg_ar",
  "rank": 0,
  "world_size": 8
}
```

查看默认推理参数：

```bash
curl http://127.0.0.1:8082/v1/config
```

## 正确退出服务

多 GPU 模式下，不能只结束 rank 0，也不要单独杀死某个 worker rank。非零 rank 正在等待 rank 0 的分布式广播；如果只杀其中一个进程，其他 rank 可能停在 NCCL collective 或广播中。

正确退出流程为：

```text
停止接收新请求
  -> 等待当前生成请求完成
  -> rank 0 广播 SHUTDOWN
  -> 非零 rank 退出 worker loop
  -> 所有 rank 释放 pipeline 和 CUDA cache
  -> 销毁 distributed process group
  -> torchrun 退出
```

### 前台运行时退出

如果服务直接运行在当前终端，使用：

```text
Ctrl+C
```

只按一次并等待日志出现：

```text
Graceful shutdown requested
Shutdown command delivered to worker ranks
[rank N] shutdown received
[rank N] model resources released
```

不要在正常退出过程中连续多次按 `Ctrl+C`，否则 torchrun 可能在优雅清理完成前强制结束 worker。

### 通过 HTTP API 退出

如需从另一个终端或管理页面停止服务，启动前设置一个 shutdown token：

```bash
export NOMAS_SHUTDOWN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "$NOMAS_SHUTDOWN_TOKEN"
```

然后正常启动服务。`seg_ar_web_server.py` 会从环境变量读取 token：

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  apps/seg_ar_web_server.py \
  +experiment=inference \
  --dit-weight "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  --host 0.0.0.0 \
  --port 8082
```

在服务器本机请求退出：

```bash
curl -X POST \
  -H "Authorization: Bearer ${NOMAS_SHUTDOWN_TOKEN}" \
  http://127.0.0.1:8082/v1/shutdown
```

响应：

```json
{
  "status": "stopping"
}
```

为避免 token 出现在 shell history 和进程列表中，推荐使用 `NOMAS_SHUTDOWN_TOKEN` 环境变量，不推荐直接写：

```bash
--shutdown-token <明文 token>
```

如果没有设置 `NOMAS_SHUTDOWN_TOKEN` 或 `--shutdown-token`，`POST /v1/shutdown` 会被禁用并返回 HTTP 404。

收到 shutdown API 后，服务会先返回 HTTP 202，再进入 uvicorn lifespan shutdown。若当时有生成任务正在执行，会等待该请求结束后再广播 `SHUTDOWN`，不会在 DiT/NCCL collective 中途释放模型。

### 在 tmux 中运行

创建会话：

```bash
tmux new -s nomas-seg-ar
```

在 tmux 中运行 `torchrun`，然后按：

```text
Ctrl+B
D
```

只会 detach，不会停止服务。

重新进入：

```bash
tmux attach -t nomas-seg-ar
```

停止时在 tmux 中按一次 `Ctrl+C`，或从其他终端调用 `/v1/shutdown`。

### 异常情况下强制退出

仅当优雅退出长时间没有响应时，才结束最外层 `torchrun` 进程，而不是逐个杀 rank：

```bash
ps -ef | grep '[t]orchrun.*seg_ar_web_server.py'
kill -TERM <torchrun-pid>
```

等待一段时间后检查：

```bash
ps -ef | grep '[s]eg_ar_web_server.py'
```

最后手段才使用：

```bash
kill -KILL <torchrun-pid>
```

强制退出可能跳过 CUDA、NCCL 和输出文件清理，因此不应作为常规停止方式。

## API

### 生成视频

```http
POST /v1/generate
Content-Type: application/json
```

最小请求：

```json
{
  "prompt": "A fox walking through a misty forest"
}
```

完整请求示例：

```json
{
  "prompt": "A fox walking through a misty forest",
  "negative_prompt": "blurry, distorted",
  "height": 352,
  "width": 640,
  "num_frames": 49,
  "seed": 42,
  "guidance_scale": 5.0,
  "num_inference_steps": 20,
  "flow_shift": 3.0,
  "solver_type": "euler"
}
```

命令行测试：

```bash
curl -X POST http://127.0.0.1:8082/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A fox walking through a misty forest",
    "height": 352,
    "width": 640,
    "num_frames": 49,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42
  }'
```

响应示例：

```json
{
  "id": "gen_000001_1234abcd",
  "status": "succeeded",
  "rollout_type": "seg_ar",
  "video_url": "outputs/gen_000001_1234abcd_20260821_120000_seed42.mp4",
  "metadata_url": "outputs/gen_000001_1234abcd_20260821_120000_seed42.json",
  "seed": 42,
  "width": 640,
  "height": 352,
  "frames": 49,
  "fps": 24,
  "duration_seconds": 2.0417,
  "generation_seconds": 80.2,
  "encoding_seconds": 0.8
}
```

### 多段 seg_ar

前端支持直接粘贴结构化 JSON。可以将 JSON 放在 Setup 页的
`Subject Description`，将 `Environment Description` 填为任意非空文本后
进入 Studio；也可以直接将 JSON 粘贴到 Studio 的 direction 输入框。

JSON 示例：

```json
{
  "perspective_view": "first-person",
  "segments": [
    {
      "visual_description_long": "A fox stands in a forest.",
      "contained_actions": [
        {
          "start_time": "00:00.000",
          "end_time": "00:02.000",
          "action_description": "The fox looks toward the river."
        }
      ]
    },
    {
      "description_change": "The fox begins walking.",
      "contained_actions": [
        {
          "start_time": "00:00.000",
          "end_time": "00:01.500",
          "action_description": "The fox walks forward."
        }
      ]
    }
  ]
}
```

前端从 `/v1/config` 获取 `fps`，将每个 `segments[i]` 序列化为独立 prompt，并用 `|||` 连接。每段时长根据 action/segment 时间戳计算，然后累加生成 pixel-frame boundaries。

上述 JSON 在 24 FPS 下发送：

```json
{
  "prompt": "{\"perspective_view\":\"first-person\",\"segments\":[...]}|||{\"segments\":[...]}",
  "num_frames": 84,
  "segment_boundaries": [0, 48, 84]
}
```

计算方式：

```text
segment 0: 2.0 秒 × 24 FPS = 48 帧
segment 1: 1.5 秒 × 24 FPS = 36 帧
boundaries: [0, 48, 84]
num_frames: 84
```

后端不会信任前端结果，而是重新解析每个 `|||` prompt 的 JSON 和时间戳，并校验：

```text
segment_boundaries 数量 = prompt segment 数量 + 1
segment_boundaries[0] = 0
最后一个 boundary = num_frames
所有 boundary 严格递增
前端 boundaries = 后端按 prompt timestamps 和 FPS 重算的 boundaries
前端 num_frames = 后端重算的总帧数
```

校验不一致时返回 HTTP 400。普通自然语言 prompt 没有 `segments` JSON 时，不会启用前端自动计算，仍使用后端配置中的 `inference.num_frames`。

### 输出文件

默认输出目录：

```text
seg_ar_web_outputs/
```

每次请求生成：

```text
<request-id>_<timestamp>_seed<seed>.mp4
<request-id>_<timestamp>_seed<seed>.json
```

可以修改输出目录：

```bash
--output-dir /path/to/output
```

## 前端开发模式

如果只调试页面：

```bash
cd apps/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

该模式不会自动代理 `/v1/generate`。若没有额外配置 Vite proxy，生成请求会失败。

联调时推荐先执行 `npm run build`，然后由 `seg_ar_web_server.py` 在同一端口托管前端和 API。

## 检查

```bash
cd apps/frontend
npm run lint
npm test -- --pool=threads --maxWorkers=1
npm run build
```

测试使用单 worker，可以避免部分共享文件系统或容器环境中 Vitest fork worker 启动超时。

## 参考图初始化更新

参考图模式内部五秒参考段完整进入 KV cache，视频只保留最后 0.5 秒并标记 **T0-Reference Image**。用户提示词生成段从 **T1 · TEXT** 开始。详见 [中文说明](../IMAGE_REFERENCE_CN.md)。
