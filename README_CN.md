<p align="left">
  <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>

<h1 align="center">Hy-Nomas IDM</h1>

**Hy-Nomas IDM** 用于从视频中识别相机运动。模型采用 VPT 视觉骨干，输出时序动作概率、变焦概率、速度预测、主体运动质量评分（`motion_score`），以及 clip-level 相机运动预测。本包提供模型权重和独立推理 API，便于在 [Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas) 中使用。

**即将发布：采用相同架构、面向游戏场景的专用 IDM。**

## 目录

- [模型概览](#模型概览)
- [相机位姿估计](#相机位姿估计)
- [环境配置](#环境配置)
- [推理](#推理)
- [输入与输出](#输入与输出)
- [指标复现](#指标复现)
- [目录结构](#目录结构)
- [致谢](#致谢)
- [引用](#引用)

## 模型概览

| 属性 | 说明 |
|---|---|
| 骨干网络 | VPT |
| 参数量 | 约 0.474B（473,986,521） |
| 输入 | 144 帧 RGB 图像，96 × 128 像素（高 × 宽） |
| 时序输出 | 每窗口 24 个时间桶，每桶 0.25 秒 |
| 预测内容 | 每桶 18 维动作概率、2 维变焦概率、6 维速度值 |
| Motion Score | 通过已训练的回归头预测画面中主体的运动质量（`motion_score`） |
| Clip-level 预测 | 整个输入视频的 15 类相机运动预测（`clip_camera_motion`） |
| Clip-level head | 15 个线性分类器，每类 13 维特征，共 210 个参数；融合权重 0.3 |
| 模型文件 | `weights/idm.pt`、`weights/idm.json` 与 `weights/clip_head.json` |

在 CameraBench 上，基于 frame-level 预测的时间聚合达到 **88.68% mAP**；冻结模型后，在 CameraBench 训练集上训练轻量 clip-level head，与原始预测融合后达到 **89.19% mAP**。两项均为 clip-level 宏平均 mAP，采用整段视频均匀采样，覆盖 969 条视频、15 个类别、7,679 个已知标签。本包提供两种方案，分别使用 `--sampling uniform` 和 `--sampling uniform --clip-head` 启用。

### CameraBench 子项结果

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">参数量</th>
<th rowspan="2">总体</th>
<th colspan="6">平移（Dolly / Pedestal / Truck）</th>
<th colspan="2">变焦</th>
<th colspan="6">旋转（Pan / Tilt / Roll）</th>
<th rowspan="2">静止</th>
</tr>
<tr>
<th>前进</th>
<th>后退</th>
<th>上移</th>
<th>下移</th>
<th>右移</th>
<th>左移</th>
<th>拉近</th>
<th>拉远</th>
<th>右转</th>
<th>左转</th>
<th>上仰</th>
<th>下俯</th>
<th>顺时针</th>
<th>逆时针</th>
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

对照数据来源：[CameraBench 官方榜单](https://sy77777en.github.io/CameraBench/leaderboard/table.html)、[CameraVLM 论文表 1](https://arxiv.org/html/2610.17242v1#S4)。

### 相机位姿估计

Hy-Nomas IDM 将预测的平移速度与角速度积分为相机轨迹，采用 6 秒窗口推理。ATE 与平移 RPE 的单位为米，旋转 RPE 的单位为度；均越低越好，粗体分别标出离线与流式分组内各列的最低值。

<!-- BEGIN CAMERA POSE RESULTS -->
<table>
<thead>
<tr>
<th rowspan="2">模型</th>
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
<tr><th colspan="10" align="left">离线方法</th></tr>
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
<tr><th colspan="10" align="left">流式方法</th></tr>
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

对照方法的数值及离线／流式分组引自 [CameraVLM 论文表 2](https://arxiv.org/html/2610.17242v1#S4.T2)。Hy-Nomas IDM 的数值为发布权重通过速度积分得到的实测结果。

评测采用公开的 [MonST3R / Cambrian-P 协议](https://github.com/cambrian-mllm/cambrian-p#pose-estimation-performance)。

## 环境配置

本算子兼容 [Hy-Nomas 的运行环境](https://github.com/Tencent-Hunyuan/Hy-Nomas#environment-setup)，可直接复用其中的 Python、PyTorch 和 CUDA。

单独使用时，环境可简化为本包的 `requirements.txt`：推荐 Linux、Python 3.10–3.13，以及支持 BF16 的 NVIDIA GPU；先安装与 CUDA 环境匹配的 PyTorch（>=2.3），再安装以下依赖。已有 Hy-Nomas 环境也执行同一命令以补齐依赖：

```bash
python -m pip install -r requirements.txt
```

仅需上述五项直接依赖；VPT 所需的动作空间类型定义已随包提供。使用 `--hud` 时，还需将 FFmpeg 加入 `PATH`。

将 `weights/`、`idm/` 和 `vendor/` 与 `infer.py` 放在同一目录下。程序相对于包目录解析路径，整体移动后无需修改代码。权重约 1.9 GB，已配置 Git LFS；仅克隆源码时，还需要获取对应的 LFS 文件。

## 推理

### 命令行

在本包目录下执行：

```bash
# 单个视频
python infer.py --video /path/to/input.mp4 --output outputs/input.json

# 目录直接包含的所有 MP4 视频
python infer.py --video /path/to/videos --output outputs/videos.json

# 指定权重与设备
python infer.py --video /path/to/input.mp4 --output outputs/input.json \
  --weights weights/idm.pt --device cuda:0 --precision bf16

# 启用 clip-level head
python infer.py --video /path/to/input.mp4 --output outputs/input_clip.json \
  --sampling uniform --clip-head

# CPU 推理
python infer.py --video /path/to/input.mp4 --output outputs/cpu.json \
  --device cpu --precision fp32
```

默认采用 `--sampling windows`，按 24 fps 处理连续的 6 秒窗口，使用 CUDA BF16 autocast，模型权重保持 FP32。目录输入按文件名排序；每处理完一个视频，原子更新 JSON 输出，同名文件会被覆盖。输入无效或权重不兼容时会报错。CPU 推理可能较慢；FP32、不同硬件或依赖版本可能产生数值差异。

### 预测结果 HUD

增加 `--hud` 参数，一次推理同时输出预测 JSON 和叠加预测结果的 MP4：

```bash
python infer.py --video assets/case_01.mp4 --output outputs/case_01.json --hud --shake-threshold 0.7
```

该命令生成 `outputs/case_01.json` 和 `outputs/case_01_hud.mp4`。目录输入会为
每个视频生成一份 HUD 视频，与 JSON 放在同一目录。需要安装 FFmpeg 并将
`ffmpeg` 加入 `PATH`；OpenCV 已包含在 Python 依赖中。

HUD 显示动作按键、八方向鼠标指示、变焦和模型原生速度值。
SHAKE 的默认激活条件为概率 **>= 0.7**，可通过 `--shake-threshold` 同时调整 JSON 判定和 HUD 高亮。
`--rotation-threshold 0.5` 控制旋转方向的互斥判定及 HUD 高亮，`--hud-threshold 0.5` 控制其他按键和变焦的高亮。原始概率和视频分数保持不变。
默认 HUD 每 0.25 秒更新一次，SHAKE 按钮同时显示预测概率。
旋转圆盘的上下左右扇区各占 20°（中心 ±10°），斜向扇区各占 70°（中心 ±35°）。
仅激活概率最高且达到旋转阈值的一个方向；未达阈值时全部熄灭，概率并列时取通道顺序中第一个方向。
输出为 H.264 MP4，使用源分辨率和平均帧率，有音轨时保留音频（AAC）。
可变帧率输入会按平均帧率输出恒定帧率视频；奇数边长补一个像素以兼容 H.264。

### 示例

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

将本包目录加入 Python 模块搜索路径后：

```python
from idm.runtime import load_model, infer_video

model, metadata = load_model(device="cuda:0")
result = infer_video(model, "/path/to/input.mp4", shake_threshold=0.7, rotation_threshold=0.5)
print(result["clip_camera_motion"])
```

使用 clip-level head：

```python
from idm.clip_head import ClipHead

head = ClipHead()  # 默认读取 weights/clip_head.json
result = head.apply(infer_video(model, "/path/to/input.mp4", sampling="uniform"))
print(result["clip_camera_motion"])
```

## 输入与输出

默认将视频划分为连续的 **6 秒窗口，按 24 fps 采样**。Decord 将图像解码为 96 × 128 像素 RGB，不做裁剪；每六帧特征输出一个 0.25 秒时间桶。最后不足 6 秒的窗口重复末帧补齐，仅返回实际视频时长内的桶。

视频级评测可显式使用 `--sampling uniform`：按 `rint(linspace(0, frame_count - 1, 144))` 在整段视频均匀取 144 帧，整段输出 24 个桶，时间间隔随视频时长变化。Python API 对应 `infer_video(..., sampling="uniform")`。

平移速度使用模型原生尺度和相机自身坐标，角速度的训练目标单位为度/秒；这些回归输出不代表经过标定的物理测量。`sampling` 保留采样帧索引，窗口模式还记录窗口边界和每个时间桶的起始时刻。

`clip_camera_motion` 汇总整个输入视频中所有有效窗口的相机运动预测；`window_motion_scores` 则分别记录每个窗口的主体运动质量评分。

JSON 输出包含 `model` 元数据与 `samples` 列表。窗口模式的 N 为 `ceil(视频秒数 × 4)`，整段均匀模式的 N 为 24：

| 字段 | 形状／含义 |
|---|---|
| `motion_prob` | N × 18 动作概率，顺序见 `model.label_names` |
| `zoom_prob` | N × 2 概率：拉近、拉远 |
| `velocity6` | N × 6：右、下、前、右转、下俯、顺时针滚转 |
| `motion_score` | 画面中主体的运动质量评分；窗口模式按有效时长加权汇总 |
| `window_motion_scores` | 各窗口的主体运动质量评分（窗口模式） |
| `shake_prob` | 来自动作头第 15 通道的 N 个值（索引从 0 开始） |
| `shake_active` | N 个布尔值：`shake_prob >= shake_threshold` |
| `shake_threshold` | SHAKE 判定阈值，默认 0.7 |
| `rotation_sector` | N 个互斥方向编号：0 右、1 右上、2 上、3 左上、4 左、5 左下、6 下、7 右下；-1 表示未激活 |
| `rotation_threshold` | 旋转判定阈值，默认 0.5 |
| `clip_camera_motion` | 整个输入视频的 15 类相机运动预测；启用 head 时为融合结果 |
| `clip_camera_motion_base` | 启用 head 时保留的原始时间聚合预测 |
| `clip_camera_motion_head` | 启用 head 时的独立分类器预测 |
| `clip_head` | 启用 head 时的权重校验值与融合参数 |
| `sampling` | 源视频 FPS、帧数、时长和采样帧索引 |

18 维动作通道依次为 `w`、`a`、`s`、`d`；八个鼠标方向（右、右上、上、左上、左、左下、下、右下）；`special`、`up`、`down`、`shake`、`roll_clockwise`、`roll_counterclockwise`。

动作和变焦 logits 先转换为概率，再进行聚合。Pan 和 Tilt 在各自方向组内取最大概率，斜向可同时支持两者；运动和变焦取时间最大值。Static 先逐桶计算 `1 - max(运动方向、变焦、抖动)`，再取时间最小值，`special` 通道不作为静止分数。这些连续分数用于排序与分析，默认时间聚合不应用速度融合或时序平滑。`shake_active` 和 `rotation_sector` 是单独的阈值判定，不改变连续分数。旋转仅选择概率最高且达到阈值的方向，并列时取通道顺序中第一个方向；JSON 与 HUD 使用同一判定。

## 指标复现

配置环境后，在包目录中执行以下命令。

### CameraBench：AP / mAP

准备[基准视频](https://huggingface.co/datasets/syCen/Videos4CameraBench/tree/1ec8d55f8c380abbb6ea7087e64778f0ca76a68f)及[官方 15 类标注](https://github.com/linzhiqiu/t2v_metrics/tree/6ecb74f92028f42c7e64546d3a71e98c8c73068f/camerabench/data/binary_classification)。以下命令复现发布权重对应的 **88.68% mAP** 方案：

```bash
python infer.py --video /path/to/CameraBench/videos --sampling uniform \
  --precision bf16 --output outputs/camerabench_predictions.json
python tools/evaluate_camerabench.py \
  --predictions outputs/camerabench_predictions.json \
  --labels /path/to/t2v_metrics/camerabench/data/binary_classification \
  --output outputs/camerabench_metrics.json
```

子项 AP 和宏平均 mAP 保存至 `outputs/camerabench_metrics.json`；已有预测时只运行第二条命令。

**89.19% mAP：** 对上述预测应用已提供的 clip-level head，然后评分，无需重复运行视频模型：

```bash
python tools/apply_clip_head.py --predictions outputs/camerabench_predictions.json \
  --output outputs/camerabench_clip_predictions.json
python tools/evaluate_camerabench.py \
  --predictions outputs/camerabench_clip_predictions.json \
  --labels /path/to/t2v_metrics/camerabench/data/binary_classification \
  --output outputs/camerabench_clip_metrics.json
```

Head 权重为 `weights/clip_head.json`，最终预测为 `0.7 × 时间聚合预测 + 0.3 × head 预测`。也可在首次推理时使用 `--sampling uniform --clip-head` 直接输出融合结果。

### ScanNet / TUM-dynamic / Sintel：ATE / RPE

按 [MonST3R 数据准备脚本](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)准备数据。`--data-root` 指向包含 `scannetv2/`、`tum/`、`sintel/` 的目录。

```bash
python -m pip install -r requirements-eval.txt

# 三个数据集：预测轨迹并计算指标
python tools/evaluate_pose.py --data-root /path/to/data \
  --datasets scannet tum sintel --methods velocity_only --output outputs/pose

# 复用已有预测，仅重算指标
python tools/evaluate_pose.py --data-root /path/to/data \
  --metrics-only --predictions-root outputs/pose --output outputs/pose_metrics
```

单独评测时使用 `--datasets scannet`、`--datasets tum` 或 `--datasets sintel`；分类＋速度对照使用 `--methods classification_velocity`。各数据集指标保存在 `outputs/pose/<dataset>/report.json`，表格对应字段为 `summary.velocity_only.full_eligible_set_mean`。

## 目录结构

```text
README.md / README_CN.md    英文／中文说明
infer.py                    命令行入口
idm/                        模型加载、视频采样与分数映射
weights/                    主模型、clip-level head 权重与元数据
vendor/video-pre-training/  VPT 运行代码及许可证
assets/                     示例视频、预测结果与 HUD 视频
requirements.txt            运行依赖
requirements-eval.txt       可选的位姿评测依赖
tests/                      输出、采样、HUD 与轨迹测试
tools/apply_clip_head.py     对已有预测应用 clip-level head
tools/evaluate_camerabench.py CameraBench AP / mAP 评测
tools/evaluate_pose.py      ScanNet、TUM-dynamic 与 Sintel 评测
tools/verify_package.py     包文件校验
```

## 致谢

本包使用 [OpenAI VPT](https://github.com/openai/Video-Pre-Training) 运行代码，其 MIT 许可证保留于 `vendor/video-pre-training/LICENSE`。来源归属及分发说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

同时感谢 [CameraVLM](https://3dagentworld.github.io/CameraVLM/) 团队为社区作出的宝贵贡献。

## 引用

如果 Hy-Nomas 对您的研究有所帮助，请引用我们的技术报告：

```bibtex
@article{chen2026hy,
  title={Hy-Nomas: Towards Free-form Interactive World Generation},
  author={Chen, Qiaoling and Cheng, Jiaxiang and Deng, Xueqing and Hang, Tiankai and Hou, Ji and Li, Jiaqi and Liu, Yifan and Lu, Tao and Tang, Junshu and Tang, Yu and others},
  journal={arXiv preprint arXiv:2610.36729},
  year={2026}
}
```
