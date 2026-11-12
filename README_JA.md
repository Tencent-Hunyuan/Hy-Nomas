<p align="left">
    <a href="README.md">English</a>&nbsp;｜&nbsp;<a href="README_CN.md">中文</a>&nbsp;｜&nbsp;日本語
</p>
<br>

<p align="center">
    <img src="assets/logo-en.png" alt="Hy-Nomas" width="400"/> <br>
</p>

<h1 align="center">
    Hy-Nomas: Towards Free-Form Interactive <br> World Generation
</h1>

**Hy-Nomas** は、複数ターンのテキストとカメラの入力で制御する、自由形式のインタラクティブな世界生成モデルです。RTX 4090 1 枚での低メモリ推論に対応し、下流タスク向けの LoRA ファインチューニングをサポートし、インタラクティブな Web デモも備えています。

<div align="center" style="line-height: 1;">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
&nbsp;&nbsp;
<a href="https://huggingface.co/tencent/Hy-Nomas" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hy--Nomas-ffc107?color=ffc107&logoColor=white" alt="Hy-Nomas"></a>
&nbsp;&nbsp;
<a href="https://huggingface.co/datasets/tencent/Hy-Nomas-Bench" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hy--Nomas--Bench-ffc107?color=ffc107&logoColor=white" alt="Hy-Nomas-Bench"></a>
&nbsp;&nbsp;
<a href="https://x.com/TencentHunyuan/status/2113676827114053892" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Hy-black.svg?logo=x" height="22" alt="Hy on X"></a>

</div>

<p align="center">
    🖥️&nbsp;<a href="https://hy-nomas.hy.tencent.com/demo"><b>インタラクティブデモ</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🌐&nbsp;<a href="https://hy-nomas.hy.tencent.com/"><b>プロジェクトサイト</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    📄&nbsp;<a href="https://arxiv.org/abs/2610.36729"><b>テクニカルレポート</b></a>
</p>

<p align="center">
    👏 <a href="./assets/wechat.png" target="_blank">WeChat コミュニティ</a>に参加しましょう
</p>

## 🔥 ニュース

- 🤝 **2026 年 11 月 3 日 — コミュニティによる紹介：** [@BennyDaBall_OG](https://x.com/BennyDaBall_OG) が、LTX-2.5 を使って Hy-Nomas の動画に動画条件付きの音声を加える例を披露しています — [音声デモを見る](https://x.com/BennyDaBall_OG/status/2117675372024070250)。[@preluluofficial](https://x.com/preluluofficial) が、Hy-Nomas で作ったインタラクティブなミニドラマのロールプレイをプレビューしています — [ロールプレイのデモを見る](https://x.com/preluluofficial/status/2117563133120176449)。[コミュニティによる貢献](#-コミュニティによる貢献)を参照してください。
- 🖼️ **2026 年 10 月 31 日 — デモで画像リファレンスに対応：** 画像をアップロードして世界を初期化し、そのままテキストプロンプトとカメラ操作で続けられます。[インタラクティブデモ](https://hy-nomas.hy.tencent.com/demo)を試すか、[デプロイガイド](apps/README.md#start-with-a-reference-image)を参照してください。
- 🎬 **2026 年 10 月 29 日 — Hy-Nomas と Genie 4 の比較：** Genie 4 の例を Hy-Nomas で再現した[動画での比較](https://x.com/TencentHunyuan/status/2115851165904814283)をご覧ください。
- 📄 **2026 年 10 月 27 日 — テクニカルレポートを arXiv で公開：** [arXiv](https://arxiv.org/abs/2610.36729)。
- 🤝 **2026 年 10 月 26 日 — コミュニティによる統合：** [SGLang](https://github.com/sgl-project/sglang/pull/46124)、[ComfyUI](https://github.com/PedroMarinhoDev/ComfyUI-Hy-Nomas)、[Reactor Runtime](reactor/README.md) で Hy-Nomas を動かせます。詳細は[コミュニティによる貢献](#-コミュニティによる貢献)を参照してください。
- 📄 **2026 年 10 月 26 日 — テクニカルレポート：** [Hy-Nomas テクニカルレポート](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/main/Hy_Nomas_Tech_Report.pdf)を更新しました。
- 🌍 **2026 年 10 月 25 日 — WanGP に対応：** [@deepbeepmeep](https://x.com/deepbeepmeep) による [WanGP v18.81](https://github.com/deepbeepmeep/WanGP) が、キーボードでの探索とテキスト駆動のイベントを備えた Hy-Nomas の世界生成に対応しました。[発表](https://x.com/deepbeepmeep/status/2114503683295662230)と[統合コミット](https://github.com/deepbeepmeep/WanGP/commit/c4d3efce4cd896c6e14fcb82787577bfc3181b99)を参照してください。
- 🛠️ **2026 年 10 月 23 日 — Diffusers での推論：** [Diffusers での推論の例](#diffusers-での推論)を追加しました。
- 🎮 **2026 年 10 月 21 日 — インタラクティブデモ：** 複数ターンのテキストとカメラの制御を[オンラインデモ](https://hy-nomas.hy.tencent.com/demo)で体験するか、[デプロイガイド](apps/README.md)に従って自分で動かしてください。
- 🚀 **2026 年 10 月 21 日 — 初回のコードリリース：** 学習と推論のコードには、[LoRA ファインチューニング](#lora-ファインチューニング)と[単一 GPU での推論](#低メモリ-gpu-での推論)が含まれます。[モデルの重み](https://huggingface.co/tencent/Hy-Nomas)を入手して、探索を始めましょう。

## 🎥 デモ

https://github.com/user-attachments/assets/e2e05b2b-9024-4b0e-9926-6eed05356b76

[フル解像度の動画を見る](https://hy-nomas.hy.tencent.com/gallery/hy_nomas_promo.mp4)

## 🧩 コミュニティによる貢献

Hy-Nomas を開発・利用しているプロジェクトがあれば、ぜひお知らせください。

- **LTX-2.5 による動画条件付きの音声** — [デモとワークフローの解説](https://x.com/BennyDaBall_OG/status/2117675372024070250)、作者は [@BennyDaBall_OG](https://x.com/BennyDaBall_OG)：完成した Hy-Nomas の動画に条件付けして音声トラックを生成する、学習不要の ComfyUI ワークフローです。動画の潜在表現は固定したまま、元のフレームに音声を多重化します。作者は、[@t8star_aix](https://x.com/t8star_aix) による Hy-Nomas ノードと、[@LatentSpacer](https://x.com/LatentSpacer) の ComfyUI-Hy-Nomas パックが土台になっていると述べています。
- **Prelulu のインタラクティブミニドラマ** — [ロールプレイのデモ](https://x.com/preluluofficial/status/2117563133120176449)、作者は [@preluluofficial](https://x.com/preluluofficial)：Hy-Nomas で作られたインタラクティブなロールプレイ体験のプレビューで、ユーザーはキャラクターと会話し、物語の別の方向を探索できます。
- **Google TPU デモ** — [Mashbox のデモ](https://mashbox.gg/world)、作者は [@boshjerns](https://x.com/boshjerns)：Google TPU 上での、キーボード操作による Hy-Nomas の世界生成です。[発表](https://x.com/boshjerns/status/2114962446947651907)によると、v6e-4 スライス（4 チップ）で 720p のリアルタイム生成を実現し、動画 1 秒あたりの計算時間は 0.933 秒、60 秒間の実行中にフレームの遅延は発生しませんでした。
- **WanGP** — [WanGP](https://github.com/deepbeepmeep/WanGP)、作者は [@deepbeepmeep](https://x.com/deepbeepmeep)：v18.81 の Hy-Nomas 対応により、テキストからの世界生成、キーボードナビゲーションとカメラ制御、テキスト駆動のイベント、調整可能な履歴メモリが使えます。探索は保存して再開することも、動画として録画することもできます。[発表](https://x.com/deepbeepmeep/status/2114503683295662230)と[統合コミット](https://github.com/deepbeepmeep/WanGP/commit/c4d3efce4cd896c6e14fcb82787577bfc3181b99)を参照してください。
- **SGLang** — [SGLang Diffusion](https://github.com/sgl-project/sglang)、作者は [@sgl_project](https://x.com/sgl_project)：Hy-Nomas への対応が `main` にマージされており、オフラインの動画ジョブとリアルタイムのストリーミングセッションの両方に対応しています。[統合 PR](https://github.com/sgl-project/sglang/pull/46124) と[インストールガイド](https://lmsysorg.mintlify.app/cookbook/diffusion/HY-Nomas/HY-Nomas#hw=h200&placement=resident&nodes=1&gpus_per_node=1&topology_mode=auto&tp_size=1&ulysses_degree=1&ring_degree=1)を参照してください。
- **ComfyUI** — [ComfyUI-Hy-Nomas](https://github.com/PedroMarinhoDev/ComfyUI-Hy-Nomas)、作者は [@LatentSpacer](https://x.com/LatentSpacer)：動画セグメントの連結と、WASD によるインタラクティブなカメラ操作を備えたコミュニティの統合です。
- **Reactor Runtime** — [Hy-Nomas 統合ガイド](reactor/README.md)、貢献者は [@Rising0321](https://github.com/Rising0321)：世界の履歴を保持したまま、プロンプトのライブ更新、WASD/矢印キーによるカメラ操作、アップロードしたカメラ軌道に対応する、ストリーミングサービスとブラウザデモです。[統合 PR](https://github.com/Tencent-Hunyuan/Hy-Nomas/pull/3) を参照してください。

Hy-Nomas を土台に開発してくださった貢献者の皆さんに感謝します！

Hy-Nomas を使って何かを作っていますか？ コミュニティのプロジェクト、統合、チュートリアル、LoRA アダプターを歓迎します。[issue を作成](https://github.com/Tencent-Hunyuan/Hy-Nomas/issues)するか、プルリクエストでこのセクションに追加して、作品を共有してください。このリポジトリへの貢献については、[コントリビューションガイド](CONTRIBUTING.md)に記載しています。

---

## 目次

- [🔥 ニュース](#-ニュース)
- [🎥 デモ](#-デモ)
- [🧩 コミュニティによる貢献](#-コミュニティによる貢献)
- [モデルの紹介](#モデルの紹介)
- [主な特長](#主な特長)
- [使い方](#使い方)
  - [環境のセットアップ](#環境のセットアップ)
  - [モデルアセット](#モデルアセット)
  - [推論](#推論)
    - [ソースコードでの推論](#ソースコードでの推論)
    - [低メモリ GPU での推論](#低メモリ-gpu-での推論)
    - [Diffusers での推論](#diffusers-での推論)
    - [SGLang での推論](#sglang-での推論)
  - [インタラクティブデモ](#インタラクティブデモ)
    - [Reactor Runtime](#reactor-runtime)
    - [Web デモ](#web-デモ)
- [LoRA ファインチューニング](#lora-ファインチューニング)
  - [SFT データの準備](#sft-データの準備)
  - [オフラインのシーケンスパッキング](#オフラインのシーケンスパッキング)
  - [学習とマージ](#学習とマージ)
- [コントリビューション](CONTRIBUTING.md)
- [セキュリティ](SECURITY.md)
- [ライセンス](#ライセンス)
- [謝辞](#謝辞)
- [引用](#引用)
- [お問い合わせ](#お問い合わせ)

---

## モデルの紹介

**Hy-Nomas** は、自由形式のインタラクティブな動画世界生成モデルです。自由形式の複数ターンのテキストプロンプト制御と、カメラ制御に対応しています。このリポジトリでは、学習、推論、インタラクティブデモ、データアノテーションのコードを提供します。

## 主な特長

- **セグメント単位の生成：** セグメントレベルのプロンプトまたはカメラ制御信号のどちらかで制御する、自己回帰的なセグメント単位の世界生成です。一貫性を保つため、生涯にわたる KV キャッシュを維持します。
- **自由形式のプロンプト制御：** あらゆる自由形式のユーザープロンプトで世界生成を制御し、想像上のオブジェクト、スタイル、シーン、アクションを生成できます。
- **カメラ制御：** テキストプロンプトに加えて、カメラ軌道を条件にした生成ができます。
- **蒸留による推論：** 推論には 4〜8 回のデノイジングステップと、32 チャンネルの DCAE-V 潜在表現向けにファインチューニングした PixelUMM デコーダーを使います。
- **インタラクティブデモ：** Web インターフェースが、複数ターンにわたる生成と、セグメント境界でのデコードに対応します。
- **ファインチューニングとデータ準備：** カスタムの動画データセットでの LoRA ファインチューニングに加え、アノテーションのワークフローを使って、ローカルの動画から学習データを準備できます。

## 使い方

### 環境のセットアップ

次の 2 つのセットアップ方法から 1 つを選んでください。**ビルド済みの Docker イメージを強く推奨します**。Python 環境、PyTorch、必要な CUDA アテンションカーネルがひとまとめに入っています。

#### 方法 1：ビルド済みの Docker イメージ（推奨）

互換性のある NVIDIA ドライバー、Docker、NVIDIA Container Toolkit を備えた Linux ホストを用意してください。次のコマンドは、リポジトリのルートで実行します。

```bash
export HY_NOMAS_IMAGE="hynomasoss/hynomasoss:cu130"

docker pull "$HY_NOMAS_IMAGE"
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$PWD:/workspace/Hy-Nomas" \
  -w /workspace/Hy-Nomas \
  "$HY_NOMAS_IMAGE" bash
```

以降のコマンドは、コンテナ内でビルド済みの Python 環境を使って実行します。モデルアセットはマウントしたリポジトリの下に置くか、モデルのディレクトリ用にボリュームマウントを追加してください。以下の起動コマンドでは、コンテナ内から見えるパスを使います。

#### 方法 2：手動インストール

Linux、Python 3.12 以上、uv、互換性のある NVIDIA ドライバー、CUDA 13.0 Toolkit（`nvcc`）、C++ コンパイラを用意してください。リポジトリのルートで次を実行します。

```bash
uv sync --extra all
source .venv/bin/activate
```

`all` エクストラには、PyTorch 2.10.0（CUDA 13.0）、FlashAttention 2、FlashAttention 3、MagiAttention が含まれます。`pyproject.toml` は、PyTorch の CUDA インデックスと公式の FA3 ソースを設定し、アテンションカーネルのビルドが環境内の PyTorch を使えるよう、ビルド分離を無効にしています。カーネルのコンパイルには時間がかかることがあります。

### モデルアセット

モデルリポジトリには、DiT 生成モデル、LoRA アダプター、動画オートエンコーダーとデコーダー、そしてオンラインのプロンプト強化用の Nomas-PE が含まれています。

Hugging Face の `tencent/Hy-Nomas` からすべてのモデルの重みをダウンロードし、リポジトリのディレクトリ構造を保ってください。

```bash
export Hy_Nomas_Model_Path=/path/to/Hy-Nomas
hf download tencent/Hy-Nomas --local-dir "${Hy_Nomas_Model_Path}"
```

`Hy_Nomas_Model_Path` は、ダウンロード先のディレクトリに一度設定してください。以下のコマンドは既存の値を再利用し、新しいシェルでは `/path/to/Hy-Nomas` が既定になります。下のモデル名をクリックすると、Hugging Face 上でそのファイルを確認できます。

| カテゴリ | モデルアセット | 用途 |
|:---|:---|:---|
| DiT | **[Base autoregressive DiT (FP32)](https://huggingface.co/tencent/Hy-Nomas/tree/main/DiT/base_ar)** | セグメントフォーシングによる LoRA ファインチューニング用の、蒸留していないチェックポイント。 |
| DiT | **[Distilled DiT (FP32)](https://huggingface.co/tencent/Hy-Nomas/tree/main/DiT/distilled)** | 4/6/8 ステップの高速推論とインタラクティブデモ用のチェックポイント。ファインチューニングした LoRA アダプターをマージする際のベースにもなります。 |
| DiT | **[Quantized Distilled DiT (INT8)](https://huggingface.co/tencent/Hy-Nomas/tree/main/DiT/distilled_int8)** | 単一 GPU でのデプロイを含む、低メモリ推論用の量子化チェックポイント。 |
| DiT | **[Cinematic LoRA](https://huggingface.co/tencent/Hy-Nomas/tree/main/DiT/lora_weights/cinematic)** | シネマティックなデータを使った LoRA アダプター。蒸留済み DiT にマージして、生成される世界の映像品質を高めます。 |
| DiT | **[Robotics LoRA](https://huggingface.co/tencent/Hy-Nomas/tree/main/DiT/lora_weights/dual_arm_robotics)** | 実世界のロボットアームのデータセット [RealSource-World](https://huggingface.co/datasets/RealmanRobot/RealSource-World) で学習した LoRA アダプター。蒸留済み DiT にマージして、生成される世界のロボティクスシミュレーション能力を高めます。 |
| VAE | **[DCAE-V](https://huggingface.co/tencent/Hy-Nomas/tree/main/VAE/DCAE-V)** | 学習用の動画を潜在表現にエンコードし、デコードするための動画オートエンコーダー。 |
| VAE | **[Fine-tuned PixelUMM Decoder](https://huggingface.co/tencent/Hy-Nomas/tree/main/VAE/PixelUMM_FT)** | 生成された DCAE-V の潜在表現を動画フレームに変換するデコーダー。 |
| VAE | **[Quantized PixelUMM Decoder (FP8)](https://huggingface.co/tencent/Hy-Nomas/tree/main/VAE/PixelUMM_FT_FP8)** | メモリ使用量を抑えてデコードを高速化する、FP8 の重みを持つデコーダー。 |
| PE | **[Nomas-PE](https://huggingface.co/tencent/Hy-Nomas/tree/main/PE)** | インタラクティブデモが、ユーザーの指示を構造化された生成プロンプトに変換するために使う、オンラインのプロンプト強化モデル。 |

### 推論

#### ソースコードでの推論

リポジトリのルートから `configs/experiment/inference.yaml` を使って実行し、モデルのパス、プロンプトファイル、出力ディレクトリを上書きします。

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

bash scripts/inference/inference_from_config.sh configs/experiment/inference.yaml \
  --sp-size 1 --tp-size 8 \
  dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.prompt=/path/to/prompts.json \
  inference.save_path=./outputs/distill
```

推論のプリセットは **SP=1、TP=8** を使います。テンソル並列は、DiT のアテンションヘッド（Q/K/V の射影と KV キャッシュ）を 8 つのランクに分割します。アテンションの後、1 回の all-to-all で各ランクがトークンのシャードに戻り、そこで出力射影と MLP が重み全体を使って実行されます。量子化していない DiT の重みでの KV キャッシュのロールアウト（`seg_ar`、`chunk_ar`）に対応しており、`sp_size > 1` と組み合わせることはできません。

#### 低メモリ GPU での推論

**Hy-Nomas は、NVIDIA GeForce RTX 4090 1 枚での推論に対応しています**。`configs/experiment/single_gpu_low_mem_inference.yaml` を使います。この構成では、次の方法で GPU メモリの使用量を減らします。

- **INT8 の DiT 重みと KV キャッシュ：** 重みとキャッシュしたキー/バリューを INT8 で保持しつつ、DiT の線形層とアテンションの計算は BF16 のままにします。
- **スライディングウィンドウの履歴：** セグメントごとに KV キャッシュが増え続けるのではなく、保持するコンテキストに上限を設けます。
- **CPU オフロード：** テキストエンコーダー、DiT、VAE を、それぞれの処理段階で GPU に移し、終わったら CPU に戻します。
- **単一 GPU 向けカーネル：** DiT には FlexAttention を、デコードには CUTLASS FP8 を使い、シーケンス並列は無効にします（`sp_size=1`）。

互換性のある INT8 の DiT チェックポイントと、FP8 のデコーダーチェックポイントを使ってください。`nvidia-smi` から GPU が 1 枚見えるホストまたはコンテナで実行します。

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

bash scripts/inference/inference_from_config.sh configs/experiment/single_gpu_low_mem_inference.yaml \
  dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors" \
  inference.prompt=/path/to/prompts.json \
  inference.save_path=./outputs/single_gpu
```

CPU オフロードには、モデルを収められるだけのシステム RAM が必要で、CPU と GPU の間の転送によるオーバーヘッドも加わります。GPU メモリのピーク値は、解像度とセグメントの長さによって変わります。

#### Diffusers での推論

`HyNomasModularPipeline` を使うには、diffusers をアップグレードしてください。

```bash
pip install -e "diffusers[hy-nomas] @ git+https://github.com/jiaxiangc/diffusers.git@hy-nomas"
```

Hy-Nomas は、複数ターンのインタラクティブな生成に対応しています。この例では、5 秒のセグメントを 2 つ生成します。

```python
import json

import torch
from diffusers import HyNomasModularPipeline
from diffusers.utils import export_to_video

pipe = HyNomasModularPipeline.from_pretrained("tencent/Hy-Nomas-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to(device="cuda", dtype=torch.bfloat16)
# Use the FlashAttention and MagiAttention kernels installed with Hy-Nomas.
pipe.transformer.enable_reference_attention(text_backend="flash_varlen", deterministic=True)

prompts = [
    {
        "Style": "Cinematic fantasy", "perspective_view": "Third-person view",
        "subject_list": [{
            "subject_id": "fox", "display_name": "a small orange fox",
            "subject_attribute": "Orange fur, white chest, and a fluffy tail.",
        }],
        "segments": [{
            "visual_description_long": (
                "A small orange fox stands on a stone path in a luminous forest. "
                "Blue fireflies drift between the trees."
            ),
            "contained_actions": [{
                "start_time": "00:00.000", "end_time": "00:05.000",
                "action_description": "The fox walks slowly along the path.",
            }],
        }],
    },
    {
        "segments": [{
            "description_change": "Glowing blue flowers open beside the path as the fox continues forward.",
            "contained_actions": [{
                "start_time": "00:00.000", "end_time": "00:05.000",
                "action_description": "The fox passes the flowers and looks around.",
            }],
        }],
    },
]

videos = pipe(
    prompt=[json.dumps(prompt) for prompt in prompts],
    height=704,
    width=1280,
    num_frames=240,
    segment_latent_lengths=[30, 30],
    fps=24,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
    output="videos",
)
export_to_video(videos[0], "hy_nomas_video.mp4", fps=24, quality=None)
```

#### SGLang での推論

[SGLang Diffusion](https://github.com/sgl-project/sglang) は、Hy-Nomas をオフラインの動画ジョブとして、またセグメントを 1 つずつ受け取るリアルタイムセッションとして提供します。`tencent/Hy-Nomas` を直接読み込みます。初回の起動では、蒸留済み DiT、PixelUMM デコーダー、テキストエンコーダーをダウンロードし、ローカルのパイプラインをキャッシュします。

SGLang は、次の推論の最適化を加えます。

- **並列化：** DiT に対する Ulysses シーケンス並列とテンソル並列。PixelUMM のデコードウィンドウは GPU 間に分散されます。
- **メモリ：** DiT のレイヤー単位のオフロードが、ホストメモリからトランスフォーマーブロックをストリーミングします。
- **アテンション：** セグメントのアテンションは Hopper GPU 上で FlashAttention-3 で動作し、FlexAttention より約 2 倍高速です。ほかの GPU では FlexAttention がフォールバックとして働きます。
- **カーネル融合：** モジュレーション、RoPE 付きの QK ノルム、SwiGLU の融合した要素単位のカーネル。融合しない経路とビット単位で同一の結果になります。

SGLang は Docker イメージ（推奨）で動かすか、ソースからインストールしてください。Hy-Nomas への対応は、まだ安定版のリリースに入っていません。毎日 `main` から再ビルドされるナイトリーの `lmsysorg/sglang:dev` イメージを使い、Hy-Nomas を含むリリースが出たら `lmsysorg/sglang:latest` に切り替えてください。

```bash
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$HOME/.cache:/root/.cache" \
  lmsysorg/sglang:dev bash
```

`~/.cache` をマウントすると、ダウンロードした重みと組み立てたパイプラインが、コンテナをまたいで保持されます。代わりにソースからインストールするには、次のようにします。

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e "python[diffusion]" --prerelease=allow
```

コンテナまたはソースの環境の中で、GPU 1 枚、またはシーケンス並列を使った GPU 2 枚でサーバーを起動します。

```bash
sglang serve --model-path tencent/Hy-Nomas --num-gpus 1 --port 30000
# sglang serve --model-path tencent/Hy-Nomas --num-gpus 2 --sp-degree 2 --port 30000
```

リクエストには、フレーム数と構造化されたキャプションを持つセグメントを並べます。この例では、上の Diffusers の例の `prompts` を、120 フレームのセグメント 2 つとして送ります。

```python
import json
import time

import requests

base = "http://127.0.0.1:30000"
captions = [json.dumps(prompt) for prompt in prompts]
job = requests.post(f"{base}/v1/videos", json={
    "prompt": captions[0],
    "prompt_format": "json",
    "segments": [{"frames": 120, "caption": caption} for caption in captions],
    "num_frames": 240,
    "height": 704,
    "width": 1280,
    "fps": 24,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42,
}).json()
while job["status"] not in ("completed", "failed"):
    time.sleep(2)
    job = requests.get(f"{base}/v1/videos/{job['id']}").json()
assert job["status"] == "completed", job
with open("hy_nomas_sglang.mp4", "wb") as f:
    f.write(requests.get(f"{base}/v1/videos/{job['id']}/content").content)
```

同じサーバーは、`/v1/realtime_video/generate` でリアルタイムの WebSocket セッションも受け付けます。各セグメントは届き次第生成され、デコードされたフレームから順にストリーミングで返されます。リアルタイムのプロトコル、カメラ制御、レイヤー単位のオフロード、KV キャッシュの予算については、[SGLang の Hy-Nomas ガイド](https://docs.sglang.io/cookbook/diffusion/HY-Nomas/HY-Nomas)を参照してください。

### インタラクティブデモ
#### Reactor Runtime

[Reactor 統合ガイド](reactor/README.md)では、世界の履歴を保持したまま、プロンプトのライブ更新、WASD/矢印キーを組み合わせたカメラ操作、アップロードしたカメラ軌道に対応する、ストリーミングサービスとブラウザデモを提供します。検証済みの構成は、B200 GPU 8 枚、NVIDIA Container Toolkit 付きの Docker、[Reactor CLI](https://docs.reactor.inc/deploy/platform/installation)、Node.js 22 です。

このリポジトリのルートから：

```bash
cd reactor
reactor build
reactor run --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size 8g --port 8080
```

`curl -fsS http://localhost:8080/health` が `available` を返すまで待ちます。別のターミナルで、リポジトリのルートから：

```bash
cd reactor/demo
npm ci
npm run dev -- --port 5173
```

**http://localhost:5173** を開き、Runtime endpoint に **http://localhost:8080** を指定して **Connect** を押し、プロンプトを入力してから **Start world** を押します。動画をクリックして、WASD と矢印キーで探索します。**Update prompt** は、再起動せずにこれ以降のセグメントを変更します。GPU がリモートにある場合は、[SSH と WebRTC のガイダンス](reactor/README.md#remote-servers-and-ssh)を参照してください。

#### Web デモ

[オンラインのインタラクティブデモ](https://hy-nomas.hy.tencent.com/demo)をお試しください。

Web ゲートウェイと GPU ワーカーの起動、モデルアセットの設定、1 台または複数のホストでの実行については、[デモのデプロイガイド](apps/README.md)を参照してください。デモは既定で `configs/experiment/inference.yaml` を使います。

## LoRA ファインチューニング

### SFT データの準備

ローカルの動画から学習データを準備するには、[captioning-harness](captioning_harness/README.md) を使います。各セグメントにアノテーションを付け、動画フレームを DCAE-V でエンコードし、結果を SFT 用の Arrow または Parquet ファイルにまとめます。K3 の起動、アノテーションとパッキングの実行、正規化したカメラ軌道の付与については、ガイドを参照してください。

### オフラインのシーケンスパッキング

エンコード済みの学習ファイルから、パッキングプランを作成します。リポジトリのルートから実行します。

```bash
python scripts/data_processing/build_sft_packing_plan.py \
  --input /path/to/train.arrow \
  --output /path/to/train_packing.json \
  --max-seq-len 360000 \
  --num-groups 16 \
  --seed 42
```

この例では、DP16 × SP8（128 GPU）向けに 16 グループを作成します。グループ数は、データ並列度の倍数で、サンプル数を超えず、`--max-seq-len` のもとで十分な容量を持つ値を選んでください。学習コマンドでも同じ学習ファイルと行の順序を使い、出力された JSON を `dataset.token_packing_sorted_groups_path` として渡します。詳細は[シーケンスパッキング](captioning_harness/README.md#offline-sequence-packing)を参照してください。

### 学習とマージ

LoRA ファインチューニングは**蒸留していないチェックポイント**から始め、セグメントフォーシングの学習パイプラインを使います。学習後、学習した LoRA の重みを対応する**蒸留済みチェックポイント**にマージして、推論用の新しいマージ済みチェックポイントを作ります。

![LoRA ファインチューニング：蒸留していないチェックポイントで学習し、LoRA の重みを蒸留済みチェックポイントにマージします。](assets/lora-finetuning.svg)

`configs/experiment/sft.yaml` は、ランク/アルファ 256、ドロップアウトなしの LoRA と、一定の学習率 `1e-4` で 2,000 ステップの Muon を有効にします。学習されるのは LoRA パラメータだけです。

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

bash scripts/train/train_from_config.sh configs/experiment/sft.yaml \
  dit_weight="${Hy_Nomas_Model_Path}/DiT/base_ar/diffusion_pytorch_model_fp32.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt" \
  dataset.data_file=/path/to/train.arrow \
  dataset.token_packing_sorted_groups_path=/path/to/train_packing.json \
  output_dir=./outputs/segment_lora
```

学習の構成は SP8 を使います。128 GPU なら DP16 × SP8 になります。お使いのハードウェアに合わせて起動サイズを設定し、対応するデータ並列度に合わせたパッキングプランを作成してください。

保存した LoRA チェックポイントをエクスポートし、アーキテクチャとパラメータの形状が一致する蒸留済みチェックポイントにマージします。

```bash
export Hy_Nomas_Model_Path="${Hy_Nomas_Model_Path:-/path/to/Hy-Nomas}"

python -m scripts.train.export_lora_checkpoint \
  ./outputs/segment_lora/checkpoints/global_step-2000 \
  ./outputs/segment_lora/adapter

python -m scripts.train.merge_lora \
  --base "${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  --adapter ./outputs/segment_lora/adapter \
  --output ./outputs/merged_model.safetensors
```

[推論コマンド](#推論)では、`dit_weight=./outputs/merged_model.safetensors` を使ってください。

## ライセンス

このプロジェクトは、[Apache License 2.0](LICENSE) のもとでライセンスされています。

## 謝辞

[Gemma 4](https://deepmind.google/models/gemma/gemma-4/)、[Qwen3.8](https://qwen.ai/blog?id=qwen3.8)、[Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)、[DC-AE-V](https://github.com/dc-ai-projects/DC-VideoGen)、[MagiAttention](https://github.com/SandAI-org/MagiAttention) の開発チームの皆さんに、コミュニティへの貴重な貢献と、このプロジェクトを支えるモデル、研究、ツールについて感謝します。

## 引用

研究で Hy-Nomas が役に立った場合は、私たちのテクニカルレポートを引用してください。

```bibtex
@article{chen2026hy,
  title={Hy-Nomas: Towards Free-form Interactive World Generation},
  author={Chen, Qiaoling and Cheng, Jiaxiang and Deng, Xueqing and Hang, Tiankai and Hou, Ji and Li, Jiaqi and Liu, Yifan and Lu, Tao and Tang, Junshu and Tang, Yu and others},
  journal={arXiv preprint arXiv:2610.36729},
  year={2026}
}
```

## お問い合わせ

ご質問、セキュリティ以外のバグ報告、ご提案は、このリポジトリで issue を作成するか、[hy_nomas@tencent.com](mailto:hy_nomas@tencent.com) までご連絡ください。コントリビューションについては、[CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。脆弱性の疑いは、[SECURITY.md](SECURITY.md) に従って非公開で報告してください。

---

<p align="center">
    <i>Hy-Nomas · Free-Form Interactive World Generation.</i>
</p>
