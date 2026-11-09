# Third-party components

`vendor/video-pre-training/lib` (except `gym3_types.py`) contains runtime code derived from
[OpenAI Video PreTraining](https://github.com/openai/Video-Pre-Training),
copyright (c) 2022 OpenAI, under the MIT License. The license is preserved in
`vendor/video-pre-training/LICENSE`.

`vendor/video-pre-training/lib/gym3_types.py` contains type definitions from
[OpenAI gym3](https://github.com/openai/gym3), copyright (c) 2020 OpenAI,
under the MIT License. Its integer-product helper uses Python's `math.prod`.
The license is preserved in `vendor/gym3/LICENSE`. No gym3 installation is required.

The VPT runtime license applies to that third-party component; it does not
by itself assign a license to the model weights. Project code and model
weights are subject to the applicable project distribution terms.

CameraBench videos and labels are not included in this package.
