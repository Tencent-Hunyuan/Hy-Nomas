import pytest
from omegaconf import OmegaConf

from apps.seg_ar_demo_config import launcher_overrides, load_demo_config


@pytest.mark.parametrize(
    ("environment", "flag", "field"),
    [
        ("DIT_WEIGHT", "--dit-weight", "dit_weight"),
        ("SAVE_PATH", "--save-path", "inference.save_path"),
        ("TRAIN_OUTPUT", None, "output_dir"),
        ("DRY_RUN_SOURCE", "--dry-run-source", "inference.prompt"),
        (None, "--text-encoder-model", "text_encoder.model_name"),
    ],
)
@pytest.mark.parametrize("path", ['/tmp/模型 "文件"', '/tmp/model "file"'])
def test_launcher_preserves_unicode_paths(environment, flag, field, path):
    if environment:
        config = load_demo_config(launcher_overrides([], {environment: path}))
        assert OmegaConf.select(config, field) == path
    if flag:
        defaults = {environment: "/tmp/default"} if environment else {}
        config = load_demo_config(launcher_overrides([flag, path], defaults))
        assert OmegaConf.select(config, field) == path
