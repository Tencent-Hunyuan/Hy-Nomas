"""Released INT8/FP8 weights with architecture-matched inference kernels."""

from functools import lru_cache
from pathlib import Path
from types import MethodType


@lru_cache(maxsize=2)
def _extension(source: Path, cutlass: Path, capability: tuple[int, int]):
    from torch.utils.cpp_extension import load

    if capability == (10, 0):
        name = "hy_nomas_fp8_sm100"
        kernel = Path(__file__).with_name("hy_nomas_fp8_sm100.cu")
        architecture = "100a"
    elif capability == (8, 9):
        name = "hymm_pixelumm_cutlass_sm89_fp8"
        kernel = source / "hymm/models/vae/pixelumm_decoder/csrc/cutlass_fp8.cu"
        architecture = "89"
    else:
        raise ValueError("PixelUMM FP8 requires SM89 or SM100 in this build.")
    return load(
        name=name,
        sources=[str(kernel)],
        extra_include_paths=[str(cutlass / "include")],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
        ],
        verbose=False,
    )


def install_fp8(source: Path, cutlass: Path, capability: tuple[int, int]) -> None:
    """Retain native block quantization and serialized FP8/FP32 weight scales."""
    from hymm.models.vae.pixelumm_decoder import cutlass_fp8

    extension = _extension(source, cutlass, capability)
    cutlass_fp8.load_extension = lambda: extension


def cache_int8_weights(transformer) -> None:
    """Materialize frozen W8/BF16 weights once; keep both representations on GPU.

    This trades resident BF16 memory for avoiding per-forward dequantization.
    It is only used when the complete DiT remains resident throughout inference.
    """
    from hymm.models.dit.modules.int8_stored_bf16_linear import Int8StoredBF16Linear

    def weight(module):
        return module._resident_bf16

    for module in transformer.modules():
        if isinstance(module, Int8StoredBF16Linear):
            module.register_buffer(
                "_resident_bf16", module.dequant_weight(), persistent=False
            )
            module.dequant_weight = MethodType(weight, module)
