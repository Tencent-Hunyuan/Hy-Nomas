"""Fit a complete image to a video frame without cropping or stretching."""

from PIL import Image, ImageOps


def fit_image_to_frame(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Apply camera orientation, preserve aspect ratio, and pad with black."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(
        image, size, method=Image.Resampling.LANCZOS,
        color=(0, 0, 0), centering=(0.5, 0.5),
    )
