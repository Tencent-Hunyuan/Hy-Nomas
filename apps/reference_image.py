"""Validated local image uploads for the first interactive segment."""

from __future__ import annotations

import io
import json
import os
import re
import uuid
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.JpegImagePlugin import JpegImageFile
from hymm.utils.image_resize import fit_image_to_frame

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
REFERENCE_SECONDS = 5
# Visible T0 tail; the complete five-second segment remains in the model cache.
REFERENCE_VISIBLE_SECONDS = 0.5


def save_reference_image(data: bytes, output_dir: Path) -> dict:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a JPEG, PNG, or WebP image up to 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                # JPEGs with MPF/auxiliary-image metadata can be opened as MPO.
                # Its primary image is still a JPEG; use that image as the one
                # reference instead of rejecting the container's format label.
                is_jpeg = isinstance(source, JpegImageFile)
                if not is_jpeg and source.format not in {"PNG", "WEBP"}:
                    raise ValueError(
                        f"The file was detected as {source.format or 'an unknown format'}. "
                        "Please export it as JPEG, PNG, or WebP."
                    )
                if is_jpeg:
                    source.seek(0)
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Reference image must be at most 25 megapixels")
                if not is_jpeg and getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Choose a single still image")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid still image") from exc
    image = fit_image_to_frame(image, (1280, 720))
    image_id = uuid.uuid4().hex
    directory = output_dir / "reference_images"
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / f"{image_id}.jpg", format="JPEG", quality=95)
    return {
        "id": image_id,
        "url": f"./outputs/reference_images/{image_id}.jpg",
        "width": image.width,
        "height": image.height,
    }


def resolve_reference_image(image_id: object, output_dir: Path) -> Path:
    if not isinstance(image_id, str) or not re.fullmatch(r"[0-9a-f]{32}", image_id):
        raise ValueError("Invalid reference image ID")
    path = (output_dir / "reference_images" / f"{image_id}.jpg").resolve()
    if path.parent != (output_dir / "reference_images").resolve() or not path.is_file():
        raise ValueError("Reference image no longer exists; upload it again")
    return path


def save_caption_record(image_id, output_dir, record):
    path = resolve_reference_image(image_id, output_dir).with_suffix(".json")
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_caption_record(image_id, output_dir):
    path = resolve_reference_image(image_id, output_dir)
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"caption_status": "failed", "caption_error": "Upload the image again to caption it with K3"}
