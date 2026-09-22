from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


LANDSCAPE_REFERENCE_SIZE = (1280, 704)
PORTRAIT_REFERENCE_SIZE = (704, 1280)


@dataclass(frozen=True)
class ReferencePreparation:
    source_path: Path
    original_size: tuple[int, int]
    orientation: str
    crop_box: tuple[int, int, int, int]
    prepared_size: tuple[int, int]
    image: Image.Image


def best_output_size(
    width: int,
    height: int,
    width_multiple: int,
    height_multiple: int,
    expected_area: int,
) -> tuple[int, int]:
    if min(width, height, width_multiple, height_multiple, expected_area) <= 0:
        raise ValueError("image dimensions, multiples, and area must be positive")
    ratio = width / height
    float_width = (expected_area * ratio) ** 0.5
    float_height = expected_area / float_width

    width_first = int(float_width // width_multiple * width_multiple)
    height_first = int(expected_area / width_first // height_multiple * height_multiple)
    height_second = int(float_height // height_multiple * height_multiple)
    width_second = int(expected_area / height_second // width_multiple * width_multiple)
    if min(width_first, height_first, width_second, height_second) <= 0:
        raise ValueError("max_area is too small for the requested alignment")
    ratio_first = width_first / height_first
    ratio_second = width_second / height_second
    if max(ratio / ratio_first, ratio_first / ratio) < max(
        ratio / ratio_second, ratio_second / ratio
    ):
        return width_first, height_first
    return width_second, height_second


def _center_crop_box(
    width: int, height: int, target_width: int, target_height: int
) -> tuple[int, int, int, int]:
    target_ratio = target_width / target_height
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_height = height
        crop_width = max(1, round(height * target_ratio))
    else:
        crop_width = width
        crop_height = max(1, round(width / target_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def orientation_and_reference_size(image_path: str | Path) -> tuple[str, tuple[int, int]]:
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        width, height = image.size
    if width >= height:
        return "landscape", LANDSCAPE_REFERENCE_SIZE
    return "portrait", PORTRAIT_REFERENCE_SIZE


def prepare_reference_image(image_path: str | Path) -> ReferencePreparation:
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width >= height:
        orientation = "landscape"
        prepared_size = LANDSCAPE_REFERENCE_SIZE
    else:
        orientation = "portrait"
        prepared_size = PORTRAIT_REFERENCE_SIZE
    crop_box = _center_crop_box(width, height, *prepared_size)
    prepared = image.crop(crop_box).resize(prepared_size, Image.Resampling.LANCZOS)
    return ReferencePreparation(
        source_path=path,
        original_size=(width, height),
        orientation=orientation,
        crop_box=crop_box,
        prepared_size=prepared_size,
        image=prepared,
    )
