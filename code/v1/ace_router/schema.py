from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SAMPLE_ID_RE = re.compile(r"^P\d{2}$")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
BANNED_CONDITION_WORDS = {"not", "no", "without", "avoid", "must", "should"}
COUNTERFACTUAL_TYPES = {
    "outcome_preset",
    "trigger_missing",
    "cause_effect_reversal",
    "instant_replacement",
    "middle_deletion",
    "terminal_reversal",
    "unjustified_entity_loss",
    "no_op",
}

CAUSAL_REQUIRED = {
    "original_prompt",
    "event_type",
    "changed_state_variables",
    "preserved_context",
    "causal_positive",
    "causal_counterfactual",
    "counterfactual_type",
    "single_changed_relation",
    "warnings",
    "confidence",
}
I0_REQUIRED = {
    "original_prompt",
    "pre_event_image_prompt",
    "required_visible_entities",
    "initial_states",
    "layout_requirements",
    "protected_scene_properties",
    "transition_variables_not_to_freeze",
    "reject_if",
    "ambiguities",
}


@dataclass(frozen=True)
class AceSample:
    sample_id: str
    sample_dir: Path
    origin_path: Path
    causal_path: Path
    i0_metadata_path: Path
    image_path: Path | None
    selection_manifest_path: Path | None
    original_prompt: str
    causal_positive: str
    causal_counterfactual: str
    causal_metadata: dict[str, Any]
    i0_metadata: dict[str, Any]
    warnings: tuple[str, ...]

    @property
    def semantic_text(self) -> str:
        return self.original_prompt

    @property
    def positive_text(self) -> str:
        return self.causal_positive.strip()

    @property
    def counterfactual_text(self) -> str:
        return self.causal_counterfactual.strip()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _require_fields(data: dict[str, Any], required: set[str], path: Path) -> None:
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"missing fields in {path}: {missing}")


def _validate_condition(text: Any, field: str, path: Path) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field} must be a non-empty string in {path}")
    words = WORD_RE.findall(text)
    if len(words) > 32:
        raise ValueError(f"{field} exceeds 32 words in {path}: {len(words)}")
    banned = sorted({word.lower() for word in words} & BANNED_CONDITION_WORDS)
    if banned:
        raise ValueError(f"{field} contains banned words in {path}: {banned}")
    return text.strip()


def _require_nonempty_string(data: dict[str, Any], field: str, path: Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _require_string_list(
    data: dict[str, Any],
    field: str,
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier} in {path}")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain only non-empty strings in {path}")
    return [item.strip() for item in value]


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"initial image has invalid dimensions: {path}")
            image.convert("RGB").load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"initial image cannot be decoded as RGB: {path}: {exc}") from exc


def _find_image(sample_dir: Path, sample_id: str) -> Path | None:
    preferred = [
        sample_dir / f"{sample_id}-i0-1280x704.png",
        sample_dir / f"{sample_id}-i0-wan.png",
        sample_dir / f"{sample_id}-i0.png",
        sample_dir / f"{sample_id}-i0.jpg",
        sample_dir / f"{sample_id}-i0.jpeg",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_sample_ids(demo_root: str | Path) -> list[str]:
    root = Path(demo_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"demo root does not exist: {root}")
    sample_ids = sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and SAMPLE_ID_RE.fullmatch(child.name)
    )
    if not sample_ids:
        raise ValueError(f"no Pxx sample directories found under {root}")
    return sample_ids


def load_sample(
    demo_root: str | Path,
    sample_id: str,
    *,
    require_image: bool = False,
    inspect_image: bool = True,
    require_selection_manifest: bool = False,
    minimum_confidence: float = 0.7,
) -> AceSample:
    if not SAMPLE_ID_RE.fullmatch(sample_id):
        raise ValueError(f"invalid sample id {sample_id!r}; expected Pxx")
    sample_dir = Path(demo_root).expanduser().resolve() / sample_id
    origin_path = sample_dir / f"{sample_id}-origin.txt"
    causal_path = sample_dir / f"{sample_id}-cplus-cminus.json"
    i0_path = sample_dir / f"{sample_id}-i0.json"
    for required_path in (origin_path, causal_path, i0_path):
        if not required_path.is_file():
            raise ValueError(f"missing sample input: {required_path}")

    original_prompt = origin_path.read_text(encoding="utf-8").strip()
    if not original_prompt:
        raise ValueError(f"empty original prompt: {origin_path}")
    if "\n" in original_prompt or "\r" in original_prompt:
        raise ValueError(f"original prompt must be a single line: {origin_path}")

    causal = _load_json(causal_path)
    i0_metadata = _load_json(i0_path)
    _require_fields(causal, CAUSAL_REQUIRED, causal_path)
    _require_fields(i0_metadata, I0_REQUIRED, i0_path)
    if _require_nonempty_string(causal, "original_prompt", causal_path) != original_prompt:
        raise ValueError(f"causal original_prompt mismatch for {sample_id}")
    if _require_nonempty_string(i0_metadata, "original_prompt", i0_path) != original_prompt:
        raise ValueError(f"i0 original_prompt mismatch for {sample_id}")

    for field in ("event_type", "single_changed_relation"):
        _require_nonempty_string(causal, field, causal_path)
    for field in ("changed_state_variables", "preserved_context"):
        _require_string_list(causal, field, causal_path)
    _require_string_list(causal, "warnings", causal_path, allow_empty=True)

    positive = _validate_condition(
        causal["causal_positive"], "causal_positive", causal_path
    )
    counterfactual = _validate_condition(
        causal["causal_counterfactual"], "causal_counterfactual", causal_path
    )
    if positive == counterfactual:
        raise ValueError(f"positive and counterfactual conditions match for {sample_id}")
    if causal["counterfactual_type"] not in COUNTERFACTUAL_TYPES:
        raise ValueError(
            f"unsupported counterfactual_type for {sample_id}: "
            f"{causal['counterfactual_type']!r}"
        )
    confidence = causal["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError(f"confidence must be in [0, 1] for {sample_id}")
    if confidence < minimum_confidence:
        raise ValueError(
            f"confidence {confidence} is below {minimum_confidence} for {sample_id}"
        )

    _require_nonempty_string(i0_metadata, "pre_event_image_prompt", i0_path)
    for field in (
        "required_visible_entities",
        "initial_states",
        "layout_requirements",
        "protected_scene_properties",
        "transition_variables_not_to_freeze",
        "reject_if",
        "ambiguities",
    ):
        _require_string_list(
            i0_metadata,
            field,
            i0_path,
            allow_empty=field == "ambiguities",
        )

    if require_image and not inspect_image:
        raise ValueError("require_image=True requires inspect_image=True")
    image_path = _find_image(sample_dir, sample_id) if inspect_image else None
    if image_path is not None:
        _validate_image(image_path)
    if require_image and image_path is None:
        raise ValueError(f"I2V inference requires an initial image for {sample_id}")
    selection_path = sample_dir / f"{sample_id}-i0-selection.json"
    selection_manifest_path = (
        selection_path.resolve()
        if inspect_image and selection_path.is_file()
        else None
    )
    if require_selection_manifest and selection_manifest_path is None:
        raise ValueError(f"missing I0 selection manifest for {sample_id}: {selection_path}")

    warnings: list[str] = []
    if inspect_image and image_path is None:
        warnings.append("initial_image_missing")
    if inspect_image and selection_manifest_path is None:
        warnings.append("selection_manifest_missing")
    causal_warnings = causal.get("warnings")
    if isinstance(causal_warnings, list):
        warnings.extend(str(item) for item in causal_warnings if str(item).strip())

    return AceSample(
        sample_id=sample_id,
        sample_dir=sample_dir,
        origin_path=origin_path.resolve(),
        causal_path=causal_path.resolve(),
        i0_metadata_path=i0_path.resolve(),
        image_path=image_path,
        selection_manifest_path=selection_manifest_path,
        original_prompt=original_prompt,
        causal_positive=positive,
        causal_counterfactual=counterfactual,
        causal_metadata=causal,
        i0_metadata=i0_metadata,
        warnings=tuple(warnings),
    )
