from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


SAMPLE_RE = re.compile(r"^P\d{2}$")
EXPECTED_DEMO_IDS = tuple(f"P{index:02d}" for index in range(1, 21))
PLAN_SUFFIX_BY_MODE = {
    "i2v": "V2-planimg.json",
    "t2v": "V2-plan.json",
}
HIGH_RES_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class DemoInputError(ValueError):
    """The requested demo input set is incomplete or ambiguous."""


@dataclass(frozen=True)
class DemoTraceInput:
    sample_id: str
    plan_path: Path
    image_path: Path | None


def parse_sample_ids(value: str) -> tuple[str, ...]:
    """Parse a comma-separated sample selection; ``all`` means exactly P01..P20."""

    if not isinstance(value, str):
        raise DemoInputError("sample_ids must be a string")
    if value.strip().lower() == "all":
        return EXPECTED_DEMO_IDS
    sample_ids = tuple(
        item.strip().upper() for item in value.split(",") if item.strip()
    )
    if not sample_ids:
        raise DemoInputError("sample_ids must not be empty")
    invalid = [sample_id for sample_id in sample_ids if not SAMPLE_RE.fullmatch(sample_id)]
    if invalid:
        raise DemoInputError(
            "sample_ids must be comma-separated Pxx ids; invalid: "
            + ", ".join(invalid)
        )
    duplicates = sorted(
        sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
    )
    if duplicates:
        raise DemoInputError("duplicate sample_ids: " + ", ".join(duplicates))
    return sample_ids


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_plan_path(plan_root: Path, sample_id: str, mode: str) -> Path:
    """Resolve only the mode-specific V2 plan so M3 and M4 cannot be mixed."""

    plan_root = Path(plan_root).expanduser().resolve()
    try:
        suffix = PLAN_SUFFIX_BY_MODE[mode]
    except KeyError as exc:
        raise DemoInputError(f"unsupported inference mode {mode!r}") from exc
    filename = f"{sample_id}-{suffix}"
    path = _first_existing((plan_root / sample_id / filename, plan_root / filename))
    if path is None:
        variant = "planimg" if mode == "i2v" else "plan"
        raise DemoInputError(
            f"{sample_id}: missing {variant} plan {filename} under {plan_root}"
        )
    return path


def resolve_first_frame(demo_root: Path, sample_id: str) -> Path:
    """Resolve a single high-resolution ``*1280x704`` first frame.

    The current demo uses PNG files, while JPG/JPEG are accepted for newly
    prepared samples. Lower-resolution ``*-i0.png`` files are never selected.
    """

    sample_root = Path(demo_root).expanduser().resolve() / sample_id
    preferred = [
        sample_root / f"{sample_id}-i0-1280x704{suffix}"
        for suffix in HIGH_RES_IMAGE_SUFFIXES
    ]
    path = _first_existing(preferred)
    if path is not None:
        return path
    matches = sorted(
        candidate.resolve()
        for candidate in sample_root.glob("*1280x704.*")
        if candidate.is_file() and candidate.suffix.lower() in HIGH_RES_IMAGE_SUFFIXES
    )
    if not matches:
        raise DemoInputError(
            f"{sample_id}: missing high-resolution first frame ending in "
            "1280x704.jpg/.jpeg/.png"
        )
    if len(matches) > 1:
        raise DemoInputError(
            f"{sample_id}: ambiguous high-resolution first frames: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def resolve_demo_inputs(
    *,
    demo_root: str | Path,
    plan_root: str | Path,
    sample_ids: Iterable[str],
    mode: str,
) -> tuple[DemoTraceInput, ...]:
    """Resolve all requested inputs and report every missing item together."""

    demo = Path(demo_root).expanduser().resolve()
    plans = Path(plan_root).expanduser().resolve()
    if not demo.is_dir():
        raise DemoInputError(f"demo_root is not a directory: {demo}")
    if not plans.is_dir():
        raise DemoInputError(f"plan_root is not a directory: {plans}")

    resolved: list[DemoTraceInput] = []
    errors: list[str] = []
    for sample_id in sample_ids:
        try:
            plan_path = resolve_plan_path(plans, sample_id, mode)
            image_path = (
                resolve_first_frame(demo, sample_id) if mode == "i2v" else None
            )
            resolved.append(
                DemoTraceInput(
                    sample_id=sample_id,
                    plan_path=plan_path,
                    image_path=image_path,
                )
            )
        except DemoInputError as exc:
            errors.append(str(exc))
    if errors:
        raise DemoInputError(
            f"TRACE input preflight failed for {len(errors)} requested sample(s):\n- "
            + "\n- ".join(errors)
        )
    return tuple(resolved)
