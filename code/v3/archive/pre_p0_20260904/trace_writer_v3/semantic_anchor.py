from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from v2.ace_router.trace_schema import TracePlan

from .control_schema import EntityLedger, STAGE_IDS


_ARTICLE = re.compile(r"^(?:the\s+same|the|a|an)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class AnchorResult:
    global_anchor: str
    stage_anchors: Mapping[str, str]
    protected_coverage: Mapping[str, tuple[str, ...]]
    issues: tuple[str, ...]


def _clean_mention(value: str) -> str:
    return _ARTICLE.sub("", " ".join(value.split())).strip()


def _inventory(ledger: EntityLedger, stage_id: str | None = None) -> str:
    entries: list[str] = []
    presence = {item.entity_id: item for item in ledger.stage(stage_id)} if stage_id else {}
    for slot in ledger.slots:
        item = presence.get(slot.entity_id)
        state = ""
        if item is not None and item.state_facts:
            state = ", state " + ", ".join(item.state_facts)
        cardinality = item.logical_count if item is not None else slot.cardinality
        visibility = f", {item.visible}" if item is not None else ""
        label = slot.entity_id.replace("_", " ")
        entries.append(f"{label}: {cardinality.label()}{state}{visibility}")
    return "; ".join(entries)


def _protected_categories(text: str) -> tuple[str, ...]:
    folded = text.casefold()
    categories: list[str] = []
    if any(x in folded for x in ("one", "two", "count", "identity", "same")):
        categories.append("identity_count")
    if any(x in folded for x in ("color", "size", "material", "appearance", "lighting")):
        categories.append("appearance")
    if any(x in folded for x in ("shape", "round", "rectangular", "proportion", "geometry")):
        categories.append("geometry")
    if any(x in folded for x in ("camera", "background", "wall", "floor", "table", "framing")):
        categories.append("camera_background")
    if any(x in folded for x in ("no ", "does not", "unchanged", "remain")):
        categories.append("closed_world_or_persistence")
    return tuple(categories or ("generic_anchor",))


def compile_semantic_anchor(
    plan: TracePlan,
    ledger: EntityLedger,
    *,
    mode: str,
) -> AnchorResult:
    source = plan.states[0].facts if plan.states else {}
    target = plan.states[-1].facts if plan.states else {}
    protected = tuple(dict.fromkeys(x.strip() for x in plan.protected_predicates if x.strip()))
    coverage = {item: _protected_categories(item) for item in protected}
    forbidden = ledger.closed_world.forbidden_new_categories
    mode_clause = (
        "The reference first frame fixes these identities and initial layout."
        if mode == "i2v"
        else "All declared entities are fully visible from the first frame in the stated layout."
    )
    parts = [
        f"[ENTITY INVENTORY] The closed scene contains {_inventory(ledger)}.",
        mode_clause,
        f"[INITIAL STATE] {', '.join(f'{k}={v}' for k, v in source.items())}.",
        f"[PHYSICAL EVENT] {plan.global_semantic}",
    ]
    if mode == "t2v":
        parts.append(
            "[INITIAL LAYOUT] "
            + "; ".join(_clean_mention(slot.canonical_mention) for slot in ledger.slots)
            + "."
        )
    if protected:
        parts.append("[PERSISTENT PROTECTION] " + "; ".join(protected) + ".")
    if forbidden:
        parts.append(
            "[CLOSED WORLD] No additional " + ", ".join(forbidden) +
            " enters or appears; unoccupied scene regions remain empty."
        )
    conservation = [
        rule.observable_proxy
        for rule in ledger.conservation
    ]
    if conservation:
        parts.append(
            "[CONSERVATION] Source loss and result gain stay coupled: "
            + "; ".join(conservation)
            + ". No result appears from an undeclared external source."
        )
    parts.append(
        "[TERMINAL STATE] The same logical entity slots persist without duplicates; "
        + ", ".join(f"{k}={v}" for k, v in target.items())
        + " remains visible through the final frames."
    )
    global_anchor = " ".join(" ".join(part.split()) for part in parts if part.strip())

    stage_anchors: dict[str, str] = {}
    for stage_id in STAGE_IDS:
        stage_parts = [
            f"[STAGE ENTITY INVENTORY] Present now: {_inventory(ledger, stage_id)}.",
            "Counts and identities stay fixed; no duplicate state copy is created.",
        ]
        if forbidden:
            stage_parts.append("No additional " + ", ".join(forbidden) + " exists.")
        if protected:
            stage_parts.append(
                "All globally protected identity, shape, material, scene, and camera attributes stay unchanged."
            )
        if conservation:
            stage_parts.append(
                "Declared source loss and result gain remain coupled."
            )
        stage_anchors[stage_id] = " ".join(stage_parts)
    return AnchorResult(
        global_anchor=global_anchor,
        stage_anchors=stage_anchors,
        protected_coverage=coverage,
        issues=(),
    )
