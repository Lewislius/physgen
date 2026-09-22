from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import AceSample


CONDITIONING_MODES = frozenset({"initial_v1", "original", "compiled"})
COMPILER_VERSION = "ace-conditioning-mvp-v1"
INITIAL_V1_COMPAT_VERSION = "ace-router-initial-v1-compat"


def _clean_fragment(value: str) -> str:
    return value.strip().rstrip(". ;")


def _deduplicate(items: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_fragment(item)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def _sentence(label: str, items: tuple[str, ...]) -> str | None:
    if not items:
        return None
    return f"{label}: {'; '.join(items)}."


@dataclass(frozen=True)
class CompiledConditioning:
    sample_id: str
    mode: str
    semantic_variant: str
    compiler_version: str
    causal_origin_prompt_prefixed: bool
    semantic_text: str
    positive_text: str
    counterfactual_text: str
    sections: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "compiler_version": self.compiler_version,
            "sample_id": self.sample_id,
            "mode": self.mode,
            "semantic_variant": self.semantic_variant,
            "prompts": {
                "semantic": self.semantic_text,
                "causal_positive": self.positive_text,
                "causal_counterfactual": self.counterfactual_text,
            },
            "causal_pair": {
                "origin_prompt_prefixed": self.causal_origin_prompt_prefixed,
                "single_changed_relation": self.sections.get(
                    "single_changed_relation", ()
                ),
                "counterfactual_type": self.sections.get(
                    "counterfactual_type", ()
                ),
            },
            "source_sections": {
                name: list(values) for name, values in self.sections.items()
            },
        }


def compile_conditioning(
    sample: AceSample,
    *,
    mode: str,
    use_initial_image: bool,
) -> CompiledConditioning:
    """Compile existing metadata into deterministic model-facing text.

    This MVP intentionally does not implement image/video quality validation.
    Reject rules and ambiguities remain available in the source JSON but are not
    converted into generation-time negative prompts or automatic rejection.
    """

    if mode not in CONDITIONING_MODES:
        raise ValueError(
            f"unsupported conditioning mode {mode!r}; choose from "
            f"{sorted(CONDITIONING_MODES)}"
        )

    i0 = sample.i0_metadata
    causal = sample.causal_metadata
    sections = {
        "required_visible_entities": _deduplicate(
            list(i0["required_visible_entities"])
        ),
        "initial_states": _deduplicate(list(i0["initial_states"])),
        "layout_requirements": _deduplicate(list(i0["layout_requirements"])),
        "protected_scene_properties": _deduplicate(
            list(i0["protected_scene_properties"])
        ),
        "transition_variables": _deduplicate(
            list(i0["transition_variables_not_to_freeze"])
        ),
        "changed_state_variables": _deduplicate(
            list(causal["changed_state_variables"])
        ),
        "preserved_context": _deduplicate(list(causal["preserved_context"])),
        "event_type": (_clean_fragment(str(causal["event_type"])),),
        "single_changed_relation": (
            _clean_fragment(str(causal["single_changed_relation"])),
        ),
        "counterfactual_type": (
            _clean_fragment(str(causal["counterfactual_type"])),
        ),
    }

    if mode in {"initial_v1", "original"}:
        semantic_text = sample.original_prompt.strip()
        semantic_variant = mode
    else:
        preserved = _deduplicate(
            list(sections["protected_scene_properties"])
            + list(sections["preserved_context"])
        )
        prompt_parts = [sample.original_prompt.strip()]
        if not use_initial_image:
            prompt_parts.append(
                f"Initial scene: {_clean_fragment(str(i0['pre_event_image_prompt']))}."
            )
        compiled_sentences = [
            _sentence("Required visible entities", sections["required_visible_entities"]),
            _sentence("At the beginning", sections["initial_states"]),
            (
                _sentence("Required layout", sections["layout_requirements"])
                if not use_initial_image
                else None
            ),
            _sentence("Required state changes", sections["changed_state_variables"]),
            _sentence("These event variables are allowed to change", sections["transition_variables"]),
            _sentence("Preserve throughout", preserved),
        ]
        prompt_parts.extend(item for item in compiled_sentences if item)
        semantic_text = " ".join(prompt_parts)
        semantic_variant = "compiled_i2v" if use_initial_image else "compiled_t2v"

    # ``initial_v1`` reproduces the first ACE-Router pilot exactly: the full
    # original prompt was prefixed to both causal branches.  The newer modes
    # deliberately keep c+ and c- relation-only.
    causal_origin_prompt_prefixed = mode == "initial_v1"
    if causal_origin_prompt_prefixed:
        positive_text = f"{sample.original_prompt.strip()} {sample.causal_positive.strip()}"
        counterfactual_text = (
            f"{sample.original_prompt.strip()} "
            f"{sample.causal_counterfactual.strip()}"
        )
        compiler_version = INITIAL_V1_COMPAT_VERSION
    else:
        positive_text = sample.causal_positive.strip()
        counterfactual_text = sample.causal_counterfactual.strip()
        compiler_version = COMPILER_VERSION
    if positive_text == counterfactual_text:
        raise ValueError(f"causal pair must differ for {sample.sample_id}")

    return CompiledConditioning(
        sample_id=sample.sample_id,
        mode=mode,
        semantic_variant=semantic_variant,
        compiler_version=compiler_version,
        causal_origin_prompt_prefixed=causal_origin_prompt_prefixed,
        semantic_text=semantic_text,
        positive_text=positive_text,
        counterfactual_text=counterfactual_text,
        sections=sections,
    )
