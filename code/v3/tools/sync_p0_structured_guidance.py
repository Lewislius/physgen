#!/usr/bin/env python3
"""Author and synchronize explicit P01-P20 structured text guidance.

This catalog is intentionally authored field by field.  It never infers counts,
states, invariants, or forbidden changes from the free-form prompts.  ``--apply``
materializes the catalog into the prompt source and both direct plans; ``--check``
only verifies that all three copies are already identical.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/liuzhirui/Project/physGen")
SOURCE = ROOT / "prompt/p0_direct_prompts_p01_p20.json"
DEMO_ROOT = ROOT / "code/v1/demo"
STAGE_IDS = ("setup", "onset", "evolution", "completion", "terminal")


def entity(
    entity_id: str,
    label: str,
    *invariants: str,
    count: int = 1,
    persistent: bool = True,
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "label": label,
        "count": count,
        "persistent": persistent,
        "invariants": list(invariants),
    }


def state(
    entity_id: str,
    description: str,
    *,
    mutable: tuple[str, ...] = (),
    presence: str = "required",
    count: int = 1,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "presence": presence,
        "count": count,
        "states": [description],
        "mutable": list(mutable),
    }


def stage(
    *entity_states: dict[str, Any], relations: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    return list(entity_states), list(relations)


def guidance(
    *,
    entities: tuple[dict[str, Any], ...],
    initial_type: str,
    initial_facts: tuple[str, ...],
    stages: dict[str, tuple[list[dict[str, Any]], list[str]]],
    initial_stages: tuple[str, ...] = ("setup",),
) -> dict[str, Any]:
    if tuple(stages) != STAGE_IDS:
        raise ValueError(f"stage catalog must use {STAGE_IDS}, got {tuple(stages)}")
    return {
        "schema_version": "structured-text-guidance-v1",
        "entities": list(entities),
        "initial_condition": {
            "type": initial_type,
            "active_stages": list(initial_stages),
            "facts": list(initial_facts),
        },
        "stages": [
            {"id": stage_id, "entity_states": rows[0], "relations": rows[1]}
            for stage_id, rows in stages.items()
        ],
    }


# P02 is retained verbatim from the already audited source block below.  Every
# other entry is explicit here; helpers only remove JSON punctuation repetition.
CATALOG: dict[str, dict[str, Any]] = {
    "P01": guidance(
        entities=(
            entity(
                "scooter",
                "black stand-up electric kick scooter",
                "rigid frame and wheel geometry",
                "black appearance",
                "single intact identity",
            ),
            entity(
                "trash_can",
                "upright black slatted metal trash can",
                "rigid cylindrical geometry",
                "black slatted appearance",
                "single intact identity",
            ),
        ),
        initial_type="initial_momentum",
        initial_facts=(
            "The scooter already has rightward rolling momentum in the first video frame.",
            "The trash can starts motionless and upright.",
        ),
        stages={
            "setup": stage(
                state("scooter", "upright and rolling right", mutable=("horizontal position",)),
                state("trash_can", "motionless and upright"),
                relations=("The clear gap between scooter and trash can continuously narrows.",),
            ),
            "onset": stage(
                state(
                    "scooter",
                    "making one front-wheel contact with the trash can",
                    mutable=("rolling speed", "orientation"),
                    presence="transitioning",
                ),
                state("trash_can", "remaining upright at the same location"),
                relations=("This is the first and only impact between the two original objects.",),
            ),
            "evolution": stage(
                state(
                    "scooter",
                    "slowing and rotating down onto its right side",
                    mutable=("horizontal position", "rolling speed", "orientation"),
                ),
                state("trash_can", "remaining upright and intact"),
                relations=("The scooter tips beside the can without passing through or deforming it.",),
            ),
            "completion": stage(
                state("scooter", "finishing its fall on its right side and stopping", mutable=("orientation",)),
                state("trash_can", "still upright and motionless"),
                relations=("The two original objects remain separately visible beside each other.",),
            ),
            "terminal": stage(
                state("scooter", "lying motionless on its right side"),
                state("trash_can", "standing upright and motionless"),
                relations=("Exactly one intact scooter and one intact trash can remain visible.",),
            ),
        },
    ),
    "P03": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity"),
            entity("racket", "black tennis racket", "rigid frame geometry", "black appearance", "single identity"),
            entity("tennis_ball", "yellow tennis ball", "yellow fuzzy appearance", "single intact identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The hand and racket are already swinging rightward in the first video frame.",
            "The tennis ball approaches leftward from the right with a visible gap.",
        ),
        stages={
            "setup": stage(
                state("hand", "swinging the racket rightward", mutable=("position", "pose")),
                state("racket", "moving right with its strings facing the approaching ball", mutable=("position", "orientation")),
                state("tennis_ball", "moving left toward the racket", mutable=("horizontal position",)),
                relations=("A visible air gap separates the racket strings and ball.",),
            ),
            "onset": stage(
                state("racket", "meeting the ball at the center of its strings once", mutable=("position",)),
                state(
                    "tennis_ball",
                    "briefly compressing at the single string contact",
                    mutable=("travel speed", "travel direction", "elastic shape"),
                    presence="transitioning",
                ),
                relations=("The hand remains connected to the same racket during impact.",),
            ),
            "evolution": stage(
                state("racket", "finishing the rightward swing", mutable=("position", "orientation")),
                state("tennis_ball", "restoring its round form and reversing to move right", mutable=("horizontal position", "travel speed", "travel direction", "elastic shape")),
                relations=("The same ball separates from the strings after exactly one impact.",),
            ),
            "completion": stage(
                state("racket", "remaining to the left after the swing"),
                state("tennis_ball", "moving right away from the racket", mutable=("horizontal position",)),
                relations=("The air gap between racket and ball continuously grows.",),
            ),
            "terminal": stage(
                state("hand", "still holding the original racket at the left"),
                state("racket", "intact at the left"),
                state("tennis_ball", "round, intact, and separated to the right"),
                relations=("Exactly one hand-held racket and one original tennis ball remain visible.",),
            ),
        },
    ),
    "P04": guidance(
        entities=(
            entity("fingertip", "visible fingertip", "natural fingertip anatomy", "single identity", persistent=False),
            entity(
                "domino_row",
                "straight row of pale wooden dominoes",
                "fixed membership of the original dominoes",
                "pale wooden appearance",
                "left-to-right order",
            ),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "A fingertip initially touches and pushes only the leftmost domino.",
            "Every domino in the one row starts upright.",
        ),
        stages={
            "setup": stage(
                state("fingertip", "pushing only the first upright domino", mutable=("position",)),
                state("domino_row", "fully upright except for the first piece beginning to tilt", mutable=("first domino orientation",)),
                relations=("The fingertip does not touch any later domino.",),
            ),
            "onset": stage(
                state("fingertip", "withdrawing after the initial push", mutable=("position",), presence="transitioning"),
                state("domino_row", "starting one chain as the first domino contacts the second", mutable=("orientations of contacted dominoes",)),
                relations=("There is one left-to-right falling front and no second trigger.",),
            ),
            "evolution": stage(
                state("domino_row", "falling sequentially from left to right", mutable=("orientations of the advancing neighboring dominoes",)),
                relations=("Each falling domino contacts only its next neighbor in the same row.",),
            ),
            "completion": stage(
                state("domino_row", "reaching the last domino as that final piece falls", mutable=("last domino orientation",)),
                relations=("The chain reaches the right end exactly once without creating new pieces.",),
            ),
            "terminal": stage(
                state("domino_row", "fully down and motionless"),
                relations=("All original dominoes remain present in their original left-to-right row order.",),
            ),
        },
    ),
    "P05": guidance(
        entities=(
            entity("wood_block", "light-wood rectangular block", "rigid rectangular geometry", "light-wood appearance", "single intact identity"),
            entity("ramp", "wooden ramp", "fixed ramp geometry", "wooden appearance", "stationary identity"),
            entity("rough_floor", "broad rough gray floor", "flat fixed geometry", "rough gray appearance", "stationary identity"),
        ),
        initial_type="gravity_or_instability",
        initial_facts=(
            "The block starts on the upper ramp and gravity already pulls it down the rightward slope.",
            "The ramp and rough floor remain fixed.",
        ),
        stages={
            "setup": stage(
                state("wood_block", "sliding right down the upper ramp", mutable=("position along ramp", "sliding speed")),
                relations=("The block stays supported by the ramp surface.",),
            ),
            "onset": stage(
                state("wood_block", "crossing continuously from the ramp end onto the rough floor", mutable=("position", "sliding speed"), presence="transitioning"),
                relations=("The same block makes one support transition without jumping or duplicating.",),
            ),
            "evolution": stage(
                state("wood_block", "sliding right across the rough floor with decreasing speed", mutable=("horizontal position", "sliding speed")),
                relations=("Successive horizontal displacements become smaller under friction.",),
            ),
            "completion": stage(
                state("wood_block", "losing its remaining speed and stopping right of the ramp", mutable=("horizontal position", "sliding speed")),
                relations=("The block remains supported by the floor.",),
            ),
            "terminal": stage(
                state("wood_block", "intact and motionless on the rough floor"),
                relations=("Exactly one unchanged block remains to the right of the fixed ramp.",),
            ),
        },
    ),
    "P06": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity("book", "blue-edged book", "blue-edged cover appearance", "same complete set of pages", "single intact identity"),
            entity("shelf", "wooden shelf", "rigid fixed geometry", "wooden appearance", "stationary identity"),
            entity("floor", "wooden floor", "flat fixed geometry", "wooden appearance", "stationary identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The visible hand initially pushes the single closed book rightward.",
            "The shelf supports the book and the floor waits below the open gap.",
        ),
        stages={
            "setup": stage(
                state("hand", "pushing the closed book rightward", mutable=("position", "pose")),
                state("book", "closed and still supported by the shelf", mutable=("horizontal position",)),
                relations=("No second book or loose page appears.",),
            ),
            "onset": stage(
                state("book", "clearing the shelf edge while still closed and beginning to fall", mutable=("position", "vertical speed"), presence="transitioning"),
                relations=("The shelf becomes empty as the same book loses support once.",),
            ),
            "evolution": stage(
                state("book", "falling continuously through the open gap as one intact closed book", mutable=("position", "vertical speed", "orientation")),
                relations=("The book remains above the floor until landing.",),
            ),
            "completion": stage(
                state("book", "contacting the floor and opening once at the landing point", mutable=("position", "orientation", "cover opening angle"), presence="transitioning"),
                relations=("All pages stay attached to the one original book.",),
            ),
            "terminal": stage(
                state("book", "open, intact, and motionless on the wooden floor"),
                relations=("The shelf is empty and exactly one original book remains on the floor.",),
            ),
        },
    ),
    "P07": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity("box", "tall tan cardboard box", "rigid rectangular box geometry", "tan cardboard appearance", "single intact identity"),
            entity("floor", "smooth gray floor", "flat fixed geometry", "smooth gray appearance", "stationary identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The hand initially pulls the lower right corner of the upright box.",
            "The tall box begins with its base supported by the floor.",
        ),
        stages={
            "setup": stage(
                state("hand", "pulling the lower right corner", mutable=("position", "pose")),
                state("box", "upright with its weight still over its base", mutable=("base position",)),
                relations=("The hand acts on only the one original box.",),
            ),
            "onset": stage(
                state("box", "losing support as its upper edge starts moving right", mutable=("center position", "orientation"), presence="transitioning"),
                relations=("The box begins one continuous rightward tip without bending.",),
            ),
            "evolution": stage(
                state("box", "rotating continuously toward its broad right side", mutable=("position", "orientation", "angular speed")),
                relations=("The same box stays intact while approaching the floor.",),
            ),
            "completion": stage(
                state("hand", "releasing after the box lands", mutable=("position",), presence="transitioning"),
                state("box", "placing its broad right side on the floor and stopping rotation", mutable=("orientation", "angular speed")),
                relations=("The broad side contacts the floor once.",),
            ),
            "terminal": stage(
                state("box", "lying motionless on its right side"),
                relations=("Exactly one intact cardboard box remains on the fixed floor.",),
            ),
        },
    ),
    "P08": guidance(
        entities=(
            entity(
                "ice_cube",
                "clear solid ice portion from the original cube",
                "conserved original water material",
                "transparent appearance",
                "single material origin",
                persistent=False,
            ),
            entity(
                "meltwater",
                "connected transparent meltwater puddle from that ice",
                "conserved original water material",
                "transparent appearance",
                "one connected region",
                persistent=False,
            ),
            entity("plate", "shallow white plate", "rigid plate geometry", "white appearance", "stationary identity"),
        ),
        initial_type="environmental_field",
        initial_facts=(
            "Warm ambient sunlight supplies heat from the first video frame onward.",
            "One sharp solid ice cube starts on a dry plate with no visible meltwater.",
        ),
        stages={
            "setup": stage(
                state("ice_cube", "solid with sharp edges and resting on the plate"),
                state("meltwater", "not yet visible as a separate puddle", presence="absent", count=0),
                relations=("The plate surface around the ice is initially dry.",),
            ),
            "onset": stage(
                state("ice_cube", "rounding at its lower edges as melting begins", mutable=("solid shape", "solid volume", "phase allocation"), presence="transitioning"),
                state("meltwater", "appearing as one thin connected wet rim", mutable=("puddle extent", "liquid volume"), presence="transitioning"),
                relations=("The first liquid remains connected to the base of the same ice cube.",),
            ),
            "evolution": stage(
                state("ice_cube", "remaining solid but becoming progressively smaller", mutable=("solid shape", "solid volume", "phase allocation")),
                state("meltwater", "spreading outward as one wider transparent puddle", mutable=("puddle extent", "liquid volume")),
                relations=("As solid ice decreases, meltwater from it increases without a second cube or puddle.",),
            ),
            "completion": stage(
                state("ice_cube", "reduced to one small solid remnant", mutable=("solid shape", "solid volume")),
                state("meltwater", "forming one visibly larger connected puddle around the remnant", mutable=("puddle extent", "liquid volume")),
                relations=("The ice remnant stays inside its own meltwater on the plate.",),
            ),
            "terminal": stage(
                state("ice_cube", "remaining as one small transparent solid remnant"),
                state("meltwater", "remaining as one transparent connected puddle"),
                relations=("Both phases remain together on the one plate with no material duplication.",),
            ),
        },
    ),
    "P09": guidance(
        entities=(
            entity(
                "butter",
                "pale-yellow solid butter portion from the original pat",
                "conserved original butter material",
                "pale-yellow appearance",
                "single material origin",
                persistent=False,
            ),
            entity(
                "liquid_butter",
                "connected yellow liquid butter pool from that pat",
                "conserved original butter material",
                "yellow glossy appearance",
                "one connected region",
                persistent=False,
            ),
            entity("pan", "heated stainless-steel pan", "rigid pan geometry", "stainless-steel appearance", "stationary identity"),
        ),
        initial_type="environmental_field",
        initial_facts=(
            "The visibly heated pan supplies heat from the first video frame onward.",
            "One firm rectangular butter pat starts near the pan center with no liquid pool.",
        ),
        stages={
            "setup": stage(
                state("butter", "firm, rectangular, and sharp-edged in the pan"),
                state("liquid_butter", "not yet visible as a separate pool", presence="absent", count=0),
                relations=("The original butter material is entirely solid.",),
            ),
            "onset": stage(
                state("butter", "softening first at its lower corners", mutable=("solid shape", "solid height", "phase allocation"), presence="transitioning"),
                state("liquid_butter", "appearing as one glossy connected yellow rim", mutable=("pool extent", "liquid volume"), presence="transitioning"),
                relations=("The first liquid remains attached to the same butter pat.",),
            ),
            "evolution": stage(
                state("butter", "lowering and shrinking as the solid portion melts", mutable=("solid shape", "solid volume", "phase allocation")),
                state("liquid_butter", "spreading outward as one connected yellow pool", mutable=("pool extent", "liquid volume")),
                relations=("The shrinking solid and growing liquid are the same conserved butter material.",),
            ),
            "completion": stage(
                state("butter", "settling as one small soft center", mutable=("solid shape", "solid volume")),
                state("liquid_butter", "forming a wider connected pool around the soft center", mutable=("pool extent", "liquid volume")),
                relations=("No second pat or separate pool appears.",),
            ),
            "terminal": stage(
                state("butter", "fully incorporated into the melted material", presence="absent", count=0),
                state("liquid_butter", "remaining as one connected yellow pool near the pan center"),
                relations=("The original pat becomes one settled pool inside the same pan.",),
            ),
        },
    ),
    "P10": guidance(
        entities=(
            entity(
                "balloon",
                "tracked green rubber material from one balloon",
                "green rubber appearance",
                "conserved original rubber material",
                "single balloon origin with no duplicate balloon",
            ),
            entity("pump_nozzle", "black air-pump nozzle", "rigid nozzle geometry", "black appearance", "single identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The one small unknotted balloon starts intact and attached to the pump nozzle.",
            "Air supplied through that nozzle begins the inflation.",
        ),
        stages={
            "setup": stage(
                state("balloon", "forming one small loose wrinkled intact balloon", mutable=("surface wrinkles",)),
                state("pump_nozzle", "attached to the balloon opening"),
                relations=("No loose fragment is present before inflation.",),
            ),
            "onset": stage(
                state("balloon", "forming the same intact balloon as it begins enlarging", mutable=("enclosed volume", "rubber shape", "surface wrinkles"), presence="transitioning"),
                state("pump_nozzle", "supplying air through the same attachment"),
                relations=("The balloon remains sealed around the nozzle and does not duplicate.",),
            ),
            "evolution": stage(
                state("balloon", "forming one steadily larger, smoother, and tauter intact balloon", mutable=("enclosed volume", "rubber shape", "surface tension")),
                relations=("Only size, smoothness, and tension change before the burst.",),
            ),
            "completion": stage(
                state("balloon", "bursting once as the taut rubber separates into fragments", mutable=("rubber topology", "fragment positions"), presence="transitioning"),
                relations=("All green fragments originate from the one balloon at a single burst event.",),
            ),
            "terminal": stage(
                state("balloon", "remaining as the settled group of original green rubber fragments"),
                state("pump_nozzle", "remaining intact beside the fragments"),
                relations=("No intact second balloon appears after the burst.",),
            ),
        },
    ),
    "P11": guidance(
        entities=(
            entity(
                "bottle",
                "tracked clear glass material from one bottle",
                "transparent glass appearance",
                "conserved original glass material",
                "single bottle origin with no duplicate bottle",
            ),
            entity("shelf", "low shelf", "rigid fixed geometry", "stationary identity", "single support surface"),
            entity("tile_floor", "hard gray tile floor", "flat rigid geometry", "gray tile appearance", "stationary identity"),
        ),
        initial_type="gravity_or_instability",
        initial_facts=(
            "The one intact bottle is already slipping beyond the shelf edge in the first video frame.",
            "Gravity pulls it downward toward the hard tile floor.",
        ),
        stages={
            "setup": stage(
                state("bottle", "forming one intact bottle while slipping past the shelf edge", mutable=("position", "sliding speed")),
                relations=("Part of the bottle is still supported by the shelf.",),
            ),
            "onset": stage(
                state("bottle", "forming the same intact bottle as it fully clears the shelf", mutable=("position", "vertical speed", "orientation"), presence="transitioning"),
                relations=("The bottle loses shelf support once and begins a continuous fall.",),
            ),
            "evolution": stage(
                state("bottle", "forming one intact bottle descending through the open gap", mutable=("position", "vertical speed", "orientation")),
                relations=("The bottle remains above the tile floor until impact.",),
            ),
            "completion": stage(
                state("bottle", "striking the tile once and changing from one intact bottle into glass fragments", mutable=("glass topology", "fragment positions"), presence="transitioning"),
                relations=("All transparent fragments originate at one impact point from the original bottle.",),
            ),
            "terminal": stage(
                state("bottle", "remaining as the settled group of original transparent glass fragments"),
                relations=("No intact duplicate bottle remains; all fragments stay around the one impact point.",),
            ),
        },
    ),
    "P12": guidance(
        entities=(
            entity(
                "dandelion",
                "dry white dandelion seed head on its stem",
                "fixed stem and central head identity",
                "white natural appearance",
                "fixed rooted position",
            ),
            entity("wind", "steady left-to-right gust", "left-to-right direction", "continuous ambient cause", "single wind field"),
            entity(
                "seeds",
                "tracked group of original dandelion seeds",
                "fixed membership of the original seeds",
                "white tufted appearance",
                "single source dandelion",
            ),
        ),
        initial_type="environmental_field",
        initial_facts=(
            "A visible steady gust moves from left to right from the first video frame.",
            "All tracked seeds begin attached to the one full round dandelion head.",
        ),
        stages={
            "setup": stage(
                state("dandelion", "full and intact on its stem"),
                state("seeds", "attached together across the full seed head", mutable=("tuft motion",)),
                relations=("The rooted stem stays at the left while the breeze begins.",),
            ),
            "onset": stage(
                state("dandelion", "beginning to lose seeds from its right-facing side", mutable=("attached seed density",), presence="transitioning"),
                state("seeds", "beginning to detach from the right-facing side", mutable=("attachment state", "positions"), presence="transitioning"),
                relations=("The first detached seeds immediately follow the left-to-right wind.",),
            ),
            "evolution": stage(
                state("dandelion", "becoming progressively sparser while staying on the stem", mutable=("attached seed density",)),
                state("seeds", "separating progressively and drifting right", mutable=("attachment state", "positions", "group spread")),
                relations=("Every drifting seed comes from the same original head; none appear from elsewhere.",),
            ),
            "completion": stage(
                state("dandelion", "remaining as one visibly depleted head on its stem"),
                state("seeds", "forming one loose cloud farther to the right", mutable=("positions", "group spread")),
                relations=("The detached cloud and depleted head show the same one-way transfer of seeds.",),
            ),
            "terminal": stage(
                state("dandelion", "sparse and fixed on the original stem"),
                state("seeds", "continuing rightward as the original detached seed group", mutable=("positions",)),
                relations=("No new full dandelion head or additional seed source appears.",),
            ),
        },
    ),
    "P13": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity("pitcher", "transparent pitcher", "rigid pitcher geometry", "transparent appearance", "single intact identity"),
            entity("glass", "clear drinking glass", "rigid glass geometry", "clear appearance", "single intact identity"),
            entity(
                "water",
                "tracked body of clear water from the pitcher",
                "clear appearance",
                "conserved original water material",
                "fixed total water amount",
            ),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The hand starts holding the half-filled pitcher upright beside the empty glass.",
            "All tracked water starts inside the pitcher and no stream exists.",
        ),
        stages={
            "setup": stage(
                state("hand", "holding the pitcher upright", mutable=("pose",)),
                state("pitcher", "upright and half-filled"),
                state("glass", "empty and upright"),
                state("water", "entirely inside the pitcher with a still level"),
                relations=("A clear air gap separates the pitcher lip and glass; no stream is visible.",),
            ),
            "onset": stage(
                state("hand", "tilting the pitcher toward the glass", mutable=("pose", "position")),
                state("pitcher", "tilting while remaining intact", mutable=("orientation",)),
                state("glass", "receiving the first water while staying upright", mutable=("contained water level",)),
                state("water", "forming one continuous clear stream into the glass", mutable=("allocation between pitcher stream and glass", "free-surface levels", "flow shape"), presence="transitioning"),
                relations=("The stream connects the pitcher lip to the inside of the one receiving glass.",),
            ),
            "evolution": stage(
                state("pitcher", "remaining tilted as its waterline falls", mutable=("orientation", "contained water level")),
                state("glass", "remaining upright as its waterline rises", mutable=("contained water level",)),
                state("water", "flowing continuously from pitcher to glass", mutable=("allocation between pitcher stream and glass", "free-surface levels", "flow shape")),
                relations=("The pitcher level decreases while the glass level increases from the same water stream.",),
            ),
            "completion": stage(
                state("hand", "returning the pitcher upright to stop pouring", mutable=("pose", "position")),
                state("pitcher", "returning upright with less water", mutable=("orientation",)),
                state("glass", "holding more water while remaining upright"),
                state("water", "ending the stream and settling into the two containers", mutable=("flow shape", "free-surface levels"), presence="transitioning"),
                relations=("The one stream disappears only after the pitcher returns upright.",),
            ),
            "terminal": stage(
                state("pitcher", "upright with a lower still waterline"),
                state("glass", "upright with a higher still waterline"),
                state("water", "conserved between pitcher and glass with no stream"),
                relations=("Exactly one pitcher and one glass remain intact with both new levels visible.",),
            ),
        },
    ),
    "P14": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity("bottle", "tall clear juice bottle", "rigid bottle geometry", "clear container appearance", "single intact identity"),
            entity("glass", "clear tumbler", "rigid tumbler geometry", "clear appearance", "single intact identity"),
            entity(
                "juice",
                "tracked body of bright orange juice from the bottle",
                "bright orange appearance",
                "conserved original juice material",
                "fixed total juice amount",
            ),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The hand starts holding the partly filled bottle upright beside the empty tumbler.",
            "All tracked orange juice starts inside the bottle and no stream exists.",
        ),
        stages={
            "setup": stage(
                state("hand", "holding the bottle upright", mutable=("pose",)),
                state("bottle", "upright and partly filled"),
                state("glass", "empty and upright"),
                state("juice", "entirely inside the bottle with a still orange level"),
                relations=("A clear air gap separates the bottle lip and tumbler; no stream is visible.",),
            ),
            "onset": stage(
                state("hand", "tilting the bottle toward the tumbler", mutable=("pose", "position")),
                state("bottle", "tilting while remaining intact", mutable=("orientation",)),
                state("glass", "receiving the first juice while staying upright", mutable=("contained juice level",)),
                state("juice", "forming one continuous orange stream into the tumbler", mutable=("allocation between bottle stream and tumbler", "free-surface levels", "flow shape"), presence="transitioning"),
                relations=("The stream connects the bottle lip to the inside of the one receiving tumbler.",),
            ),
            "evolution": stage(
                state("bottle", "remaining tilted as its orange level falls", mutable=("orientation", "contained juice level")),
                state("glass", "remaining upright as its orange level rises", mutable=("contained juice level",)),
                state("juice", "flowing continuously from bottle to tumbler", mutable=("allocation between bottle stream and tumbler", "free-surface levels", "flow shape")),
                relations=("The bottle level decreases while the tumbler level increases from the same orange stream.",),
            ),
            "completion": stage(
                state("hand", "returning the bottle upright to stop pouring", mutable=("pose", "position")),
                state("bottle", "returning upright with less juice", mutable=("orientation",)),
                state("glass", "holding more juice while remaining upright"),
                state("juice", "ending the stream and settling into the two containers", mutable=("flow shape", "free-surface levels"), presence="transitioning"),
                relations=("The one orange stream disappears only after the bottle returns upright.",),
            ),
            "terminal": stage(
                state("bottle", "upright with a lower still orange level"),
                state("glass", "upright with a higher still orange level"),
                state("juice", "conserved between bottle and tumbler with no stream"),
                relations=("Exactly one bottle and one tumbler remain intact with both new levels visible.",),
            ),
        },
    ),
    "P15": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity("cup", "clear sand cup", "rigid cup geometry", "clear appearance", "single intact identity"),
            entity(
                "sand",
                "tracked amount of pale dry sand from the cup",
                "pale dry grain appearance",
                "conserved original grains",
                "fixed total sand amount",
            ),
            entity("table", "bare wooden table", "flat rigid geometry", "wooden appearance", "stationary identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The hand starts holding the full sand cup upright above the bare table.",
            "All tracked sand starts inside the cup and no grain stream or pile exists.",
        ),
        stages={
            "setup": stage(
                state("hand", "holding the full cup upright", mutable=("pose",)),
                state("cup", "upright and full of sand"),
                state("sand", "entirely inside the cup with no falling grains"),
                relations=("The table directly below is bare with no pre-existing pile.",),
            ),
            "onset": stage(
                state("hand", "tilting the cup over one fixed landing area", mutable=("pose", "position")),
                state("cup", "tilting while remaining intact", mutable=("orientation",)),
                state("sand", "beginning one continuous downward grain stream", mutable=("allocation between cup stream and pile", "grain positions", "flow shape"), presence="transitioning"),
                relations=("The first grains land directly below the cup and start one pile.",),
            ),
            "evolution": stage(
                state("cup", "remaining tilted as the sand level inside falls", mutable=("orientation", "contained sand level")),
                state("sand", "flowing downward while one conical pile grows below", mutable=("allocation between cup stream and pile", "grain positions", "pile height and width")),
                relations=("The cup loses sand at the same time the single pile gains it.",),
            ),
            "completion": stage(
                state("cup", "becoming nearly empty while still held above the pile"),
                state("sand", "sending the final grains into the pile and ending the stream", mutable=("grain positions", "pile height and width", "flow shape"), presence="transitioning"),
                relations=("No second pile or sideways grain source appears.",),
            ),
            "terminal": stage(
                state("cup", "empty and intact above the table"),
                state("sand", "settled as one conical pile on the table"),
                relations=("The original sand is conserved in exactly one settled pile beneath the cup.",),
            ),
        },
    ),
    "P16": guidance(
        entities=(
            entity(
                "dye",
                "tracked amount of dark-blue dye",
                "dark-blue material appearance",
                "conserved original dye",
                "fixed total dye amount",
            ),
            entity("water", "one body of initially clear still water", "same water body", "transparent uncolored region outside the dye", "contained identity"),
            entity("vessel", "rectangular glass vessel", "rigid rectangular geometry", "transparent glass appearance", "single intact identity"),
        ),
        initial_type="gravity_or_instability",
        initial_facts=(
            "The compact dark-blue dye drop is already falling under gravity toward the water surface.",
            "The water starts still and clear inside the one vessel.",
        ),
        stages={
            "setup": stage(
                state("dye", "compact and falling toward the flat water surface", mutable=("vertical position", "falling speed")),
                state("water", "still and clear below the separated drop"),
                relations=("An air gap separates the compact dye drop and water surface.",),
            ),
            "onset": stage(
                state("dye", "crossing the water surface at one visible entry point", mutable=("vertical position", "drop shape"), presence="transitioning"),
                state("water", "disturbed only at that one entry point", mutable=("local surface shape",), presence="transitioning"),
                relations=("The same drop enters the vessel without splashing outside it.",),
            ),
            "evolution": stage(
                state("dye", "spreading outward and downward from the entry point", mutable=("spatial extent", "concentration distribution", "shape")),
                state("water", "remaining inside the vessel as part of it becomes blue", mutable=("local color distribution",)),
                relations=("The colored region expands continuously from the original dye location.",),
            ),
            "completion": stage(
                state("dye", "forming one broad continuous blue region", mutable=("spatial extent", "concentration distribution")),
                state("water", "containing the broad dye region with clear water still around it", mutable=("local color distribution",)),
                relations=("All blue color remains within the glass vessel.",),
            ),
            "terminal": stage(
                state("dye", "remaining distributed as one expanded blue region inside the water"),
                state("water", "contained inside the intact vessel around the dye region"),
                relations=("No dye leaves the vessel and no second dye source appears.",),
            ),
        },
    ),
    "P17": guidance(
        entities=(
            entity("hand", "visible human hand", "natural hand anatomy", "single identity", persistent=False),
            entity(
                "sponge",
                "dry yellow rectangular sponge",
                "yellow porous material appearance",
                "intact sponge identity",
                "continuous material with no tearing",
            ),
            entity("surface", "white marble surface", "flat rigid geometry", "white marble appearance", "stationary identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The open hand begins above the sponge and supplies a straight downward pressing force.",
            "The dry sponge starts at its full uncompressed thickness on the marble surface.",
        ),
        stages={
            "setup": stage(
                state("hand", "moving straight down above the sponge", mutable=("vertical position", "pose")),
                state("sponge", "uncompressed at full thickness"),
                relations=("A small air gap initially separates the palm and sponge top.",),
            ),
            "onset": stage(
                state("hand", "making first palm contact and beginning to press", mutable=("vertical position", "pose"), presence="transitioning"),
                state("sponge", "beginning one elastic downward compression", mutable=("thickness", "elastic shape"), presence="transitioning"),
                relations=("The sponge remains supported by the marble and does not slide away.",),
            ),
            "evolution": stage(
                state("hand", "continuing steady downward pressure", mutable=("vertical position",)),
                state("sponge", "becoming visibly flatter under the palm", mutable=("thickness", "elastic shape")),
                relations=("Only elastic shape and thickness change; the sponge stays one intact object.",),
            ),
            "completion": stage(
                state("hand", "lifting fully clear of the sponge", mutable=("vertical position",), presence="transitioning"),
                state("sponge", "expanding upward toward its original thickness", mutable=("thickness", "elastic shape"), presence="transitioning"),
                relations=("Recovery begins only after pressure is removed.",),
            ),
            "terminal": stage(
                state("sponge", "recovered to full thickness and resting motionless"),
                relations=("Exactly one intact yellow sponge remains on the same marble surface.",),
            ),
        },
    ),
    "P18": guidance(
        entities=(
            entity(
                "ball",
                "red rubber ball",
                "red rubber appearance",
                "intact ball identity",
                "equilibrium size between contacts",
            ),
            entity("floor", "hard gray floor", "flat rigid geometry", "hard gray appearance", "stationary identity"),
        ),
        initial_type="initial_momentum",
        initial_facts=(
            "The one red ball is already falling vertically toward the floor in the first video frame.",
            "The hard floor remains fixed below the clear vertical path.",
        ),
        stages={
            "setup": stage(
                state("ball", "intact and descending vertically toward the floor", mutable=("vertical position", "vertical speed")),
                relations=("A visible air gap below the ball continuously narrows.",),
            ),
            "onset": stage(
                state("ball", "making its first floor contact and briefly compressing elastically", mutable=("vertical speed", "travel direction", "elastic shape"), presence="transitioning"),
                relations=("The ball contacts the floor at one fixed impact area without penetrating it.",),
            ),
            "evolution": stage(
                state("ball", "rebounding repeatedly to progressively lower peak heights", mutable=("vertical position", "vertical speed", "travel direction", "elastic shape at contacts")),
                relations=("Each bounce peak is lower than the preceding peak and stays near the same vertical path.",),
            ),
            "completion": stage(
                state("ball", "ending its final small rebound at the impact area", mutable=("vertical position", "vertical speed"), presence="transitioning"),
                relations=("The repeated rebounds decay into rest without a new ball appearing.",),
            ),
            "terminal": stage(
                state("ball", "round, full-sized, intact, and motionless on the floor"),
                relations=("Exactly one original red ball remains at the impact area.",),
            ),
        },
    ),
    "P19": guidance(
        entities=(
            entity("left_hand", "visible left hand", "natural hand anatomy", "left-hand identity", "single identity"),
            entity("right_hand", "visible right hand", "natural hand anatomy", "right-hand identity", "single identity"),
            entity(
                "paper",
                "tracked white paper material from one notched sheet",
                "white paper appearance",
                "conserved original paper material",
                "single-sheet origin and exactly two terminal pieces",
            ),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The two hands begin pulling opposite side edges of the same sheet apart.",
            "The paper starts as one intact sheet except for one centered top notch.",
        ),
        stages={
            "setup": stage(
                state("left_hand", "holding and pulling the left edge leftward", mutable=("horizontal position", "pose")),
                state("right_hand", "holding and pulling the right edge rightward", mutable=("horizontal position", "pose")),
                state("paper", "forming one taut intact notched sheet", mutable=("tension",)),
                relations=("Only the small pre-existing top notch is open; no long tear exists yet.",),
            ),
            "onset": stage(
                state("left_hand", "continuing the leftward pull", mutable=("horizontal position", "pose")),
                state("right_hand", "continuing the rightward pull", mutable=("horizontal position", "pose")),
                state("paper", "starting one tear downward from the existing notch", mutable=("tear length", "side separation"), presence="transitioning"),
                relations=("There is exactly one tear front and it begins at the notch.",),
            ),
            "evolution": stage(
                state("paper", "lengthening the same tear continuously toward the opposite edge", mutable=("tear length", "side separation")),
                relations=("The two sides move farther apart while remaining connected ahead of the one tear front.",),
            ),
            "completion": stage(
                state("paper", "reaching the opposite edge and changing from one sheet into exactly two pieces", mutable=("paper topology", "piece separation"), presence="transitioning"),
                relations=("The original one sheet creates exactly two main pieces and no extra scraps.",),
            ),
            "terminal": stage(
                state("left_hand", "holding the left original paper piece"),
                state("right_hand", "holding the right original paper piece"),
                state("paper", "remaining as exactly two visibly separated pieces"),
                relations=("Both pieces together contain the conserved material of the one original sheet.",),
            ),
        },
    ),
    "P20": guidance(
        entities=(
            entity("hand", "visible gloved hand", "glove appearance", "single hand identity", persistent=False),
            entity("kettle", "stainless-steel kettle", "rigid kettle geometry", "stainless-steel appearance", "single intact identity"),
            entity("cup", "gray camping cup", "rigid cup geometry", "gray appearance", "single intact identity"),
            entity(
                "coffee",
                "tracked body of dark coffee from the kettle",
                "dark coffee appearance",
                "conserved original coffee material",
                "fixed total coffee amount",
            ),
            entity("campfire", "small campfire", "single fixed fire location", "orange flame appearance", "continuous burning identity"),
        ),
        initial_type="visible_actuator",
        initial_facts=(
            "The gloved hand starts holding the kettle upright beside the empty cup on snow.",
            "All tracked coffee starts inside the kettle with no stream, while the one campfire already flickers.",
        ),
        stages={
            "setup": stage(
                state("hand", "holding the kettle upright", mutable=("pose",)),
                state("kettle", "upright with all coffee inside"),
                state("cup", "empty and upright on the snow"),
                state("coffee", "entirely inside the kettle with no stream"),
                state("campfire", "flickering at the fixed left location", mutable=("flame shape", "flame brightness")),
                relations=("The kettle lip and cup are separated by air before pouring.",),
            ),
            "onset": stage(
                state("hand", "tilting the kettle over the cup", mutable=("pose", "position")),
                state("kettle", "tilting while remaining intact", mutable=("orientation",)),
                state("cup", "receiving the first coffee while staying upright", mutable=("contained coffee level",)),
                state("coffee", "forming one continuous dark stream into the cup", mutable=("allocation between kettle stream and cup", "free-surface levels", "flow shape"), presence="transitioning"),
                state("campfire", "continuing to flicker at the left", mutable=("flame shape", "flame brightness")),
                relations=("The stream connects the kettle spout to the inside of the one cup.",),
            ),
            "evolution": stage(
                state("kettle", "remaining tilted as coffee leaves it", mutable=("orientation", "contained coffee level")),
                state("cup", "remaining upright as its coffee level rises", mutable=("contained coffee level",)),
                state("coffee", "flowing continuously from kettle to cup", mutable=("allocation between kettle stream and cup", "free-surface levels", "flow shape")),
                state("campfire", "continuing to flicker independently at the left", mutable=("flame shape", "flame brightness")),
                relations=("The cup gains the same coffee that leaves the kettle; the fire does not move or multiply.",),
            ),
            "completion": stage(
                state("hand", "returning the kettle upright to stop pouring", mutable=("pose", "position")),
                state("kettle", "returning upright with less coffee", mutable=("orientation",)),
                state("cup", "holding the poured coffee while remaining upright"),
                state("coffee", "ending the stream and settling in the cup and kettle", mutable=("flow shape", "free-surface levels"), presence="transitioning"),
                state("campfire", "continuing to flicker at the same location", mutable=("flame shape", "flame brightness")),
                relations=("The dark stream disappears after the kettle returns upright.",),
            ),
            "terminal": stage(
                state("kettle", "upright and intact beside the cup"),
                state("cup", "upright with a still dark coffee level"),
                state("coffee", "conserved with no stream and visibly present in the cup"),
                state("campfire", "still flickering at the fixed left location", mutable=("flame shape", "flame brightness")),
                relations=("Exactly one filled cup and one continuously flickering campfire remain visible on the snow.",),
            ),
        },
    ),
}


def insert_after_entities(document: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    inserted = False
    for key, item in document.items():
        if key == "structured_guidance":
            continue
        result[key] = item
        if key == "entities":
            result["structured_guidance"] = deepcopy(value)
            inserted = True
    if not inserted:
        raise ValueError("document has no entities field")
    return result


def materialized_documents() -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    samples = {item["sample_id"]: item for item in source["samples"]}
    expected = {f"P{index:02d}" for index in range(1, 21)}
    if set(samples) != expected:
        raise ValueError("prompt source must contain exactly P01-P20")
    if "structured_guidance" not in samples["P02"]:
        raise ValueError("audited P02 structured_guidance is missing")
    full_catalog = dict(CATALOG)
    full_catalog["P02"] = samples["P02"]["structured_guidance"]
    if set(full_catalog) != expected:
        missing = sorted(expected - set(full_catalog))
        raise ValueError(f"structured guidance catalog is incomplete: {missing}")

    new_samples = []
    plans: list[tuple[Path, dict[str, Any]]] = []
    for sample_id in sorted(samples):
        authored = full_catalog[sample_id]
        declared = {item["id"] for item in samples[sample_id]["entities"]}
        structured = {item["id"] for item in authored["entities"]}
        if declared != structured:
            raise ValueError(
                f"{sample_id}: structured entity ids {sorted(structured)} do not match "
                f"declared ids {sorted(declared)}"
            )
        new_samples.append(insert_after_entities(samples[sample_id], authored))
        for suffix in ("plan", "planimg"):
            path = DEMO_ROOT / sample_id / f"{sample_id}-V2-{suffix}.json"
            plan = json.loads(path.read_text(encoding="utf-8"))
            plan_declared = {item["id"] for item in plan["entities"]}
            if plan_declared != declared:
                raise ValueError(f"{path}: entity ids differ from prompt source")
            plans.append((path, insert_after_entities(plan, authored)))
    source["samples"] = new_samples
    return source, plans


def serialized(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source, plans = materialized_documents()
    targets = [(SOURCE, source), *plans]
    stale = [path for path, value in targets if path.read_text(encoding="utf-8") != serialized(value)]
    if args.check:
        if stale:
            raise SystemExit("stale structured guidance: " + ", ".join(map(str, stale)))
        print(f"[OK] {len(targets)} JSON documents contain synchronized P01-P20 guidance")
        return
    for path, value in targets:
        path.write_text(serialized(value), encoding="utf-8")
    print(f"[OK] synchronized explicit structured guidance into {len(targets)} JSON documents")


if __name__ == "__main__":
    main()
