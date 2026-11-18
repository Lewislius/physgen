"""Static configuration for attention map generation."""

PHENOMENON_DICT = {
    0: "rigid body motion",
    1: "collision",
    2: "liquid motion",
    3: "gas motion",
    4: "elastic motion",
    5: "deformation",
    7: "melting",
    8: "solidification",
    9: "vaporization",
    10: "liquefaction",
    11: "combustion",
    12: "explosion",
    14: "reflection",
    15: "refraction",
    16: "scattering",
    17: "interference and diffraction",
    18: "unnatural light source",
}

APPEARANCE_DICT = {
    22: "liquid objects",
    23: "solid objects",
    24: "gas objects",
    25: "object decomposition and splitting",
    26: "mixing of multiple objects",
    27: "object disappearance",
}

SYSTEM_PROMPT = "You are a helpful assistant."
DESCRIBE_QUESTION = "Describe the video briefly."
PHENOMENON_QUESTION_TEMPLATE = (
    "Describe the {label} phenomenon briefly. Don't describe anything else."
)
APPEARANCE_QUESTION_TEMPLATE = "Describe the {label} briefly. Don't describe anything else."
CUSTOM_ATTRIBUTE_QUESTION_TEMPLATE = (
    "Describe the physical attribute '{label}' briefly. Don't describe anything else."
)

VIDEO_TOKEN_ID = 151656
EOS_TOKEN_ID = 151645

