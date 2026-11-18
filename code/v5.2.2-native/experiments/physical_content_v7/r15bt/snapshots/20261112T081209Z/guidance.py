"""Two-branch CFG description retained from R15B; task scope is T2V."""


def description(text_scale, physical_scale):
    return dict(kind='separate_native_text_and_reasoning', text_scale=text_scale,
        physical_scale=physical_scale,
        neutral='Same positive native T5 request, own request-only negative C',
        legacy_equivalence='Physical scale equal to text scale is exactly the original two-branch formula')


def combine(positive, negative, neutral, text_scale, physical_scale):
    if physical_scale == text_scale:
        return negative + text_scale * (positive - negative)
    base = negative + text_scale * (neutral - negative)
    if physical_scale == 0:
        return base
    return base + physical_scale * (positive - neutral)
