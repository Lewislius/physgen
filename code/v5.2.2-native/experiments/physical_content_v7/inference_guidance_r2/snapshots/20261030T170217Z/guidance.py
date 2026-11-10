"""Separate native-text guidance from input-reasoning guidance.

neutral means the SAME positive native T5 request with the OWN image-only C.
It is neither a foreign C nor a future observation nor disabled injection.
"""


def combine(positive,negative,neutral,text_scale,physical_scale):
    if physical_scale==text_scale:
        return negative+text_scale*(positive-negative)
    base=negative+text_scale*(neutral-negative)
    if physical_scale==0:return base
    return base+physical_scale*(positive-neutral)
