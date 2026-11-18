"""Paired inference interventions; native inputs always belong to the case."""

MODES = ('normal', 'no_c', 'semantic_expert_off', 'scene_memory_off', 'no_video_feedback', 'no_temporal',
         'reasoning_swapped', 'empty_reasoning', 'global_train_mean')

def physics_evidence(own, replacement):
    """Replace physical analysis while retaining the current request words.

    I2V replacements are already re-encoded against the current image/request.
    Keep this case's explicit image-position memory as well as native inputs;
    only replace the listed physical analysis features.
    """
    result = dict(own)
    for key in ('words', 'word_groups', 'word_positions', 'word_strings',
                'native_words', 'native_condition', 'relations', 'analysis'):
        if key in replacement:
            result[key] = replacement[key]
        else:
            result.pop(key, None)
    return result


def fixed_content(own, replacement):
    """Install fixed C without replacing the current native text context."""
    return dict(own, routing=replacement['routing'], execution=replacement['execution'])


def velocity_mode(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return {'no_c': 'off', 'no_video_feedback': 'no_video_feedback', 'scene_memory_off': 'scene_memory_off',
            'no_temporal': 'no_temporal'}.get(mode, 'normal')
