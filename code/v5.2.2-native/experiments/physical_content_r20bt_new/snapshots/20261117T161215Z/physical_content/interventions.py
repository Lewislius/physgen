"""Paired inference interventions; native inputs always belong to the case."""

MODES = ('normal', 'no_c', 'source_queries_off', 'physical_condition_off',
         'reasoning_swapped', 'empty_reasoning', 'global_train_mean')

def physics_evidence(own, replacement):
    """Replace physical analysis while retaining the current request words."""
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
    return {'no_c': 'off',
            'physical_condition_off': 'physical_condition_off'}.get(mode, 'normal')
