"""Matched joint CFG5 with identical execution controls in both arms."""
LEGACY = 'joint_two_branch_cfg'
NATIVE_PLUS_PHYSICS = 'native_guidance_plus_conditional_physical_residual'

CONTROL_MODES = ('no_c', 'condition_off', 'global_off', 'local_off',
                 'global_direct_off', 'global_request_off', 'global_joint_off',
                 'value_correction_off', 'temporal_off',
                 'reasoning_swapped', 'empty_reasoning', 'global_train_mean')


def description(composition, guidance=5.):
    if composition not in (LEGACY, NATIVE_PLUS_PHYSICS):
        raise ValueError(composition)
    return dict(kind=composition, text_scale=guidance,
                physical_scale=guidance if composition == LEGACY else 1.,
                formula='v_full_neg + gamma*(v_full_pos-v_full_neg)' if composition == LEGACY
                else 'v_full_pos + (gamma-1)*(v_native_pos-v_native_neg)',
                physical_residual_scale=None if composition == LEGACY else 1.,
                native_definition='Same trained Wan LoRA with all physical additions disabled')


def execution_options(mode):
    """Same trained weights; remove one declared input/operator at a time."""
    if mode == 'no_c':
        return dict(physical_enabled=False)
    if mode == 'condition_off':
        return dict(condition_enabled=False)
    if mode == 'global_off':
        return dict(global_enabled=False)
    if mode == 'local_off':
        return dict(local_enabled=False)
    if mode == 'global_direct_off':
        return dict(global_direct_enabled=False)
    if mode == 'global_request_off':
        return dict(global_request_enabled=False)
    if mode == 'global_joint_off':
        return dict(global_joint_enabled=False)
    if mode == 'value_correction_off':
        return dict(value_correction_enabled=False)
    if mode == 'temporal_off':
        return dict(temporal_enabled=False)
    if mode in ('real', 'reasoning_swapped', 'empty_reasoning', 'global_train_mean'):
        return dict()
    raise ValueError(mode)


def guided_velocity(model, noisy, first, text, null, sigma, c, nc, times, groups, ngroups, *,
                    composition, guidance=5., mode='real'):
    if composition != LEGACY or guidance != 5.:
        raise ValueError('Matched generation and all declared controls retain historical joint CFG5')
    options = execution_options(mode)
    positive, _ = model.velocity(noisy, first, text, sigma, c, times, groups, **options)
    negative, _ = model.velocity(noisy, first, null, sigma, nc, times, ngroups, **options)
    return negative + guidance * (positive - negative)
