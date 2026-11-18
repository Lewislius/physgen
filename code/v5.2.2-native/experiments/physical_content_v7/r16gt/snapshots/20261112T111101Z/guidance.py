"""Matched joint CFG5 for the R16Gt T2V candidate.

The R16G I2V runs use the historical joint two-branch composition: one CFG
over the full positive and negative physical evidences.  R16Gt retains the
identical formula, with the negative arm using the null request and its own
request-only negative C.  No first frame exists in either arm.
"""
LEGACY = 'joint_two_branch_cfg'
NATIVE_PLUS_PHYSICS = 'native_guidance_plus_conditional_physical_residual'

CONTROL_MODES = ('no_c', 'condition_off', 'global_off', 'local_off',
                 'global_direct_off', 'global_request_off', 'global_joint_off',
                 'temporal_off')


def description(guidance=5.):
    return dict(kind=LEGACY, text_scale=guidance, physical_scale=guidance,
                formula='v_full_neg + gamma*(v_full_pos-v_full_neg)',
                physical_residual_scale=None,
                native_definition='Same trained weights with all physical additions disabled')


def execution_options(mode):
    """Same trained weights; remove one declared input/operator at a time."""
    if mode in ('real',):
        return dict()
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
    if mode == 'temporal_off':
        return dict(temporal_enabled=False)
    raise ValueError(mode)


def guided_velocity(model, noisy, text, null, sigma, c, nc, times, groups,
                    ngroups, *, guidance=5., mode='real'):
    """Historical joint CFG5 with identical execution controls in both arms."""
    if guidance != 5.:
        raise ValueError('Matched generation and all declared controls retain historical joint CFG5')
    options = execution_options(mode)
    positive, _ = model.velocity(noisy, None, text, sigma, c, times, groups, **options)
    negative, _ = model.velocity(noisy, None, null, sigma, nc, times, ngroups, **options)
    return negative + guidance * (positive - negative)
