"""Matched joint CFG5 with identical execution controls in both arms."""
LEGACY = 'joint_two_branch_cfg'
NATIVE_PLUS_PHYSICS = 'native_guidance_plus_conditional_physical_residual'


def description(composition, guidance=5.):
    if composition not in (LEGACY, NATIVE_PLUS_PHYSICS):
        raise ValueError(composition)
    return dict(kind=composition, text_scale=guidance,
                physical_scale=guidance if composition == LEGACY else 1.,
                formula='v_full_neg + gamma*(v_full_pos-v_full_neg)' if composition == LEGACY
                else 'v_full_pos + (gamma-1)*(v_native_pos-v_native_neg)',
                physical_residual_scale=None if composition == LEGACY else 1.,
                native_definition='Same trained Wan LoRA with all physical additions disabled')


def guided_velocity(model, noisy, first, text, null, sigma, c, nc, times, groups, ngroups, *,
                    composition, guidance=5., physical_enabled=True, native_physical_query=False,
                    memory_mode="joint", state_value_enabled=True, shared_physical_query=False):
    if composition != LEGACY or guidance != 5.:
        raise ValueError('Matched generation and all declared controls retain historical joint CFG5')
    options = dict(physical_enabled=physical_enabled, native_physical_query=native_physical_query,
                   memory_mode=memory_mode, state_value_enabled=state_value_enabled,
                   shared_physical_query=shared_physical_query)
    positive, _ = model.velocity(noisy, first, text, sigma, c, times, groups, **options)
    negative, _ = model.velocity(noisy, first, null, sigma, nc, times, ngroups, **options)
    return negative + guidance * (positive - negative)
