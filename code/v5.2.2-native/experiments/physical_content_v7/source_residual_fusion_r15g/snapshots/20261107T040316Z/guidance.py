"""Explicitly recorded native-text and physical-velocity composition."""

NATIVE_PLUS_PHYSICS='native_guidance_plus_conditional_physical_residual'
LEGACY='joint_two_branch_cfg'


def description(composition,guidance=5.):
    if composition not in (NATIVE_PLUS_PHYSICS,LEGACY):raise ValueError(composition)
    return dict(kind=composition,text_scale=guidance,
        physical_scale=guidance if composition==LEGACY else 1.,
        formula=('v_full_pos + (gamma-1)*(v_native_pos-v_native_neg)'
                 if composition==NATIVE_PLUS_PHYSICS else 'v_full_neg + gamma*(v_full_pos-v_full_neg)'),
        physical_residual_scale=1. if composition==NATIVE_PLUS_PHYSICS else None,
        native_definition='Same trained Wan self-attention/FFN LoRA; physical branch bypassed')


def combine_native(positive,native_positive,native_negative,guidance=5.):
    return positive+(guidance-1.)*(native_positive-native_negative)


def guided_velocity(model,noisy,first,text,null,sigma,c,nc,times,groups,ngroups,*,
                    composition,guidance=5.,physical_enabled=True,native_physical_query=False):
    """Every component is evaluated at the same latent and denoising time."""
    options=dict(physical_enabled=physical_enabled,native_physical_query=native_physical_query)
    positive,_=model.velocity(noisy,first,text,sigma,c,times,groups,**options)
    if not physical_enabled or composition==LEGACY:
        negative,_=model.velocity(noisy,first,null,sigma,nc,times,ngroups,**options)
        return negative+guidance*(positive-negative)
    if composition!=NATIVE_PLUS_PHYSICS:raise ValueError(composition)
    native_positive,_=model.velocity(noisy,first,text,sigma,c,times,groups,physical_enabled=False)
    native_negative,_=model.velocity(noisy,first,null,sigma,nc,times,ngroups,physical_enabled=False)
    return combine_native(positive,native_positive,native_negative,guidance)
