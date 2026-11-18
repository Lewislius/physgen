"""One two-arm text/physics CFG; no third guidance branch."""


def description(scale=5.0, task='i2v'):
    positive = ('own first image, original request, positive native T5 and positive Qwen C'
                if task == 'i2v' else
                'request only, positive native T5 and positive request-only Qwen C')
    negative = ('same first image, official quality-negative native T5 and null Qwen C'
                if task == 'i2v' else
                'null request, null native T5 and null request-only Qwen C')
    return dict(kind='two_arm_joint_condition_cfg', scale=float(scale),
                formula=f'v = v_neg + {float(scale):g} * (v_pos - v_neg)',
                positive=positive, negative=negative,
                separate_physical_guidance=False)
