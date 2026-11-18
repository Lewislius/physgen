"""One two-arm text/physics CFG; no third guidance branch."""


def description(scale=5.0):
    return dict(kind='two_arm_joint_condition_cfg', scale=float(scale),
                formula='v = v_neg + 5 * (v_pos - v_neg)',
                positive='own first image, original request, positive native T5 and positive Qwen C',
                negative='same first image, null native T5 and null Qwen C',
                separate_physical_guidance=False)
