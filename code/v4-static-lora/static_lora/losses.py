"""Conditional flow-matching loss; I2V excludes the known first latent slice."""


def fm_loss(prediction, target, first_known):
    if first_known:
        prediction, target = prediction[:, :, 1:], target[:, :, 1:]
    return (prediction.float() - target.detach().float()).square().mean()
