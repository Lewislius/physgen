"""Complete signed visible-process supervision on decoded generator outputs.

Every RGB time is supervised. Temporal VAE feature changes are never treated as
physical motion. Scales depend only on stopped reference changes, not predictions.
"""
import torch
from torch.nn import functional as F


def weighted_mean(value, valid):
    weight = torch.broadcast_to(valid.to(value), value.shape)
    return (value*weight).sum()/weight.sum().clamp_min(1e-8)


def features(frames, size, scale):
    shape = (max(1, size[0]//scale), max(1, size[1]//scale))
    return F.interpolate(frames.float(), size=shape, mode='bilinear', align_corners=False)


def pair_scale(squared_change, weight, floor):
    # Normalize each time pair independently. A strong early event must not set
    # the scale for all late pairs; stationary pairs remain bounded by the floor.
    energy=(squared_change*weight).sum((-2,-1))/weight.sum().clamp_min(1e-8)
    return energy.sqrt().clamp_min(floor).detach()[:,None,None,None]


def temporal_flow_error(prediction, target, valid, lags):
    """A fixed quadratic metric on errors of the native FM vector field.

    Differentiate along VIDEO time, not diffusion time. The target is still
    noise minus clean latent, never a physical speed. With FM this is positive
    definite and preserves the conditional-mean FM optimum: unlike a nonlinear
    decoded future regression, the metric does not depend on the target video.
    The observed first block has no prediction error and anchors long lags.
    """
    error = prediction.float()-target.detach().float()
    error = torch.cat((torch.zeros_like(error[:, :, :1]), error[:, :, 1:]), 2)
    values = [weighted_mean((error[:, :, lag:]-error[:, :, :-lag]).square().mean(1)/2, valid)
              for lag in lags if lag < error.shape[2]]
    if not values: raise ValueError('No temporal FM lag fits the video')
    return torch.stack(values).mean()


def sequence_error(prediction, reference, valid, cfg):
    reference = reference.detach()
    levels, changes = [], []
    for scale in cfg['scales']:
        p, y = features(prediction, cfg['size'], scale), features(reference, cfg['size'], scale)
        weight = F.interpolate(valid[None, None].float(), size=p.shape[-2:], mode='area')[0, 0]
        levels.append(weighted_mean(F.smooth_l1_loss(p, y, reduction='none').mean(1), weight))
        for lag in cfg['lags']:
            if lag >= len(y): continue
            py, yy = p[lag:]-p[:-lag], y[lag:]-y[:-lag]
            magnitude = pair_scale(yy.square().mean(1),weight,cfg['scale_floor'])
            changes.append(weighted_mean(F.smooth_l1_loss(py/magnitude, yy/magnitude, reduction='none').mean(1), weight))
    level, change = torch.stack(levels).mean(), torch.stack(changes).mean()
    alpha = cfg['state_weight']
    return alpha*level+(1-alpha)*change, dict(sequence_level=level.detach(), sequence_change=change.detach())


def perceptual_process_error(prediction, reference, valid, cfg):
    """Signed multihorizon changes in frozen pretrained spatial visual features."""
    reference = reference.detach()
    weight = F.interpolate(valid[None,None].float(),size=reference.shape[1:3],mode='area')[0,0]
    scale = weighted_mean(reference.square().mean(-1),weight).sqrt().clamp_min(.1).detach()
    p,y = prediction/scale,reference/scale
    level = weighted_mean(F.smooth_l1_loss(p,y,reduction='none').mean(-1),weight)
    changes=[]
    for lag in (1,2,4,8,16):
        if lag>=len(y): continue
        dp,dy = p[lag:]-p[:-lag],y[lag:]-y[:-lag]
        magnitude = pair_scale(dy.square().mean(-1),weight,cfg['scale_floor'])
        changes.append(weighted_mean(F.smooth_l1_loss(dp/magnitude,dy/magnitude,reduction='none').mean(-1),weight))
    change = torch.stack(changes).mean()
    alpha=cfg['state_weight']
    return alpha*level+(1-alpha)*change,dict(perceptual_level=level.detach(),perceptual_change=change.detach())


@torch.no_grad()
def sequence_metrics(prediction, reference, valid):
    result = {}
    weight = F.interpolate(valid[None, None].float(), size=prediction.shape[-2:], mode='area')[0, 0]
    square = lambda x: weighted_mean(x.float().square().mean(1), weight)
    for h in range(1, (len(reference)-1)//4+1):
        now, previous = h*4, (h-1)*4
        p, y = prediction[now:now+1], reference[now:now+1]
        dp, dy = p-prediction[previous:previous+1], y-reference[previous:previous+1]
        error, baseline = square(p-y), square(prediction[previous:previous+1]-y)
        pe, ye, de = square(dp), square(dy), square(dp-dy)
        cosine = weighted_mean((dp.float()*dy.float()).mean(1), weight)/(pe*ye).sqrt().clamp_min(1e-8)
        result.update({f'rgb/h{h}_mse': error, f'rgb/h{h}_previous_copy_mse': baseline,
                       f'rgb/h{h}_true_change_mse':ye,f'rgb/h{h}_predicted_change_mse':pe,
                       f'rgb/h{h}_change_error':de,
                       f'rgb/h{h}_gain_over_previous': (baseline-error)/baseline.clamp_min(1e-8),
                       f'rgb/h{h}_change_rms_ratio': (pe/ye.clamp_min(1e-8)).sqrt(),
                       f'rgb/h{h}_change_cosine': cosine, f'rgb/h{h}_change_skill': 1-de/ye.clamp_min(1e-8)})
    return result
