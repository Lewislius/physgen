import torch


@torch.no_grad()
def similarity(a, b, weight, scale, instances=None):
    """JEPA diagnostics are descriptive, never physical-success labels or training losses."""
    a, b = a.float().reshape(16, -1, a.shape[-1]), b.float().reshape(16, -1, b.shape[-1])
    w = weight.float().reshape(1, -1, 1).expand(16, -1, a.shape[-1])
    def metrics(x, y, weights):
        denom = weights.sum().clamp_min(1e-12)
        mse = ((x - y).square() * weights).sum() / denom / (scale * scale + 1e-8)
        dot = (x * y * weights).sum()
        cosine = dot / ((x.square() * weights).sum().sqrt() * (y.square() * weights).sum().sqrt()).clamp_min(1e-12)
        return float(mse), float(cosine)
    mse, cosine = metrics(a, b, w)
    def dynamic_similarity(weights):
        # Instance visibility can vary in time: remove the mean over observed times only.
        count = weights.sum(0, keepdim=True).clamp_min(1e-12)
        da = a - (a * weights).sum(0, keepdim=True) / count
        db = b - (b * weights).sum(0, keepdim=True) / count
        dynamic_a = ((da.square() * weights).sum() / weights.sum().clamp_min(1e-12)).sqrt()
        dynamic_b = ((db.square() * weights).sum() / weights.sum().clamp_min(1e-12)).sqrt()
        value = None if float(torch.minimum(dynamic_a, dynamic_b)) <= float(scale) * 1e-6 else metrics(da, db, weights)[1]
        return dict(dynamic_cosine=value, dynamic_status="not_applicable_near_static" if value is None else "defined")
    result = dict(mse_normalized=mse, cosine=cosine, **dynamic_similarity(w))
    if instances is not None:
        result["instances"] = []
        for i, mask in enumerate(instances):
            weights = w * mask.reshape(16, -1, 1)
            if float(weights.sum()) > 0:
                local_mse, local_cos = metrics(a, b, weights)
                result["instances"].append(dict(index=i, mse_normalized=local_mse, cosine=local_cos,
                                                **dynamic_similarity(weights)))
    return result


def state_relations(result, target, geo, scale, instances=None):
    states = {name: similarity(result[name], target, geo.p_weight, scale, instances) for name in ("p0", "p5", "p15")}
    states["gain_0_to_5"] = states["p0"]["mse_normalized"] - states["p5"]["mse_normalized"]
    states["gain_5_to_15"] = states["p5"]["mse_normalized"] - states["p15"]["mse_normalized"]
    return states
