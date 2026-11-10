"""Hardware restriction for the user's blk-96g no-verifier experiments."""
import json
import re


def is_rtx_pro_6000(name):
    normalized = re.sub(r'[^A-Z0-9]', '', name.upper())
    return 'H100' not in normalized and ('RTXPRO6000' in normalized or 'RTX6000BLACKWELL' in normalized)


def require_rtx_pro_6000():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU policy: an allocated RTX PRO 6000 is required')
    name = torch.cuda.get_device_name(0)
    accepted = is_rtx_pro_6000(name)
    print(json.dumps(dict(kind='gpu_policy', required='RTX PRO 6000 / RTX 6000 Blackwell',
                          actual=name, accepted=accepted, device=0)), flush=True)
    if not accepted:
        raise RuntimeError(f'GPU policy rejected {name}: only RTX PRO 6000 / RTX 6000 Blackwell is allowed; exiting before model loading')
    return name
