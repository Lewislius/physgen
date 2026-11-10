"""Explicit architectures for the isolated five-way Q workspace experiment."""
import math
import torch


Q_WORKSPACE_VARIANTS = {
    'v522_native_q_content_pool_b_v31': dict(
        workspace_variant='b', query_initialization='content_pool_plus_seed',
        self_attention=True, cross_query_identity=False, self_query_identity=False),
    'v522_native_q_content_pool_no_self_c_v32': dict(
        workspace_variant='c', query_initialization='content_pool_plus_seed',
        self_attention=False, cross_query_identity=False, self_query_identity=False),
    'v522_native_q_identity_no_self_dn_v33': dict(
        workspace_variant='d-n', query_initialization='seed',
        self_attention=False, cross_query_identity=True, self_query_identity=False),
    'v522_native_q_identity_self_ds_v34': dict(
        workspace_variant='d-s', query_initialization='seed',
        self_attention=True, cross_query_identity=True, self_query_identity=False),
    'v522_native_q_identity_qk_self_di_v35': dict(
        workspace_variant='d-i', query_initialization='seed',
        self_attention=True, cross_query_identity=True, self_query_identity=True),
}


def validate_variant(cfg):
    expected = Q_WORKSPACE_VARIANTS.get(cfg['revision'])
    if expected is None:
        if 'workspace_variant' in cfg.get('message', {}):
            raise ValueError('A Q workspace variant requires its own checkpoint revision')
        return
    message, knowledge = cfg['message'], cfg['knowledge']
    for key, value in expected.items():
        if message.get(key) != value or type(message.get(key)) is not type(value):
            raise ValueError(f'{cfg["revision"]} requires message.{key}={value!r}')
    if message.get('content_pooling') != 'ordered_disjoint_mean_repeat_short':
        raise ValueError('Use the declared valid-token pooling rule')
    if message.get('identity_normalization') != 'rms_no_affine':
        raise ValueError('Query identity has parameter-free per-slot RMS normalization')
    if message.get('identity_start_round') != 1:
        raise ValueError('R0 keeps the original seed query; identity is explicit in R1/R2')
    if not math.isfinite(message.get('identity_eps', 0)) or message['identity_eps'] <= 0:
        raise ValueError('identity_eps must be finite and positive')
    if knowledge['reasoning_mode'] != 'q_seed128_message' or knowledge['relation_tokens'] != 128:
        raise ValueError('All five variants have exactly 128 Q slots')
    if (knowledge['latent_reasoning_steps'] != 2 or knowledge['train_reasoning_rounds'] != [2]
            or knowledge['inference_reasoning_round'] != 2):
        raise ValueError('These training recipes use R0 plus two refinement rounds')
    if cfg['train']['seed'] != 42 or cfg['validation'].get('sample_seeds') != [42]:
        raise ValueError('Training and generated validation samples use seed 42')
    if cfg['validation'].get('seed_mode') != 'fixed_training':
        raise ValueError('Validation uses the training seed without per-case offsets')


def pool_valid_content(content, valid, slots=128):
    """Length-based initialization only; the complete attention memory is retained.

    Compact valid tokens in their original order, then use disjoint intervals
    [floor(j*N/S), floor((j+1)*N/S)). For N<S, repeat the nearest token at each
    output-bin centre. Short inputs do not acquire invented semantic content.
    Accumulation is FP32 and the output retains the input compute dtype.
    """
    if content.ndim != 3 or valid.shape != content.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Expected content [B,T,D] and boolean valid [B,T]')
    if not isinstance(slots, int) or slots < 1:
        raise ValueError('slots must be positive')
    outputs = []
    for row, mask in zip(content, valid):
        values = row[mask].float()
        count = values.shape[0]
        if count == 0:
            raise ValueError('Content pooling requires at least one valid token')
        if count < slots:
            indices = ((2*torch.arange(slots, device=row.device)+1)*count)//(2*slots)
            pooled = values[indices]
        else:
            edges = torch.arange(slots+1, device=row.device)*count//slots
            prefix = torch.cat((values.new_zeros(1, values.shape[-1]), values.cumsum(0)))
            pooled = (prefix[edges[1:]]-prefix[edges[:-1]])/(edges[1:]-edges[:-1])[:, None]
        outputs.append(pooled.to(content.dtype))
    return torch.stack(outputs)


def normalized_query_identity(seed, eps=1e-6):
    """Keep gradients into E; do not add parameters, cross-slot centring or roles."""
    value = seed.float()
    return (value*torch.rsqrt(value.square().mean(-1, keepdim=True)+eps)).to(seed.dtype)
