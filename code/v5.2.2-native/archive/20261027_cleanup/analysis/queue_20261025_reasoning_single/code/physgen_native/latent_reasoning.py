"""Grounded continuous reasoning with Qwen's own late decoder layers.

The memory contains only the legal first image, text and physical analysis. We
reuse pretrained hidden-space operators instead of creating a new transformer
or feeding final normalized states back into word-embedding space. No cache is
mutated, and every refinement stays in the generation-loss computation graph.
"""
from contextlib import contextmanager
import torch


@contextmanager
def capture_native_memory(language, depth):
    captured = {}
    def memory_hook(module, args): captured['memory'] = args[0]
    def output_hook(module, args): captured['raw_output'] = args[0]
    handles = [language.layers[-depth].register_forward_pre_hook(memory_hook),
               language.norm.register_forward_pre_hook(output_hook)]
    try: yield captured
    finally:
        for handle in handles: handle.remove()


def refine_dependency_latent(language, memory, initial, positions, attention_mask,
                             *, steps, depth, return_states=False):
    """Reuse native late Qwen blocks on a grounded memory and recurrent C state.

    Memory retains its native multimodal RoPE. Working tokens have ordinary text
    positions after that memory. Their raw hidden scale matches the native late
    layers; the native final normalization is applied only at the output.
    """
    if steps < 1 or not 1 <= depth <= len(language.layers):
        raise ValueError('Positive latent reasoning steps and valid native depth required')
    batch, count, _ = initial.shape
    prefix = memory.shape[1]
    extension = positions.amax(dim=(0, 2), keepdim=True)+torch.arange(1, count+1, device=positions.device)[None, None]
    position_ids = torch.cat((positions, extension.expand(positions.shape[0], batch, -1)), -1)
    rotary_ids = position_ids[-3:]
    total = prefix+count
    allowed = torch.ones(total, total, device=memory.device, dtype=torch.bool).tril()
    mask = memory.new_zeros(batch, 1, total, total).masked_fill(~allowed[None, None], float('-inf'))
    mask[:, :, :, :prefix] = mask[:, :, :, :prefix].masked_fill(~attention_mask[:, None, None].bool(), float('-inf'))
    state = initial
    states = [state]
    for _ in range(steps):
        hidden = torch.cat((memory, state), 1)
        rotary = language.rotary_emb(hidden, rotary_ids)
        for layer in language.layers[-depth:]:
            # Native checkpoint wrappers preserve the active knowledge-LoRA
            # role. No detached state, cloned decoder, or mutable KV cache.
            hidden = layer(hidden, position_embeddings=rotary, attention_mask=mask,
                           position_ids=position_ids[0], past_key_values=None, use_cache=False)
        state = hidden[:, -count:]
        states.append(state)
    result = language.norm(state)
    return (result, states) if return_states else result
