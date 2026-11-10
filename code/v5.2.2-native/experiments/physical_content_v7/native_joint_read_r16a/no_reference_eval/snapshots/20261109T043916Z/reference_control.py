"""Evaluation-only removal of the final zero-value reference option."""
import torch


def without_reference(reader):
    def read(q, k, v, request_keys, reference, prior, mode='joint', *,
             request_query, relation_count):
        if mode != 'joint':
            raise ValueError('Only the native R16A reader is evaluated here')
        masked_prior = prior.clone()
        masked_prior[-1] = -torch.inf
        return reader(q, k, v, request_keys, reference, masked_prior, mode,
                      request_query=request_query, relation_count=relation_count)
    return read


def install():
    from physical_content import joint_read
    joint_read.read_physical = without_reference(joint_read.read_physical)
