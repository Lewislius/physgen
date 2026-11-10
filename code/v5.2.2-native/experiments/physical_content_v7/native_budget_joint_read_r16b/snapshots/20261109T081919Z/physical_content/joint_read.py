"""Native bank allocation; learned physical selection within direct C.

Equal native/physical queries recover B400+I joint attention. No request-prior
product and no subtraction of the common physical write.
"""
import math
import torch


def read_physical(q,k,v,request_keys,reference,prior,mode='joint',*,request_query,relation_count):
    if mode not in {'joint','direct','request'}:raise ValueError(mode)
    scale=math.sqrt(q.shape[-1]);n=k.shape[-2]
    if mode!='direct':
        binding=(request_keys.float()@k.float().transpose(-1,-2)/scale+prior[:-1]).softmax(-1)
        request_values=binding.to(v)@v
        request_prior=prior.new_full((request_keys.shape[-2],),math.log(relation_count)-math.log(request_keys.shape[-2]))
    if mode=='direct':keys,values,physical_prior=k,v,prior[:-1].float()
    elif mode=='request':keys,values,physical_prior=request_keys,request_values,request_prior.float()
    else:
        keys=torch.cat((k,request_keys),-2);values=torch.cat((v,request_values),-2)
        physical_prior=torch.cat((prior[:-1],request_prior),-1).float()-math.log(2.)
    keys=torch.cat((keys,reference),-2)
    values=torch.cat((values,torch.zeros_like(reference)),-2)
    native=(request_query.float()@keys.float().transpose(-1,-2))/scale
    native=native+torch.cat((physical_prior,prior[-1:].float()),-1)
    if mode!='request':
        physical=(q.float()@k.float().transpose(-1,-2))/scale+physical_prior[:n]
        physical=physical+native[...,:n].logsumexp(-1,keepdim=True)-physical.logsumexp(-1,keepdim=True)
        logits=torch.cat((physical,native[...,n:]),-1)
    else:logits=native
    return logits.softmax(-1).to(values)@values
