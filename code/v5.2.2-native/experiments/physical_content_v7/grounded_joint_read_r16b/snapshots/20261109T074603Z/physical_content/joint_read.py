"""B400 joint direct/request C values, with separate physical/native queries."""
import math
import torch


def read_physical(q,k,v,request_keys,reference,prior,mode='joint',*,request_query,relation_count):
    if mode not in {'joint','direct','request','absolute'}:raise ValueError(mode)
    scale=math.sqrt(q.shape[-1])
    direct=q.float()@k.float().transpose(-1,-2)/scale+prior[:-1]
    null=q.float()@reference.float().transpose(-1,-2)/scale+prior[-1:]
    if mode!='direct':
        binding=(request_keys.float()@k.float().transpose(-1,-2)/scale+prior[:-1]).softmax(-1)
        request_values=binding.to(v)@v
        request=request_query.float()@request_keys.float().transpose(-1,-2)/scale
        request=request+math.log(relation_count)-math.log(request_keys.shape[-2])
    if mode=='direct':logits,values=direct,v
    elif mode=='request':logits,values=request,request_values
    else:
        logits=torch.cat((direct,request),-1)-math.log(2.)
        values=torch.cat((v,request_values),-2)
    logits=torch.cat((logits,null),-1)
    values=torch.cat((values,torch.zeros_like(reference)),-2)
    return logits.softmax(-1).to(values)@values
