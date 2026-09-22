"""Small CPU algebra checks; deliberately not a real-Wan performance claim."""
import json
import torch
from writer_projection import SharedWriterProjection

torch.set_num_threads(2)
torch.manual_seed(20260921)
dtype = torch.float64
truth = torch.randn(4,6,dtype=dtype)*.05
fit = SharedWriterProjection(6,4)
for count in (20,31,43):
    z = torch.randn(1,count,6,dtype=dtype)
    e = z @ truth.T
    mask = torch.ones(1,count,1,dtype=dtype)
    mask[:,:3] = 0
    e[:,:3] = 1000.  # Protected positions must not affect the fit.
    fit.add(z,e,mask)
update, report = fit.propose(relative_ridge=1e-9)
assert torch.allclose(update, truth, atol=1e-8, rtol=1e-7)
writer = torch.nn.Linear(6,4,bias=False,dtype=dtype)
unseen = torch.randn(1,19,6,dtype=dtype)
base = writer(unseen).detach()
with torch.no_grad():
    writer.weight.add_(update)
normal_output = writer(unseen)
heldout_relative_error = float(((normal_output-base-unseen@truth.T).square().sum()
                              /(unseen@truth.T).square().sum()).detach())
assert heldout_relative_error < 1e-12
# Contradictory sample-specific Oracle labels cannot be represented by one W.
z = torch.randn(1,20,6,dtype=dtype)
e = z@truth.T
mask = torch.ones(1,20,1,dtype=dtype)
conflict = SharedWriterProjection(6,4)
conflict.add(z,e,mask)
conflict.add(z,-e,mask)
cancelled, conflicting_report = conflict.propose()
assert torch.equal(cancelled,torch.zeros_like(cancelled))
assert abs(conflicting_report['relative_fit']-1) < 1e-12
result = dict(scope='CPU linear algebra only; no Wan/real-video efficacy claim',
              shared_parameter_mapping_correct=True, protected_tokens_excluded=True,
              ordinary_forward_uses_updated_parameters=True,
              synthetic_unseen_relative_error=heldout_relative_error,
              opposing_targets_not_claimed_learned=True, fit=report, conflict=conflicting_report)
print(json.dumps(result,indent=2))
