"""CPU small-tensor checks for algebra/optimizer contracts; NOT real Wan tests."""
from pathlib import Path
import math, json, copy
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
ROOT=Path(__file__).resolve().parents[1]
results=[]
def ck(name, value):
    ok=bool(value)
    results.append({'name':name,'passed':ok})
    if not ok: raise AssertionError(name)

def lr_mult(s):
    if not 1<=s<=1000: raise ValueError('optimizer update must be 1..1000')
    if s<=64:return s/64
    if s<=200:return 1.
    return .2+.8*.5*(1+math.cos(math.pi*(s-200)/800))

def route_weight(s):
    if not 1<=s<=1000: raise ValueError('update must be 1..1000')
    if s<=64:return .03*min(s/32,1.)
    if s<320:return .03*(320-s)/256
    return 0.

class ToyLoRA(nn.Module):
    def __init__(self,base,rank=2):
        super().__init__();self.base=base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b)

def norm0(x):return F.layer_norm(x,(x.shape[-1],),eps=1e-6)

def d_executor(p,h,s,w):
    # Deliberately simplified single-head algebra; no native RoPE/kernel validation.
    q=F.linear(norm0(h+p),w['q'])+F.linear(norm0(s),w['uq'])
    k=F.linear(norm0(h+p),w['k'])+F.linear(norm0(s),w['uk'])
    v=F.linear(p,w['v'])
    att=(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
    p1=p+F.linear(att@v,w['o'])
    z=norm0(p1)*(1+F.linear(norm0(s),w['g']))
    p2=p1+F.linear(F.gelu(F.linear(z,w['f1'])),w['f2'])
    return F.linear(p2,w['write'])

def gen_loss(pred,noise,clean,valid):
    e=(pred[:,:,1:]- (noise-clean)[:,:,1:]).square().mean(1)
    m=valid[:,None,:,:].expand_as(e)
    mean=(e*m).sum()/m.sum()
    t,h,w=e.shape[-3:]
    terms=[]
    for window in [(1,h,w),(min(4,t),h,w),(min(4,t),max(1,h//5),max(1,w//4)),(t,max(1,h//5),max(1,w//4))]:
        n=F.avg_pool3d((e*m)[:,None],window,stride=window,ceil_mode=True)
        d=F.avg_pool3d(m[:,None],window,stride=window,ceil_mode=True)
        values=(n/d.clamp_min(1e-8))[d>0]
        terms.append(values.topk(max(1,math.ceil(values.numel()*.25))).values.mean())
    tail=torch.stack(terms).mean()
    return .8*mean+.2*tail,mean,tail

def main():
    ck('lr_first_update',abs(lr_mult(1)-1/64)<1e-12)
    ck('lr_peak_64',lr_mult(64)==1)
    ck('lr_peak_200',lr_mult(200)==1)
    ck('lr_end_1000',abs(lr_mult(1000)-.2)<1e-12)
    ck('lr_decay_monotone',all(lr_mult(i)>=lr_mult(i+1) for i in range(200,1000)))
    for s in (0,1001):
        failed=False
        try:lr_mult(s)
        except ValueError:failed=True
        ck(f'reject_update_{s}',failed)
    ck('route32_peak',route_weight(32)==.03)
    ck('route192_half',abs(route_weight(192)-.015)<1e-12)
    ck('route320_zero',route_weight(320)==0 and route_weight(1000)==0)
    ck('exact_exposure_count',1000*4==4000)
    gen=torch.Generator().manual_seed(42)
    order=torch.cat([torch.randperm(3857,generator=gen),torch.randperm(3857,generator=gen)])[:4000]
    ck('first_epoch_covers_all',len(set(order[:3857].tolist()))==3857)
    ck('correct_remainder',order[3857:].numel()==143)

    base=nn.Linear(8,8)
    model=ToyLoRA(base)
    x=torch.randn(4,8);target=torch.randn(4,8)
    ck('lora_zero_initial_equivalence',torch.equal(model(x),base(x)))
    y=model(x);loss=(y-target).square().mean();loss.backward()
    ck('lora_B_first_grad',model.b.grad.abs().sum()>0)
    ck('lora_A_initial_grad_zero',model.a.grad.abs().sum()==0)
    before=base.weight.detach().clone()
    opt=torch.optim.SGD([model.a,model.b],lr=.1)
    opt.step();opt.zero_grad();(model(x)-target).square().mean().backward()
    ck('lora_A_second_grad',model.a.grad.abs().sum()>0)
    opt.step()
    ck('base_remains_frozen',torch.equal(base.weight,before) and not base.weight.requires_grad)
    second=ToyLoRA(base)
    ck('frozen_base_can_be_shared',second.base.weight is model.base.weight)
    ck('adapter_not_shared',second.a is not model.a and second.b is not model.b)
    upstream=torch.randn(2,8,requires_grad=True);base(upstream).sum().backward()
    ck('frozen_base_preserves_input_gradient',upstream.grad is not None and upstream.grad.abs().sum()>0)

    m1=nn.Linear(8,3);m2=copy.deepcopy(m1)
    xx=torch.randn(4,8);yy=torch.randn(4,3)
    ((m1(xx)-yy).square().mean()).backward()
    for i in range(4):((m2(xx[i:i+1])-yy[i:i+1]).square().mean()/4).backward()
    ck('batch4_gradient_accumulation',torch.allclose(m1.weight.grad,m2.weight.grad,atol=1e-7,rtol=1e-6))

    inner=nn.Linear(8,8,bias=False);write=nn.Linear(8,8,bias=False)
    nn.init.zeros_(write.weight);backbone=nn.Linear(8,8).requires_grad_(False)
    ck('zero_writer_official_start',torch.equal(backbone(x)+write(inner(x)),backbone(x)))
    optim=torch.optim.SGD(list(inner.parameters())+list(write.parameters()),lr=.05)
    grad_ok=False
    for step in range(3):
        optim.zero_grad();((backbone(x)+write(inner(x))-target).square().mean()).backward()
        if step>=1 and inner.weight.grad.abs().sum()>0:grad_ok=True
        optim.step()
    ck('inner_gradient_after_writer_opens',grad_ok)

    w={k:torch.randn(8,8)/math.sqrt(8) for k in ['q','k','v','o','uq','uk','g','write']}
    w['f1']=torch.randn(16,8)/math.sqrt(8);w['f2']=torch.randn(8,16)/4
    h=torch.randn(1,5,8);s=torch.randn_like(h);p=torch.randn_like(h)
    ck('D_zero_physics_zero_output',d_executor(torch.zeros_like(p),h,s,w).abs().max()==0)
    ck('D_semantic_condition_can_change_result',not torch.allclose(d_executor(p,h,s,w),d_executor(p,h,-s,w)))
    p=p.requires_grad_();s=s.requires_grad_();d_executor(p,h,s,w).square().mean().backward()
    ck('D_both_inputs_differentiable',p.grad.abs().sum()>0 and s.grad.abs().sum()>0)

    phys=torch.randn(1,3,8);text1=torch.randn(1,4,8);text2=torch.randn(1,4,8)
    c1=phys+F.scaled_dot_product_attention(phys[:,None],text1[:,None],text1[:,None]).squeeze(1)
    c2=phys+F.scaled_dot_product_attention(phys[:,None],text2[:,None],text2[:,None]).squeeze(1)
    ck('C_semantics_changes_execute_not_only_key',not torch.allclose(c1,c2))
    eye=torch.eye(8);rawroute=torch.randn_like(phys)
    route=rawroute+F.linear(c1-phys,eye)
    ck('C_route_identity_maps_content_update',torch.allclose(route-rawroute,c1-phys,atol=1e-6))

    clean=torch.randn(1,2,5,4,4);noise=torch.randn_like(clean);pred=noise-clean
    valid=torch.ones(1,4,4)
    ck('FM_target_sign_perfect_zero',gen_loss(pred,noise,clean,valid)[0]==0)
    known=pred.clone();known[:,:,0]+=100
    ck('known_first_excluded',gen_loss(known,noise,clean,valid)[0]==0)
    valid[:,:,0]=0;masked=pred.clone();masked[:,:,:, :,0]+=100
    ck('invalid_padding_excluded',gen_loss(masked,noise,clean,valid)[0]==0)
    uniform=pred.clone();uniform[:,:,1:]+=2
    l,m,t=gen_loss(uniform,noise,clean,torch.ones_like(valid))
    ck('tail_uniform_error_consistency',torch.allclose(l,torch.tensor(4.),atol=1e-5) and torch.allclose(m,t,atol=1e-5))
    sigma=torch.tensor(.4);zsig=(1-sigma)*clean+sigma*noise
    ck('clean_reconstruction_identity',torch.allclose(zsig-sigma*(noise-clean),clean,atol=1e-6))
    d=3072;ff=14336
    ck('native_lora_parameter_budget',30*(10*d+2*ff)*32==57016320)
    ck('B_lora_parameter_budget',(20*32+10*64)*(10*d+2*ff)==76021760)
    report={'kind':'cpu_small_tensor_contract_checks_only','checks':len(results),'passed':sum(x['passed'] for x in results),'details':results,'not_tested':['actual_Wan_forward','real_GPU_training','actual_video_generation','PC_SA_Joint_or_cross_benchmark_scores']}
    (ROOT/'TOY_CHECK_RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ['kind','checks','passed']},ensure_ascii=False))

if __name__=='__main__':main()
