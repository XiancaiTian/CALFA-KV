"""Joint mu/A/K/V smooth directions; actual hard/native acceptance is separate."""
import torch


def tree_rms(x):
    n=x.shape[-2]*x.shape[-1];assert n and not n&(n-1)
    y=x.square().reshape(*x.shape[:-2],n)
    for _ in range(n.bit_length()-1):y=y[...,0::2]+y[...,1::2]
    return (y[...,0]/n).sqrt().clamp_min(1e-4)


def radial_from_coordinates(original,u,unit):
    pair=u.reshape(original.shape[0],original.shape[1]//2);pair=pair-pair.mean()
    return original*torch.cat([pair,pair],-1).mul(unit).exp()


def center_from_coordinates(u,unit):return u.reshape_as(unit)*unit


def rotate(x,cos,sin):
    half=x.shape[-1]//2
    return torch.cat([x[...,:half]*cos-x[...,half:]*sin,x[...,half:]*cos+x[...,:half]*sin],-1)


def dot(q,k):
    half=q.shape[-1]//2;k=k.repeat_interleave(q.shape[1]//k.shape[1],1).permute(1,0,2)
    return torch.einsum('qhd,htd->qht',q[...,:half],k[...,:half])+torch.einsum('qhd,htd->qht',q[...,half:],k[...,half:])


def residual(books,center,radial,raw,value,labels_k,labels_v,weight,permutation,reference,scale16,scale_anchor32,fit_root):
    t,h,d=raw.shape;groups,width=books.shape[2],books.shape[-1]
    hi=torch.arange(h,device=books.device)[None,:,None];gi=torch.arange(groups,device=books.device)[None,None,:]
    words=books[0][hi,gi,labels_k].reshape(t,h,d)
    physical=torch.zeros_like(words).scatter(-1,permutation.expand_as(words),words/weight.gather(-1,permutation))
    scale=scale16.to(books.dtype)+(tree_rms((raw-center)*radial)-scale_anchor32)
    quant=dot(reference['q'],rotate(physical,reference['cos'],reference['sin']))*scale[None,None,:]
    quant=quant+dot(reference['q'],rotate(center[None].expand_as(raw),reference['cos'],reference['sin']))
    history=reference['history'][:,None,:];score=torch.where(history,quant,reference['hp_scores'])
    probability=(score/(d**.5)).masked_fill(~reference['causal'][:,None,:],-torch.inf).softmax(-1)
    ratio=reference['q'].shape[1]//h
    v=value.repeat_interleave(ratio,1).permute(1,0,2)
    vq=books[1][hi,gi,labels_v].reshape(t,h,d).repeat_interleave(ratio,1).permute(1,0,2)
    context=(torch.einsum('qht,htd->qhd',probability*(~history),v)+torch.einsum('qht,htd->qhd',probability*history,vq)).reshape(probability.shape[0],-1)
    return (context@reference['projection'].T-reference['teacher']).reshape(-1)/fit_root


def self_checks():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(26102253)
    def rand(shape):return torch.randn(shape,generator=g,dtype=torch.float64)
    t,h,d,qheads=11,2,8,4;positions=torch.tensor([2,4,6,8,10]);counts=torch.tensor([0,1,3,5,7])
    raw,value=rand((t,h,d)),rand((t,h,d));weight=rand((h,d)).mul(.1).exp();radial=weight.clone();center=torch.zeros_like(weight)
    permutation=torch.tensor([[0,4,1,5,2,6,3,7],[3,7,2,6,1,5,0,4]])
    q=rand((5,qheads,d));angle=rand((t,1,d//2));cos,sin=angle.cos(),angle.sin()
    history=torch.arange(t)[None,:]<counts[:,None];causal=torch.arange(t)[None,:]<=positions[:,None]
    projection=rand((7,qheads*d));hp=dot(q,rotate(raw,cos,sin))
    probability=(hp/(d**.5)).masked_fill(~causal[:,None],-torch.inf).softmax(-1)
    teacher=torch.einsum('qht,htd->qhd',probability,value.repeat_interleave(2,1).permute(1,0,2)).reshape(5,-1)@projection.T
    ref=dict(q=q,cos=cos,sin=sin,history=history,causal=causal,hp_scores=hp,projection=projection,teacher=teacher)
    books=rand((2,h,2,16,4));ik=torch.randint(16,(t,h,2),generator=g);iv=torch.randint(16,(t,h,2),generator=g)
    unit=tree_rms(raw).mean(0).expand_as(center);log_unit=.03;n32=tree_rms((raw-center)*radial);n16=n32.half().double()
    assert torch.equal(n16+(tree_rms((raw-center)*radial)-n32),n16)
    point=torch.cat([books.reshape(-1),torch.zeros(h*d//2+h*d,dtype=torch.float64)])
    def function(p):
        cb=p[:books.numel()].reshape_as(books);u=p[books.numel():];a=radial_from_coordinates(radial,u[:h*d//2],log_unit);mu=center_from_coordinates(u[h*d//2:],unit)
        return residual(cb,mu,a,raw,value,ik,iv,weight,permutation,ref,n16,n32,1.)
    errors={}
    for name,sl in [('K',slice(0,books[0].numel())),('V',slice(books[0].numel(),books.numel())),('A',slice(books.numel(),books.numel()+h*d//2)),('mu',slice(books.numel()+h*d//2,None))]:
        direction=torch.zeros_like(point);direction[sl]=rand(point[sl].shape);direction/=direction.norm()
        _,jvp=torch.func.jvp(function,(point,),(direction,));eps=1e-5
        fd=(function(point+eps*direction)-function(point-eps*direction))/(2*eps)
        relative=float((fd-jvp).norm()/jvp.norm().clamp_min(1e-30));assert relative<1e-7,(name,relative);errors[name]=relative
        _,pull=torch.func.vjp(function,point);cot=rand(function(point).shape);vjp,=pull(cot)
        torch.testing.assert_close((jvp*cot).sum(),(vjp*direction).sum(),rtol=1e-10,atol=1e-10)
    # Absolute physical center and independent radial coordinates have nonzero
    # joint attention derivatives; D is deliberately fixed, with no fake metric gradient.
    shifted=point.clone();shifted[books.numel():books.numel()+h*d//2]+=5
    torch.testing.assert_close(function(point),function(shifted),rtol=1e-14,atol=1e-14)
    return dict(passed=True,all_K_V_A_center_JVP_FD_and_adjoint_checked=True,relative_FD_errors=errors,
        scale_smooth_value_exact_actual_FP16_anchor=True,radial_global_log_gauge_removed=True,
        same_joint_attention_objective_and_actual_HP_causal_masks=True,
        no_derivative_claim_through_FP16_rounding_or_hard_labels=True,no_native_candidate_or_accuracy_claim=True)
