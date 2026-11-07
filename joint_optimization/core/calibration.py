"""Joint center/radial/K/V calibration and exact native replay."""
import datetime,json,math,time
from pathlib import Path
import torch
from artifact_io import sha,dump
from problem import Problem
from policy import FIT,VALIDATION,accepts
from direction import propose_direction




def read(path):return json.loads(Path(path).read_text())




def compare(state,saved,labels=True):
    assert state['metrics']==saved['metrics'] and state['sse']==saved['sse']
    for name in ['center','radial','N','outputs','aux']:assert torch.equal(state[name].cpu(),saved[name]),name
    assert torch.equal(state['books'].half().cpu(),saved['books'])
    if labels:
        for name in ['labels_k','labels_v']:assert torch.equal(state[name].cpu(),saved[name]),name


def project_aux(point,n):
    result=point.clone();a=result[n:n+512];a-=a.mean();return result


def train(c,config):
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    def status(phase,**kw):dump(out/'status.json',dict(phase=phase,layer=c['layer'],optimization=c['optimization'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    status('loading_same_teacher_native_problem');problem=Problem(c)
    initial=problem.pack(problem.parent,problem.zero);parent=problem.actual(initial)
    assert torch.equal(parent['books'],problem.parent) and not parent['center'].any() and torch.equal(parent['radial'],problem.weight)
    torch.save(problem.snapshot(parent),out/'parent.pt')
    repeated=problem.actual(initial);compare(repeated,torch.load(out/'parent.pt',map_location='cpu',weights_only=True))
    state=parent;history=[]
    for iteration in range(8):
        status('joint_output_direction',iteration=iteration)
        point=problem.pack(state['books'],state['aux']);labels=(state['labels_k'],state['labels_v']);functions=problem.functions(labels,point)
        if iteration==0:
            generator=torch.Generator(device='cuda').manual_seed(2610225300+problem.layer)
            direction=torch.zeros_like(point);direction[-1536:]=torch.randn((1536,),generator=generator,device='cuda')
            direction[problem.book_count:problem.book_count+512]-=direction[problem.book_count:problem.book_count+512].mean()
            cotangent=torch.randn((16*2560,),generator=generator,device='cuda')
            _,jvp=torch.func.jvp(functions[0],(point,),(direction,));_,pullback=torch.func.vjp(functions[0],point);vjp,=pullback(cotangent)
            left=float((jvp.double()*cotangent.double()).sum());right=float((vjp.double()*direction.double()).sum())
            assert abs(left-right)<=2e-4*max(abs(left),abs(right),1e-8)
            eps=.01;fd=(functions[0](point+eps*direction)-functions[0](point-eps*direction))/(2*eps)
            relative=float((fd.double()-jvp.double()).norm()/jvp.double().norm().clamp_min(1e-30));assert relative<.02,relative
            check=dict(passed=True,actual_FIT0_center_radial_JVP_FD_relative=relative,adjoint_relative=abs(left-right)/max(abs(left),abs(right),1e-8),no_discrete_derivative_claim=True)
            del pullback,jvp,vjp,fd
            dump(out/'smooth_coordinate_checks.json',check)
        direction,gradient,step,diagnostic=propose_direction(functions,point,'gauss_newton',24,1e-4)
        dfile=out/('direction_%02d.pt'%iteration);torch.save(dict(point=point.cpu(),direction=direction.cpu(),gradient=gradient.cpu(),step=step.cpu(),diagnostic=diagnostic),dfile)
        if diagnostic['zero_gradient']:history.append(dict(iteration=iteration,stop='zero_gradient'));break
        trials=[];accepted=False
        for backtrack in range(12):
            alpha=.02*.5**backtrack;candidate=project_aux(point+alpha*direction,problem.book_count)
            status('actual_native_writer_kernel_and_BF16_WO_acceptance',iteration=iteration,backtrack=backtrack)
            proposed=problem.actual(candidate);passed,reasons=accepts(state['metrics'],proposed['metrics'],parent['metrics'])
            file=out/('proposal_%02d_%02d.pt'%(iteration,backtrack));saved=problem.snapshot(proposed,full=False);saved.update(alpha=alpha,accepted=passed,reasons=reasons)
            torch.save(saved,file);trials.append(dict(backtrack=backtrack,alpha=alpha,accepted=passed,reasons=reasons,file=file.name,sha256=sha(file),metrics=proposed['metrics']))
            if passed:state=proposed;accepted=True;break
        history.append(dict(iteration=iteration,accepted=accepted,trials=trials,direction_file=dfile.name,direction_sha256=sha(dfile),direction=diagnostic));dump(out/'history.json',history)
        if not accepted:break
    status('selected_actual_native_recheck');selected=problem.actual(problem.pack(state['books'],state['aux']))
    compare(selected,problem.snapshot(state));file=out/'selected.pt';torch.save(problem.snapshot(selected),file)
    dump(out/'complete.json',dict(passed=True,config_sha256=sha(config),calfa_source_sha256=c.get('calfa_source_sha256'),layer=problem.layer,optimization=problem.optimization,
        parent_metrics=parent['metrics'],selected_metrics=selected['metrics'],history=history,accepted_steps=sum(bool(h.get('accepted')) for h in history),
        parent_sha256=sha(out/'parent.pt'),selected_sha256=sha(file),actual_writer_frozen_N_and_Vunit_all64_each_proposal=True,
        actual_native_attention_dynamic_splits_and_BF16_WO_all64_each_proposal=True,original_acceptance_guards_unchanged=True,
        D_P_fixed=True,one_K_FP16_scale_unique_FP16_C_Vunit_same_bits=True,no_full_model_KL_or_task_accuracy_admission=True,
        duration_seconds=time.monotonic()-began,peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    status('training_complete');torch.distributed.destroy_process_group()


def verify(c,config):
    out=Path(c['output']);done=read(out/'complete.json');assert done['passed'] and done['config_sha256']==sha(config)
    problem=Problem(c);parent=torch.load(out/'parent.pt',map_location='cpu',weights_only=True)
    state=problem.actual(problem.pack(problem.parent,problem.zero));compare(state,parent)
    current=parent['metrics'];count=0
    for h in done['history']:
        if h.get('stop'):continue
        dfile=out/h['direction_file'];assert sha(dfile)==h['direction_sha256'];direction=torch.load(dfile,map_location='cpu',weights_only=True)
        point=direction['point'].cuda();d=direction['direction'].cuda()
        b,u,_,_=problem.unpack(point);assert torch.equal(b.half(),state['books'].half()) and torch.equal(u,state['aux'])
        for trial in h['trials']:
            file=out/trial['file'];assert sha(file)==trial['sha256'];saved=torch.load(file,map_location='cpu',weights_only=True)
            candidate=project_aux(point+trial['alpha']*d,problem.book_count);result=problem.actual(candidate);compare(result,saved,labels=False)
            passed,reasons=accepts(current,result['metrics'],parent['metrics']);assert passed==saved['accepted']==trial['accepted'] and reasons==saved['reasons']==trial['reasons']
            count+=1
            if passed:state=result;current=result['metrics']
    selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=True);result=problem.actual(problem.pack(state['books'],state['aux']));compare(result,selected)
    assert result['metrics']==done['selected_metrics']==current and sha(out/'selected.pt')==done['selected_sha256']
    dump(out/'independent_GPU_verification.json',dict(passed=True,layer=problem.layer,optimization=problem.optimization,trial_native_output_replays=count,
        all_direction_candidate_bytes_and_original_acceptance_replayed=True,parent_and_selected_outputs_labels_N_center_A_bitwise=True,
        D_P_and_all_frozen_teacher_inputs_unchanged=True,actual_native_kernel_and_BF16_projection_used=True,selected_metrics=current,
        selected_sha256=done['selected_sha256'],no_full_model_KL_or_task_accuracy_admission=True))
    torch.distributed.destroy_process_group()
