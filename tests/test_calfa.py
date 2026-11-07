"""CPU contracts and focused CUDA tests of the integrated native decode kernel."""
import importlib.util
from pathlib import Path
import sys
import types
import json
import hashlib
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/python'),str(ROOT/'fakv')]
from calfa_anchor_cache import AnchorCache
from calfa_anchor_coordinates import (coordinate_identity, signed_gradient_to_z,
    regularize_joint_fisher, restore_quantized_center)
from calfa_affine import encode_with_stored_scale, freeze_scale


def permutation(device='cpu'):
    pairs = torch.arange(64,device=device).roll(7)
    return torch.stack((pairs,pairs+64),-1).reshape(1,128).expand(8,-1).contiguous()


def pack8(indices):
    x = indices.long().reshape(*indices.shape[:-1],4,4)
    return (x << torch.arange(4,device=x.device).mul(8)).sum(-1).to(torch.int32).contiguous()


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class CPUContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1537)

    def test_signed_gradient_fixed_scale_chain_rule(self):
        raw=torch.randn(5,8,128,requires_grad=True)
        D=torch.rand(8,128)+.5;mu=torch.randn(8,128)*.1;P=permutation()
        scale=freeze_scale(raw.detach(),mu,D*1.1)
        z=encode_with_stored_scale(raw,D,P,scale,mu)
        gz=torch.randn_like(z)
        gx,=torch.autograd.grad((z*gz).sum(),raw)
        mapped=signed_gradient_to_z(gx,D,P,scale)
        torch.testing.assert_close(mapped,gz,atol=5e-7,rtol=5e-7)

    def test_identity_binds_center_and_radial(self):
        D=torch.ones(36,8,128);P=permutation()[None].expand(36,-1,-1).clone();mu=torch.zeros_like(D)
        ref=coordinate_identity(D,P,mu,D)
        changed=mu.clone();changed[0,0,0]=.1
        self.assertNotEqual(ref,coordinate_identity(D,P,changed,D))
        changed=D.clone();changed[0,0,0]=1.1
        self.assertNotEqual(ref,coordinate_identity(D,P,mu,changed))

    def test_prefix_center_only_on_quantized_rows(self):
        raw=torch.zeros(3,8,128,dtype=torch.bfloat16)
        out=restore_quantized_center(raw,torch.tensor([0,1,10]),10,torch.ones(8,128))
        self.assertTrue(out[:2].eq(1).all());self.assertTrue(out[2].eq(0).all())

    def test_anchor_top56_ties_release_and_request_isolation(self):
        cache=AnchorCache(1,2,'cpu')
        raw=torch.randn(1,80,8,128).bfloat16();pos=torch.arange(80)[None]
        score=pos[:,:,None].expand(-1,-1,8).float().contiguous()
        cache.admit(0,torch.tensor([0]),pos,pos,torch.ones_like(pos,dtype=torch.bool),raw,raw,score)
        self.assertTrue(cache.counts[0,0].eq(56).all());self.assertTrue(cache.counts[0,1].eq(0).all())
        self.assertEqual(set(cache.positions[0,0,0].tolist()),set(range(24,80)))
        cache.reset(torch.tensor([0]));self.assertTrue(cache.positions.eq(-1).all())
        score.zero_()
        cache.admit(0,torch.tensor([0]),pos,pos,torch.ones_like(pos,dtype=torch.bool),raw,raw,score)
        self.assertEqual(cache.positions[0,0,0].tolist(),list(range(56)))
        hit,k,v=cache.lookup(0,torch.tensor([1]),pos,pos)
        self.assertFalse(hit.any());self.assertFalse(k.any());self.assertFalse(v.any())

    def test_fisher_joint_regularization(self):
        k,v=regularize_joint_fisher(torch.rand(36,8,128),torch.rand(36,8,128))
        self.assertTrue((k>0).all() and (v>0).all())
        torch.testing.assert_close(.5*(k.mean(-1)+v.mean(-1)),torch.ones(36,8))


class PipelineContracts(unittest.TestCase):
    def test_anchor_plan_uses_one_bundle_without_execution(self):
        module=load_file('calfa_pipeline',ROOT/'pipeline/run.py')
        with tempfile.TemporaryDirectory(prefix='calfa_plan_') as tmp:
            root=Path(tmp);file=root/'pipeline.json'
            file.write_text(json.dumps(dict(method='CALFA-KV',model='model',bundle='bundle',
                anchor=dict(input_ids='tokens.json',output='anchor'),serve_args=['--port','30001'])))
            plan=module.commands(file,'anchor','serve')
            self.assertEqual([r['stage'] for r in plan],['anchor','anchor','anchor','serve'])
            for row in plan:
                self.assertIn(str(root/'bundle'),row['argv'])
            self.assertIn(str(root/'anchor/manifest.json'),plan[-1]['argv'])
            self.assertEqual(list(root.iterdir()),[file])

    def test_joint_config_resolves_current_modules_and_capture_fields(self):
        module=load_file('calfa_optimize',ROOT/'joint_optimization/optimize.py')
        with tempfile.TemporaryDirectory(prefix='calfa_config_') as tmp:
            root=Path(tmp)
            # Static identity validation only; these bytes are not model activations.
            (root/'capture.bin').write_bytes(b'static-input-identity')
            (root/'projection.bin').write_bytes(b'static-projection-identity')
            (root/'tokens.json').write_text(json.dumps([dict(input_ids=[1]*2056) for _ in range(64)]))
            def sha(name):return hashlib.sha256((root/name).read_bytes()).hexdigest()
            rows={str(w):dict(file='capture.bin',sha256=sha('capture.bin')) for w in range(64)}
            config=dict(layer=0,model='model',assets='initial',output='runs/layer_00',
                input_ids='tokens.json',input_ids_sha256=sha('tokens.json'),
                teacher_layers=rows,cache_metadata=rows,
                projection=dict(file='projection.bin',key='model.layers.0.self_attn.o_proj.weight',sha256=sha('projection.bin')))
            file=root/'layer.json';file.write_text(json.dumps(config))
            prepared=module.prepare(file)
            self.assertIn('cache_metadata',prepared)
            self.assertEqual(prepared['optimization'],'joint_calibration')
            self.assertTrue(all((ROOT/name).is_file() for name in prepared['calfa_source_sha256']))
            config['cache_metadata']['3']['sha256']='0'*64;file.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'hash differs'):
                module.prepare(file)

    def test_joint_plan_binds_all_layer_outputs_and_inputs(self):
        module=load_file('calfa_pipeline',ROOT/'pipeline/run.py')
        with tempfile.TemporaryDirectory(prefix='calfa_joint_plan_') as tmp:
            root=Path(tmp);files=[]
            for layer in range(36):
                file=root/('layer_%02d.json'%layer);files.append(file.name)
                file.write_text(json.dumps(dict(layer=layer,model='model',assets='initial',
                    input_ids='tokens.json',output='runs/layer_%02d'%layer)))
            config=dict(method='CALFA-KV',model='model',bundle='bundle',
                anchor=dict(input_ids='tokens.json',output='anchor'),
                joint=dict(configs=files,runs='runs',initial_assets='initial'))
            file=root/'pipeline.json';file.write_text(json.dumps(config))
            plan=module.commands(file,'joint','anchor')
            self.assertEqual(len(plan),40)
            self.assertTrue(all(row['argv'][-2:]==['--stage','all'] for row in plan[:36]))
            self.assertIn('--initial-assets',plan[36]['argv'])
            wrong=json.loads((root/files[9]).read_text());wrong['output']='runs/layer_08'
            (root/files[9]).write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'output'):
                module.commands(file,'joint','anchor')
            config['joint']['configs'][9]=files[8];file.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'distinct'):
                module.commands(file,'joint','anchor')


class ExportIntegration(unittest.TestCase):
    def test_all36_export_round_trip(self):
        # Synthetic tensors exercise export wiring, never calibration quality.
        with tempfile.TemporaryDirectory(prefix='calfa_export_test_') as tmp:
            root=Path(tmp);parent=root/'parent/bundle';parent.mkdir(parents=True)
            D=torch.ones(36,8,128);P=permutation()[None].expand(36,-1,-1).clone()
            zero=torch.zeros(256,8,dtype=torch.float16)
            blob=dict(codebooks={(l,h):[zero]*16 for l in range(36) for h in range(8)},
                channel_weights=D,channel_permutations=P,scale_contract='calfa_fp16_scale_v1')
            torch.save(blob,parent/'codebook.pt');torch.save(blob,parent/'codebook_v.pt')
            for name in ['k_rotation_qqt_r_h_pbr.pt','v_rotation_sst_r_h_pbr.pt']:
                torch.save(dict(layers={l:dict(rotation=torch.eye(128)) for l in range(36)}),parent/name)
            for l in range(36):
                out=root/'runs'/('layer_%02d'%l);out.mkdir(parents=True)
                book=torch.zeros(2,8,16,256,8,dtype=torch.float16)
                book[0].fill_(l/100);book[1].fill_(-l/100)
                selected=out/'selected.pt'
                torch.save(dict(books=book,weight=D[l],permutation=P[l],
                    center=torch.full((8,128),l/1000),radial=torch.full((8,128),1+l/1000)),selected)
                sha=hashlib.sha256(selected.read_bytes()).hexdigest()
                proof=dict(passed=True,layer=l,optimization='joint_calibration',selected_sha256=sha)
                (out/'complete.json').write_text(json.dumps(proof))
                (out/'independent_GPU_verification.json').write_text(json.dumps(proof))
            command=[sys.executable,str(ROOT/'joint_optimization/export.py'),'--initial-assets',str(root/'parent'),
                     '--runs',str(root/'runs'),'--output',str(root/'export')]
            subprocess.run(command,check=True,capture_output=True,text=True)
            k=torch.load(root/'export/codebook.pt',map_location='cpu',weights_only=False)
            v=torch.load(root/'export/codebook_v.pt',map_location='cpu',weights_only=False)
            self.assertEqual(k['scale_contract'],'calfa_center_radial_v1')
            self.assertTrue(torch.equal(k['channel_weights'],D) and torch.equal(k['channel_permutations'],P))
            for l in range(36):
                torch.testing.assert_close(k['affine_center'][l],torch.full((8,128),l/1000))
                self.assertTrue(k['codebooks'][l,0][0].eq(torch.tensor(l/100,dtype=torch.float16)).all())
                self.assertTrue(v['codebooks'][l,0][0].eq(torch.tensor(-l/100,dtype=torch.float16)).all())
            # A proof from another layer must not be accepted.
            bad=root/'runs/layer_35/independent_GPU_verification.json'
            proof=json.loads(bad.read_text());proof['layer']=34;bad.write_text(json.dumps(proof))
            command[-1]=str(root/'reject')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'reject').exists())


@unittest.skipUnless(torch.cuda.is_available(),'CUDA device unavailable')
class CUDAIntegration(unittest.TestCase):
    def test_anchor_cache_cuda_graph_replay(self):
        cache=AnchorCache(1,1,'cuda:0')
        req=torch.tensor([0],device='cuda:0');pos=torch.arange(8,device='cuda:0')[None]
        valid=torch.ones_like(pos,dtype=torch.bool)
        raw=torch.randn(1,8,8,128,device='cuda:0').bfloat16()
        score=torch.ones(1,8,8,device='cuda:0')
        for _ in range(2):
            cache.reset(req);cache.admit(0,req,pos,pos,valid,raw,raw,score)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cache.reset(req);cache.admit(0,req,pos,pos,valid,raw,raw,score)
        graph.replay();torch.cuda.synchronize()
        self.assertTrue(cache.counts.eq(8).all())
        self.assertEqual(cache.positions[0,0,0,:8].cpu().tolist(),list(range(8)))

    def test_affine_reconstruction_score_and_anchor_native_decode(self):
        from calfa_anchor_codec import reconstruct_candidates,score_actual_reconstruction
        torch.manual_seed(4421);torch.backends.cuda.matmul.allow_tf32=False
        device='cuda:0';T=64;P=permutation(device);freq=P[:,::2].contiguous()
        D=torch.rand(8,128,device=device)+.5
        center=torch.randn(8,128,device=device)*.08
        cb=torch.randn(8,16,256,8,device=device).half()*.1
        vb=torch.randn_like(cb)*.5
        ik=torch.randint(256,(T,8,16),device=device);iv=torch.randint(256,(T,8,16),device=device)
        ki,vi=pack8(ik),pack8(iv)
        scale=(torch.rand(T,1,1,device=device)+.3).half().expand(T,8,1)
        ones=torch.ones(1,1,1,dtype=torch.float16,device=device).expand(T,8,1)
        pos=torch.arange(T,device=device);valid=torch.ones(T,dtype=torch.bool,device=device)
        inv=1/(10000**(torch.arange(64,device=device).float()/64))
        angle=torch.arange(T+2,device=device)[:,None]*inv
        cs=torch.cat((angle.cos(),angle.sin()),-1).contiguous()
        inv=torch.atan2(cs[1,64:],cs[1,:64]).contiguous()
        k,v=reconstruct_candidates(pos,pos,valid,ki,vi,cb,vb,scale,freq,D.gather(-1,P).contiguous(),inv,center=center)
        # Load the real native kernel source. Only unrelated SGLang imports are
        # stubbed; no arithmetic, launch, packing or Anchor implementation is replaced.
        utils=types.ModuleType('sglang.srt.utils');utils.is_hip=lambda:False
        env=types.ModuleType('sglang.srt.environ');env.envs=types.SimpleNamespace()
        with patch.dict(sys.modules,{'sglang.srt.utils':utils,'sglang.srt.environ':env}):
            native=load_file('calfa_native_decode_test',ROOT/'backend/python/sglang/srt/layers/attention/triton_ops/decode_attention.py')
        q=torch.randn(1,32,128,device=device).bfloat16()*.1
        out=torch.empty(1,32,1,128,device=device);lse=torch.empty(1,32,1,device=device)
        indptr=torch.tensor([0,T],dtype=torch.int32,device=device);splits=torch.ones(1,dtype=torch.int32,device=device)
        tk=torch.empty_like(k);tv=torch.empty_like(v);logits=torch.empty(1,32,T,device=device)
        def run(mu,**kwargs):
            native._decode_grouped_att_m_fwd_quant_vqwide(q,ki,cb,vi,vb,scale,ones,out,lse,
                indptr,pos,splits,1,128**-.5,0.,cos_sin_cache=cs,freq_idx=freq,
                w_perm=D.gather(-1,P).contiguous(),center_perm=mu.gather(-1,P).contiguous(),
                trace_k=tk,trace_v=tv,trace_logits=logits,**kwargs)
            torch.cuda.synchronize()
            return tk.clone(),tv.clone(),logits.clone()
        nk,nv,baseline=run(center)
        torch.testing.assert_close(nk,k,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(nv,v,atol=0,rtol=0)
        raw=torch.randn(T,8,128,device=device).bfloat16();value=torch.randn_like(raw)
        score=score_actual_reconstruction(raw,value,k,v,pos,valid,scale[:,0,0].contiguous(),D,P,
                    torch.ones_like(D),torch.ones_like(D),cs)
        a,b=k[...,:64],k[...,64:];c=cs[:T,:64,None].transpose(1,2);s=cs[:T,64:,None].transpose(1,2)
        khat=torch.cat(((a*c+b*s)/(c*c+s*s),(b*c-a*s)/(c*c+s*s)),-1)
        dk=((raw.float()-khat)*D/scale[:,0,0].float()[:,None,None]).gather(-1,P.expand(T,-1,-1))
        expected=(dk.square()+(value.float()-v).square()).sum(-1)
        torch.testing.assert_close(score,expected,atol=3e-4,rtol=3e-6)
        ak=torch.zeros(1,8,56,128,dtype=torch.float16,device=device);av=torch.zeros_like(ak)
        ap=torch.full((1,8,56),-1,dtype=torch.int64,device=device);slots=ap.clone()
        for lane,p in enumerate([3,17]):
            ak[0,:,lane]=raw[p].half();av[0,:,lane]=value[p].half();ap[0,:,lane]=p;slots[0,:,lane]=p
        kw=dict(anchor_k=ak,anchor_v=av,anchor_positions=ap,anchor_slots=slots,request_indices=torch.tensor([0],device=device))
        anchored_k,anchored_v,actual=run(center,**kw)
        _,_,changed_center=run(center*37,**kw)
        torch.testing.assert_close(actual[:,:, [3,17]],changed_center[:,:, [3,17]],atol=0,rtol=0)
        for p in [3,17]:
            a,b=raw[p,:,:64].half().float(),raw[p,:,64:].half().float()
            r1=(a*cs[p,:64]-b*cs[p,64:]).half().float()
            r2=(b*cs[p,:64]+a*cs[p,64:]).half().float()
            wanted=(q[0,:,:64].half().float()*r1.repeat_interleave(4,0)).sum(-1)
            wanted+=(q[0,:,64:].half().float()*r2.repeat_interleave(4,0)).sum(-1)
            torch.testing.assert_close(actual[0,:,p],wanted*128**-.5,atol=1e-6,rtol=2e-5)
            torch.testing.assert_close(anchored_v[p],value[p].half().float(),atol=0,rtol=0)
        mask=torch.ones(T,dtype=torch.bool,device=device);mask[[3,17]]=False
        torch.testing.assert_close(actual[:,:,mask],baseline[:,:,mask],atol=1e-6,rtol=2e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
