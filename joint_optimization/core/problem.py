"""Same-BF16-state joint mu/A/K/V fitting with exact native output acceptance."""
import json,math,os
from pathlib import Path
from types import SimpleNamespace
import torch
from native_output import NativeJointOutput
from objective import tree_rms,radial_from_coordinates,center_from_coordinates,residual,rotate,dot
from direction import group_coordinate_scale
from policy import FIT,VALIDATION,SPLITS


class Problem:
    def __init__(self,c):
        from safetensors import safe_open
        from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
        from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool
        from initial_assets import verify_parent_pool
        from transformers import AutoConfig
        from sglang.srt.models.qwen3_config_compat import resolve_qwen3_rope_theta
        self.c=c;self.layer=c['layer'];self.optimization=c['optimization'];torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261020)
        init_distributed_environment(world_size=1,rank=0,local_rank=c.get('local_rank',0),distributed_init_method='tcp://[::1]:%d'%c['port'],backend='nccl');initialize_model_parallel(tensor_model_parallel_size=1)
        self.pool=UnifiedInt2HPKVPool(num_quant_pages=257,hp_dtype=torch.bfloat16,hp_prefix_tokens=0,hp_recent_tokens=64,dtype='int2',head_num=8,head_dim=128,layer_num=36,device='cuda',enable_memory_saver=False,max_req_slots=2,model_dtype=torch.bfloat16,kv_cache_quant_group_size=128,scale_dtype=torch.float16)
        assert verify_parent_pool(self.pool,c['assets'])['passed']
        codec=self.pool._vq;self.weight=codec.channel_weights[self.layer].clone();self.permutation=codec.channel_permutations[self.layer].clone()
        self.parent=torch.stack([codec.cb16[self.layer],self.pool._vq_v.cb16[self.layer]]).float().clone();self.units=group_coordinate_scale(self.parent)
        self.book_count=self.parent.numel();self.zero=torch.zeros(1536,device='cuda');self.original_radial=self.weight.clone()
        self.log_unit=math.sqrt(512)/float((self.parent/self.units).double().norm())
        # Private, serializable learning state; the audited original asset files stay immutable.
        codec.affine_center=torch.zeros_like(codec.channel_weights);codec.affine_radial=codec.channel_weights.clone();codec.affine_center_perm=None;codec.scale_contract='calfa_center_radial_v1'
        pr=c['projection'];assert pr['key']=='model.layers.%d.self_attn.o_proj.weight'%self.layer
        with safe_open(pr['file'],framework='pt',device='cpu') as f:self.projection=f.get_tensor(pr['key']).cuda()
        self.projection32=self.projection.float()
        mc=AutoConfig.from_pretrained(c['model'],trust_remote_code=True,local_files_only=True)
        model_config=dict(rope_theta=resolve_qwen3_rope_theta(mc),max_position_embeddings=mc.max_position_embeddings,rope_scaling=mc.rope_parameters,attention_bias=mc.attention_bias)
        self.native=NativeJointOutput(self.pool,self.layer,self.projection,model_config)
        self.cos=self.native.trig[:2056,:64,None].transpose(1,2);self.sin=self.native.trig[:2056,64:,None].transpose(1,2)
        self.raw=[];self.value=[];self.queries=[];self.targets=[];self.counts=[];self.refs=[]
        for w in range(64):
            row=c['teacher_layers'][str(w)];t=torch.load(row['file'],map_location='cpu',weights_only=True)
            meta=torch.load(c['cache_metadata'][str(w)]['file'],map_location='cpu',weights_only=True)
            assert t['mode']=='bf16' and t['window']==meta['window']==w and t['layer']==meta['layer']==self.layer
            assert t['positions']==meta['positions']==c['query_positions'] and t['rope_theta']==meta['rope_theta']==5000000
            x=t['tensors'];self.raw.append(x['k_pre_rope_full'].reshape(2056,8,128).cuda());self.value.append(x['v_live_full'].reshape(2056,8,128).cuda())
            self.queries.append(x['q_post_rope'].reshape(16,32,128).cuda());self.targets.append(x['attention_output_post_wo'].reshape(16,2560).cuda())
            self.counts.append([r['quant_tokens'] for r in meta['query_metadata']])
        self.raw=torch.stack(self.raw);self.value=torch.stack(self.value);self.queries=torch.stack(self.queries);self.targets=torch.stack(self.targets)
        assert self.raw.dtype==self.value.dtype==self.queries.dtype==self.targets.dtype==torch.bfloat16
        self.center_unit=self.raw[FIT].double().square().mean((0,1)).sqrt().clamp_min(1e-4).float()
        self.energy=[float(t.double().square().sum()) for t in self.targets];self.fit_root=math.sqrt(math.fsum(self.energy[w] for w in FIT))
        for w in range(64):
            k,v=self.raw[w],self.value[w];kh,_=self.pool._prepare_hp_kv_tensors(self.layer,k,v,False,v_rotation_absorbed=True)
            physical=torch.zeros_like(kh,dtype=torch.float32).scatter(-1,self.permutation.expand_as(kh),kh.float()/self.pool._vq.w_perm[self.layer])
            hp=rotate(physical,self.cos,self.sin).half().float();history=torch.arange(2056,device='cuda')[None,:]<torch.tensor(self.counts[w],device='cuda')[:,None]
            causal=torch.arange(2056,device='cuda')[None,:]<=torch.tensor(c['query_positions'],device='cuda')[:,None]
            self.refs.append(dict(q=self.queries[w].float(),cos=self.cos,sin=self.sin,history=history,causal=causal,hp_scores=dot(self.queries[w].float(),hp),projection=self.projection32,teacher=self.targets[w].float()))
        self.apply_aux(self.zero)

    def unpack(self,point):
        books=point[:self.book_count].reshape_as(self.parent)*self.units;aux=point[self.book_count:];assert aux.numel()==1536
        center=center_from_coordinates(aux[512:],self.center_unit);radial=radial_from_coordinates(self.original_radial,aux[:512],self.log_unit)
        return books,aux,center,radial

    def pack(self,books,aux):return torch.cat([(books/self.units).reshape(-1),aux])

    @torch.no_grad()
    def apply_aux(self,aux):
        _,_,center,radial=self.unpack(self.pack(self.parent,aux));assert torch.isfinite(center).all() and torch.isfinite(radial).all() and (radial>0).all()
        self.center,self.radial=center.clone(),radial.clone();vq=self.pool._vq
        vq.affine_center[self.layer].copy_(center);vq.affine_radial[self.layer].copy_(radial)
        vq.affine_center_perm=vq.affine_center.gather(-1,vq.channel_permutations).contiguous() if center.any() else None
        from calfa_affine import freeze_scale
        self.scale=freeze_scale(self.raw,center,radial);self.scale_anchor32=tree_rms((self.raw.float()-center)*radial)
        assert torch.equal(self.weight,vq.channel_weights[self.layer]) and torch.equal(self.permutation,vq.channel_permutations[self.layer])

    @torch.no_grad()
    def install_books(self,books):
        from sglang.srt.mem_cache import vq_codebook
        for side,codec in enumerate([self.pool._vq,self.pool._vq_v]):
            half=books[side].half();assert torch.isfinite(half).all();codec.cb16[self.layer].copy_(half)
            codec.cb_sq[self.layer].copy_(.5*half.cpu().float().square().sum(-1))
        # This process learns C. The original cache key omits tensor mutation versions.
        vq_codebook._VQ_PAD_CACHE.clear()

    @torch.no_grad()
    def actual(self,point):
        books,aux,_,_=self.unpack(point);books=books.half().float();self.apply_aux(aux);self.install_books(books)
        outputs=[];ik=[];iv=[];sse=[]
        for w in range(64):
            x=self.native.forward(w,self.raw[w],self.value[w],self.queries[w],self.counts[w])
            assert x['V_unit'] and torch.equal(x['N'],self.scale[w])
            outputs.append(x['post']);ik.append(x['K_indices']);iv.append(x['V_indices'])
            sse.append(float((x['post'].double()-self.targets[w].double()).square().sum()))
        metrics={name:math.fsum(sse[w] for w in windows)/math.fsum(self.energy[w] for w in windows) for name,windows in SPLITS.items()}
        assert all(math.isfinite(x) for x in metrics.values())
        return dict(books=books,aux=aux.clone(),center=self.center.clone(),radial=self.radial.clone(),N=self.scale.clone(),
            metrics=metrics,sse=sse,outputs=torch.stack(outputs),labels_k=torch.stack(ik),labels_v=torch.stack(iv))

    def functions(self,labels,point):
        self.apply_aux(point[self.book_count:]);anchors=self.scale_anchor32.detach();scales=self.scale.detach()
        def one(w):
            def function(p):
                books,_,center,radial=self.unpack(p)
                return residual(books,center,radial,self.raw[w].float(),self.value[w].float(),labels[0][w].long(),labels[1][w].long(),
                    self.weight,self.permutation,self.refs[w],scales[w],anchors[w],self.fit_root)
            return function
        return [one(w) for w in FIT]

    def snapshot(self,state,full=True):
        snap={k:(v.half().cpu() if k=='books' else v.cpu()) if isinstance(v,torch.Tensor) else v for k,v in state.items() if full or k not in ['labels_k','labels_v']}
        if full:snap.update(weight=self.weight.cpu(),permutation=self.permutation.cpu(),center_unit=self.center_unit.cpu(),units=self.units.cpu(),log_unit=self.log_unit,energy=self.energy,target=self.targets.cpu())
        snap['scope']='Same frozen BF16 teacher inputs; actual joint quantized native attention and BF16 WO acceptance; fixed best D/P.'
        return snap
