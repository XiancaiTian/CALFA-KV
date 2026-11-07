"""Actual mixed-VQ attention and BF16 WO, using original decode scheduling."""
from types import SimpleNamespace
import torch


class NativeJointOutput:
    def __init__(self,pool,layer,projection,model_config):
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        from sglang.srt.utils import get_device_core_count
        from sglang.srt.environ import envs
        self.pool,self.layer,self.projection=pool,layer,projection
        assert projection.dtype==torch.bfloat16 and projection.shape==(2560,4096)
        assert model_config['rope_theta']==5000000 and not model_config.get('attention_bias',False)
        set_global_server_args_for_scheduler(SimpleNamespace(rl_on_policy_target=None))
        with torch.device('cuda'):
            rope=get_rope(128,rotary_dim=128,max_position=model_config['max_position_embeddings'],base=5000000,
                rope_scaling=model_config.get('rope_scaling'),dtype=torch.bfloat16)
        self.trig=rope.cos_sin_cache;assert self.trig.dtype==torch.float32
        self.backend_method=TritonAttnBackend.get_num_kv_splits
        self.backend=SimpleNamespace(max_kv_splits=48,static_kv_splits=False,device_core_count=get_device_core_count(0),
            enable_deterministic=False,split_tile_size=None,num_head=32,num_kv_head=8)
        self.hp_max=envs.SGLANG_MIXED_KV_HP_MAX_SPLITS.get();assert self.hp_max==8
        self.positions=list(range(135,2056,128));self.profiles={}
        self.output=torch.empty((16,32,128),device='cuda',dtype=torch.bfloat16)
        self.logits=torch.empty((16,32,self.hp_max+48,128),device='cuda',dtype=torch.float32)
        self.lse=torch.empty((16,32,self.hp_max+48),device='cuda',dtype=torch.float32)
        self.slots=torch.arange(2056,device='cuda',dtype=torch.int64)

    @torch.no_grad()
    def profile(self,w,k,v,counts):
        if w in self.profiles:
            p=self.profiles[w];assert p['raw_k_ptr']==k.data_ptr() and p['raw_v_ptr']==v.data_ptr() and p['counts']==counts;return p
        hp_k,hp_v=self.pool._prepare_hp_kv_tensors(self.layer,k,v,False,v_rotation_absorbed=True)
        prefixes=[128+(j//4)%8 for j in range(w//8*8,w//8*8+8)]
        hp_keys=[];hp_values=[];hp_ptr=[0];quant_ptr=[0];quant_idx=[];qs=[];batch_rows=[];target_rows=[]
        for pos,cut in zip(self.positions,counts):
            assert 0<=cut<pos+1 and 0<pos+1-cut<=72
            hp_keys.append(hp_k[cut:pos+1]);hp_values.append(hp_v[cut:pos+1]);hp_ptr.append(hp_ptr[-1]+pos+1-cut)
            quant_idx.append(torch.arange(cut,device='cuda',dtype=torch.int32));quant_ptr.append(quant_ptr[-1]+cut)
            step=pos-(128+(w//4)%8);active=[j for j,prefix in enumerate(prefixes) if prefix+step<=2055]
            lengths=torch.tensor([prefixes[j]+step+1 for j in active],device='cuda',dtype=torch.int32)
            split=torch.empty(len(active),device='cuda',dtype=torch.int32);self.backend_method(self.backend,split,lengths)
            row=active.index(w%8);qs.append(split[row:row+1]);batch_rows.append(len(active));target_rows.append(row)
        result=dict(hp_k=torch.cat(hp_keys).contiguous(),hp_v=torch.cat(hp_values).contiguous(),
            hp_ptr=torch.tensor(hp_ptr,device='cuda',dtype=torch.int32),hp_idx=torch.arange(hp_ptr[-1],device='cuda',dtype=torch.int32),
            quant_ptr=torch.tensor(quant_ptr,device='cuda',dtype=torch.int32),quant_idx=torch.cat(quant_idx),
            hp_splits=torch.full((16,),self.hp_max,device='cuda',dtype=torch.int32),quant_splits=torch.cat(qs),
            full=torch.tensor([p+1 for p in self.positions],device='cuda',dtype=torch.int32),batch_rows=batch_rows,target_rows=target_rows,
            raw_k_ptr=k.data_ptr(),raw_v_ptr=v.data_ptr(),counts=list(counts))
        self.profiles[w]=result;return result

    @torch.no_grad()
    def forward(self,w,k,v,q,counts):
        from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd_vq2_unified
        from sglang.srt.mem_cache.vq_codebook import vq_unpack_idx
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        p=self.profile(w,k,v,counts);pool=self.pool;layer=self.layer
        assert q.shape==(16,32,128) and q.dtype==torch.bfloat16 and k.shape==v.shape==(2056,8,128)
        pool.set_kv_buffer(SimpleNamespace(layer_id=layer,calfa_v_rotation_absorbed=True),self.slots,k,v)
        decode_attention_fwd_vq2_unified(q,p['hp_k'],p['hp_v'],pool._vq_k_idx_big[layer],None,pool.get_raw_value_buffer(layer),
            pool._vq_k_sz_big[layer],pool._vq_v_sz_big[layer],self.output,p['hp_ptr'],p['hp_idx'],p['quant_ptr'],p['quant_idx'],self.logits,self.lse,
            p['hp_splits'],p['quant_splits'],self.hp_max,48,128**-.5,
            cb16=pool._vq.cb16[layer],cb16_v=pool._vq_v.cb16[layer],quant_v_idx_buffer=pool._vq_v_idx_big[layer],wide_g=True,
            cos_sin_cache=self.trig,full_seq_len=p['full'],pos_offset=0,prefix_tokens=0,freq_idx=pool._vq.freq_idx[layer],w_perm=pool._vq.w_perm[layer],
            center_perm=None if pool._vq.affine_center_perm is None else pool._vq.affine_center_perm[layer])
        projection_layer=SimpleNamespace(weight=self.projection);method=UnquantizedLinearMethod();post=[]
        for qi,(batch,row) in enumerate(zip(p['batch_rows'],p['target_rows'])):
            context=self.output[qi].reshape(1,4096).expand(batch,-1).contiguous()
            post.append(method.apply(projection_layer,context,None)[row])
        return dict(pre=self.output.reshape(16,4096).clone(),post=torch.stack(post),
            K_indices=vq_unpack_idx(pool._vq_k_idx_big[layer,:2056],8).to(torch.uint8),V_indices=vq_unpack_idx(pool._vq_v_idx_big[layer,:2056],8).to(torch.uint8),
            N=pool._vq_k_sz_big[layer,:2056,0,0].clone(),V_unit=bool((pool._vq_v_sz_big[layer,:2056,...,0]==1).all()),
            quant_splits=p['quant_splits'].clone(),hp_max=self.hp_max,batch_rows=p['batch_rows'],target_rows=p['target_rows'])
