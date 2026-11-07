"""Parallel layer execution of native SimQuant on verified saved BF16 graph features."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import time
import numpy as np
import torch
from safetensors import safe_open
from calfa_calibration import frozen_windows
from calfa_scale import SCALE_CONTRACT
from . import gpu_kmeans
from .simquant_module_quantizer import SimQuant


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_layer(root, layer, expected_input_sha):
    tensors={key:[] for key in ['k_feature','v_feature','k_grad_sq_fp32','v_grad_sq_fp32']}
    identities=[]
    for window in range(64):
        base=Path(root)/('window_%03d'%window)
        meta=json.loads((base/'complete.json').read_text())
        assert meta['window_id']==window and meta['input_ids_sha256']==expected_input_sha
        assert meta['objective']=='full_window_shifted_ce_mean'
        assert meta['k_coordinate']=='post_k_norm_pre_rope'
        assert meta['model_dtype']=='bfloat16'
        assert len(meta['layers'])==36 and meta['input_tokens']==2048
        entry=meta['layers'][layer];path=base/entry['file']
        assert sha(path)==entry['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        assert data['window_id']==window and data['layer_id']==layer
        for side in ('k','v'):
            feature=data[side+'_feature'];signed=data[side+'_grad_signed_fp32'];squared=data[side+'_grad_sq_fp32']
            assert feature.shape==signed.shape==squared.shape==(1,2048,1024)
            assert feature.dtype==torch.bfloat16 and signed.dtype==squared.dtype==torch.float32
            assert torch.equal(signed.square(),squared)
            assert torch.isfinite(feature).all() and torch.isfinite(squared).all()
        for key in tensors:tensors[key].append(data[key])
        identities.append({'window':window,'file_sha256':entry['sha256'],
                           'model_config_sha256':meta['model_config_sha256'],
                           'collector_sha256':meta['collector_sha256']})
    assert len({x['collector_sha256'] for x in identities})==1
    return {key:torch.cat(value,dim=1) for key,value in tensors.items()},identities


def projection(model, layer, side):
    key='model.layers.%d.self_attn.%s_proj.weight'%(layer,side)
    index=json.loads((Path(model)/'model.safetensors.index.json').read_text())
    with safe_open(str(Path(model)/index['weight_map'][key]),framework='pt',device='cpu') as f:
        value=f.get_tensor(key)
    assert value.dtype==torch.bfloat16 and value.shape[0]==1024
    linear=torch.nn.Linear(value.shape[1],value.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
    linear.weight=torch.nn.Parameter(value.cuda(),requires_grad=False)
    return linear


def train_frozen(args):
    # This execution adapter cannot silently enable any additional native ablation.
    assert (args.nsamples,args.seqlen,args.num_coupled,args.abits,args.head_group,args.post_norm_bits)==(64,2048,8,8,8,16)
    assert args.dataset=='gpqa_code' and args.mix_c4_n==16 and args.seed==0
    assert args.quantize and args.nuq and args.per_token_norm and args.ptnorm_weight and args.fisher_native_coord
    assert args.kmeans_iters==100 and args.fisher==args.frozen_feature_dir
    assert not any([args.fisher_kmeans_init,args.fisher_diag_metric,args.maha_metric,
                    args.per_token_norm_v,args.tail_mass_file,args.vocab_probe_path,args.norm])
    assert args.cap_outliers==-1 and args.first_few_fp16==-1
    assert args.frozen_sides in ('k,v', 'k')
    sides = args.frozen_sides.split(',')
    _,inputs=frozen_windows()
    calfa=pickle.load(open(args.calfa_file,'rb'))
    assert calfa['scale_contract']==SCALE_CONTRACT
    assert calfa['frozen_input_manifest']['input_ids_sha256']==inputs['input_ids_sha256']
    layers=[int(x) for x in args.layer_ids.split(',')]
    assert layers and len(set(layers))==len(layers) and all(0<=x<36 for x in layers)
    out=Path(args.quantizer_path);out.parent.mkdir(parents=True,exist_ok=True)
    diag=Path(args.diagnostics_dir);diag.mkdir(parents=True,exist_ok=True)
    assert not out.exists(), 'Do not overwrite a completed quantizer shard'
    quantizers={};layer_records=[]
    for layer in layers:
        tensors,identities=load_layer(args.frozen_feature_dir,layer,inputs['input_ids_sha256'])
        for side in sides:
            start=time.time();seed=args.training_seed_base+2*layer+(side=='v')
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            linear=projection(args.model,layer,side)
            quant=SimQuant(linear,8,8,perm=calfa['perms'][layer] if side=='k' else None,
                           chan_w=calfa['weights'][layer] if side=='k' else None,
                           per_token_norm=side=='k',post_norm_bits=16 if side=='k' else 0,
                           head_group=8 if side=='k' else 1,
                           head_order=calfa.get('head_order',{}).get(layer) if side=='k' else None,
                           ptnorm_weight=True,fisher_native_coord=True,kmeans_iters=100)
            # add_batch is the original feature consumer: same BF16 values and global row order.
            quant.add_batch(None,tensors[side+'_feature'].cuda())
            path=diag/('layer_%02d_%s.jsonl'%(layer,side));assert not path.exists()
            group_start=0
            def observe(payload):
                nonlocal group_start
                with path.open('a') as f:
                    for local,row in enumerate(payload):
                        group=group_start+local
                        row.update(layer=layer,side=side,global_head=group//16,group=group%16,
                                   seed=seed,seed_rule='base+2*layer+is_v; one native batched RNG stream per side',
                                   input_rows=131072)
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                group_start+=len(payload)
            gpu_kmeans._DIAGNOSTICS_OBSERVER=observe
            try:
                with torch.no_grad():
                    result=quant.quantize(include_sparse=args.include_sparse,
                         sparsity_threshold=args.sparsity_threshold,nuq=True,
                         fisher=tensors[side+'_grad_sq_fp32'],norm=False,
                         cap_outliers=False,first_few_fp16=-1)
            finally:gpu_kmeans._DIAGNOSTICS_OBSERVER=None
            assert group_start==128 and result.shape==(128,256,8)
            assert result.dtype==torch.float32 and torch.isfinite(result).all()
            key='model.layers.%d.self_attn.%s_proj'%(layer,side)
            quantizers[key]=result
            row=dict(layer=layer,side=side,seed=seed,seconds=time.time()-start,
                     diagnostic_file=str(path),diagnostic_sha256=sha(path),
                     feature_source_identity=identities)
            layer_records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='feature_source_identity'}),flush=True)
            quant.free();del quant,linear;torch.cuda.empty_cache()
        del tensors
    temp=out.with_suffix(out.suffix+'.tmp')
    with temp.open('wb') as f:pickle.dump(quantizers,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp,out)
    identity=dict(complete=True,layers=layers,trained_sides=sides,model=args.model,input_ids_sha256=inputs['input_ids_sha256'],
                  calfa_file_sha256=sha(args.calfa_file),quantizer_sha256=sha(out),scale_contract=SCALE_CONTRACT,
                  training_seed_base=args.training_seed_base,execution='native SimQuant, saved verified actual BF16 graph features',
                  records=layer_records,source_sha256={str(p):sha(p) for p in
                      [Path(__file__),Path(gpu_kmeans.__file__),Path(__file__).with_name('simquant_module_quantizer.py')]})
    out.with_suffix(out.suffix+'.complete.json').write_text(json.dumps(identity,indent=2)+'\n')
