#!/usr/bin/env python3
"""Run CALFA layer-local joint center/radial/K/V calibration or replay verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
LEARNING=ROOT/'joint_optimization/core'


def prepare(path):
    c=json.loads(path.read_text())
    for name in ['model','assets','output','input_ids']:
        c[name]=str((path.parent/c[name]).resolve())
    if c['layer'] not in range(36):
        raise ValueError('layer must be 0..35')
    for rows in [c['teacher_layers'],c['cache_metadata']]:
        if set(rows)!={str(w) for w in range(64)}:
            raise ValueError('Capture exactly 64 same-state windows per layer')
        for row in rows.values():
            f=(path.parent/row['file']).resolve()
            if hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Captured input hash differs: '+str(f))
            row['file']=str(f)
    c['projection']['file']=str((path.parent/c['projection']['file']).resolve())
    if c['projection']['key']!='model.layers.%d.self_attn.o_proj.weight'%c['layer']:
        raise ValueError('Projection does not belong to the selected layer')
    if hashlib.sha256(Path(c['projection']['file']).read_bytes()).hexdigest()!=c['projection']['sha256']:
        raise ValueError('Projection file hash differs')
    if hashlib.sha256(Path(c['input_ids']).read_bytes()).hexdigest()!=c['input_ids_sha256']:
        raise ValueError('Frozen token-window hash differs')
    records=json.loads(Path(c['input_ids']).read_text())
    if len(records)!=64 or any(len(row['input_ids'])!=2056 for row in records):
        raise ValueError('Use the 64 frozen 2056-token calibration windows')
    if c.get('gpu',0)<0:
        raise ValueError('gpu must be nonnegative')
    c['gpu']=c.get('gpu',0);c['local_rank']=c['gpu'];c['port']=c.get('port',48800+c['gpu'])
    c['optimization']='joint_calibration'
    c['maximum_rounds']=8;c['cg_iterations']=24;c['maximum_backtracks']=12
    c['fit_windows']=[i for i in range(64) if i%4!=3]
    c['validation_windows']=[i for i in range(64) if i%4==3]
    c['query_positions']=list(range(135,2056,128));c['sequence_length']=2056
    paths=sorted(LEARNING.glob('*.py'))+[Path(__file__).resolve()]
    paths += [ROOT/'backend/python'/n for n in ['calfa_scale.py','calfa_affine.py',
        'sglang/srt/layers/attention/triton_ops/decode_attention.py',
        'sglang/srt/mem_cache/unified_kv_pool.py','sglang/srt/mem_cache/vq_codebook.py']]
    c['calfa_source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--stage',choices=['all','train','verify'],default='all')
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();c=prepare(a.config.resolve())
    if a.validate_only:
        print(json.dumps(dict(passed=True,layer=c['layer'],windows=64,write_performed=False,scope='File identities and static config; native tensor checks occur during training'),indent=2));return
    if a.stage == 'all':
        for stage in ('train', 'verify'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--config', str(a.config.resolve()), '--stage', stage], check=True)
        return
    if os.environ.get('SGLANG_CALFA_ANCHOR_ENABLE','0')!='0':
        raise ValueError('Joint calibration precedes Anchor and must run with Anchor disabled')
    bundle=Path(c['assets'])/'bundle'
    env=dict(SGLANG_ENABLE_MIXED_KV_WINDOWS='1',SGLANG_VQ_PRE_ROPE='1',
         SGLANG_MIXED_KV_PREFIX_TOKENS='0',SGLANG_MIXED_KV_RECENT_TOKENS='64',
         SGLANG_MIXED_KV_HP_DTYPE='bfloat16',SGLANG_MIXED_KV_SCALE_DTYPE='float16',
         SGLANG_VQ_CODEBOOK_PATH=str(bundle/'codebook.pt'),SGLANG_VQ_V_CODEBOOK_PATH=str(bundle/'codebook_v.pt'),
         SGLANG_CALFA_K_ROTATION_PATH=str(bundle/'k_rotation_qqt_r_h_pbr.pt'),
         SGLANG_CALFA_V_ROTATION_PATH=str(bundle/'v_rotation_sst_r_h_pbr.pt'),
         SGLANG_MIXED_KV_EXACT_CHUNKED_PREFILL='0',SGLANG_VQ2_CUDA='0')
    os.environ.update(env)
    sys.path[:0]=[str(ROOT/'backend/python'),str(LEARNING)]
    import torch
    torch.cuda.set_device(c['gpu'])
    if a.stage=='verify':
        done=json.loads((Path(c['output'])/'complete.json').read_text())
        if done.get('calfa_source_sha256')!=c['calfa_source_sha256']:
            raise ValueError('Learning source changed since training; exact replay requires the recorded source')
    from calibration import train,verify
    (train if a.stage=='train' else verify)(c,a.config.resolve())



if __name__=='__main__':
    main()
