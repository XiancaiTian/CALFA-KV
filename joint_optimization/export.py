#!/usr/bin/env python3
"""Export all 36 verified selected CALFA layers as one frozen K/V bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-assets',type=Path,default=ROOT/'joint_optimization/coordinates/initialization')
    p.add_argument('--runs',type=Path,required=True,help='Contains layer_00 through layer_35 training outputs')
    p.add_argument('--output',type=Path,required=True,help='New flat serving bundle directory')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a new directory')
    import torch
    k=torch.load(a.initial_assets/'bundle/codebook.pt',map_location='cpu',weights_only=False)
    v=torch.load(a.initial_assets/'bundle/codebook_v.pt',map_location='cpu',weights_only=False)
    if k['scale_contract']!='calfa_fp16_scale_v1':raise ValueError('Expected the zero-center initialization parent')
    centers=[];radials=[];inputs=[]
    for layer in range(36):
        base=a.runs/('layer_%02d'%layer);selected=base/'selected.pt';complete=base/'complete.json'
        done=json.loads(complete.read_text())
        proof=json.loads((base/'independent_GPU_verification.json').read_text())
        if not done['passed'] or done['layer']!=layer or done['optimization']!='joint_calibration' or sha(selected)!=done['selected_sha256']:
            raise ValueError('Unfinished or mismatched selected layer '+str(layer))
        if not proof['passed'] or proof.get('layer')!=layer or proof.get('optimization')!='joint_calibration' or proof.get('selected_sha256')!=done['selected_sha256']:
            raise ValueError('Run verify and preserve its per-layer GPU proof: '+str(layer))
        state=torch.load(selected,map_location='cpu',weights_only=True)
        if not torch.equal(state['weight'],k['channel_weights'][layer]) or not torch.equal(state['permutation'],k['channel_permutations'][layer]):
            raise ValueError('D/P changed or selected layer is misplaced')
        if state['books'].shape!=(2,8,16,256,8) or state['books'].dtype!=torch.float16:
            raise ValueError('Wrong selected book geometry or dtype')
        if state['center'].shape!=(8,128) or state['radial'].shape!=(8,128) or not torch.isfinite(state['center']).all() or not (torch.isfinite(state['radial']) & (state['radial']>0)).all():
            raise ValueError('Invalid selected center/radial')
        for side,blob in enumerate([k,v]):
            if not torch.isfinite(state['books'][side]).all():raise ValueError('Nonfinite selected codeword')
            for head in range(8):blob['codebooks'][layer,head]=list(state['books'][side,head].unbind(0))
        centers.append(state['center'].float());radials.append(state['radial'].float())
        inputs.append(dict(layer=layer,selected_sha256=sha(selected),complete_sha256=sha(complete),verification_sha256=sha(base/'independent_GPU_verification.json')))
    k.update(scale_contract='calfa_center_radial_v1',affine_center=torch.stack(centers),affine_radial=torch.stack(radials),method='CALFA-KV')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(k,a.output/'codebook.pt');torch.save(v,a.output/'codebook_v.pt')
    for n in ['k_rotation_qqt_r_h_pbr.pt','v_rotation_sst_r_h_pbr.pt']:shutil.copy2(a.initial_assets/'bundle'/n,a.output/n)
    files={f.name:sha(f) for f in a.output.iterdir() if f.is_file()}
    (a.output/'export_manifest.json').write_text(json.dumps(dict(method='CALFA-KV',layers=36,
                     input_layers=inputs,files_sha256=files,D_P_fixed=True),indent=2)+'\n')
    print('Exported frozen CALFA bundle:',a.output)


if __name__=='__main__':main()
