"""Verify a trained LoRA artifact and report nonzero, finite updates."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from safetensors.torch import load_file
import torch

def locate(root):
    candidates=list(Path(root).rglob('adapter_model.safetensors'))
    if not candidates:raise FileNotFoundError(f'No adapter under {root}')
    return max(candidates,key=lambda p:max([int(x) for x in re.findall(r'global_step_(\d+)',str(p))] or [0])).parent

def audit(path,reference=None):
    path=Path(path)
    tensors=load_file(str(path/'adapter_model.safetensors'))
    assert tensors,'Empty adapter'
    from experiment_config import CFG
    config=json.loads((path/'adapter_config.json').read_text())
    assert config['r']==CFG['lora']['rank'] and config['lora_alpha']==CFG['lora']['alpha'], 'Adapter differs from profile'
    for key,value in tensors.items():
        if 'lora_A' in key: assert value.shape[0]==config['r'],f'Invalid rank: {key}'
        if 'lora_B' in key: assert value.shape[1]==config['r'],f'Invalid rank: {key}'
    assert all('lora_' in k and 'visual' not in k for k in tensors), 'Unexpected trainable modules'
    assert all(torch.isfinite(t).all() for t in tensors.values()),'Nonfinite checkpoint'
    b=[v.float() for k,v in tensors.items() if 'lora_B' in k]
    norm=sum(v.square().sum().item() for v in b)**0.5
    assert norm>0,'No evidence of a LoRA update: every B matrix is zero'
    result={'path':str(path),'tensor_count':len(tensors),'parameters':sum(v.numel() for v in tensors.values()),
            'lora_B_l2':norm,'finite':True,'visual_adapters':False,
            'sha256':hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()}
    if reference:
        ref=load_file(str(Path(reference)/'adapter_model.safetensors'))
        assert set(ref)==set(tensors),'Incompatible adapters'
        delta=sum((tensors[k].float()-ref[k].float()).square().sum().item() for k in tensors)**0.5
        assert delta>0,'Combined GRPO did not change SFT checkpoint'
        result['reference']=str(reference);result['parameter_delta_l2']=delta
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--reference');p.add_argument('--path-only',action='store_true')
    a=p.parse_args();path=locate(a.root)
    if a.path_only:print(path)
    else:
        result=audit(path,locate(a.reference) if a.reference else None)
        (path/'artifact_audit.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
