"""Independent combined arm: repeat SFT from the base, then continue its LoRA in GRPO."""
import argparse
import subprocess
from grasp_common import ROOT
from pathlib import Path
from experiment_config import save_profile

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',default=str(ROOT/'runs/sft_grpo'));a=p.parse_args()
    extra=['--smoke'] if a.smoke else []
    out=Path(a.output).resolve()
    save_profile(out)
    subprocess.run(['bash',str(ROOT/'run_sft.sh'),'--output',str(out/'sft_stage')]+extra,check=True)
    adapters=list((out/'sft_stage').rglob('adapter_config.json'))
    if not adapters:
        raise RuntimeError('SFT adapter checkpoint missing; do not silently restart GRPO from base')
    adapter=max(adapters,key=lambda x:x.stat().st_mtime).parent
    subprocess.run(['bash',str(ROOT/'run_grpo.sh'),'--output',str(out/'grpo_stage'),'--adapter',str(adapter)]+extra,check=True)
