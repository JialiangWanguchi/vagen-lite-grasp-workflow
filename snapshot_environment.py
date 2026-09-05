"""Persist non-secret environment and code provenance."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from grasp_common import ROOT, MODEL

versions={name:importlib.metadata.version(name) for name in [
    'torch','transformers','vllm','verl','vagen','peft','flash-attn','ray','tensordict','grasp-vllm-compat']}
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
code={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
      for path in list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+list((ROOT/'profiles').glob('*.json'))+list((ROOT/'compat').glob('*.py'))+list((ROOT/'compat').glob('*.toml'))}
result={'python':platform.python_version(),'packages':versions,'gpu':hardware,'model_view':str(MODEL),
        'vagen_commit':'04bf4bd13bd93688d5cd66331745190486fd14d1',
        'verl_commit':'3fe0a29975e1b02ae2bd1dec249f7807dd7966f5','code_sha256':code}
(ROOT/'reports/environment.json').write_text(json.dumps(result,indent=2))
(ROOT/'reports/pip_freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
print(json.dumps(result,indent=2))
