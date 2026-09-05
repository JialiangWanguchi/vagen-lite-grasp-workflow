"""Package reusable code/config and small evidence, never models/data/secrets."""
import hashlib
import json
import tarfile
from experiment_config import ROOT

files=[
    'experiment_config.py','grasp_common.py','grasp_env.py','grasp_seed.py','grasp_sft_dataset.py',
    'task_contract.py','train_sft.py','train_grpo.py','train_sft_grpo.py','evaluate_vllm.py',
    'audit_adapter.py','preflight.py','test_experiment_config.py','snapshot_environment.py',
    'summarize_results.py','status.py','export_bundle.py','write_workflow_report.py',
    'runtime.sh','run_sft.sh','run_grpo.sh','run_sft_grpo.sh','run_all.sh','run_logged.sh','finalize_results.sh',
    'MIGRATION.md','ENVIRONMENT.md','WORKFLOW_REPORT.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json',
]
paths=[ROOT/name for name in files if (ROOT/name).is_file()]
paths+=list((ROOT/'profiles').glob('*.json'))
paths+=list((ROOT/'compat').glob('*.py'))+list((ROOT/'compat').glob('*.toml'))
paths+=[ROOT/'reports'/name for name in ('environment.json','pip_freeze.txt','profile_preflight.json',
    'final_results.json','results_summary.md','data_audit.json','split_audit.json','environment_at_launch.json') if (ROOT/'reports'/name).is_file()]
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
target=ROOT/'reports/reusable_workflow.tar.gz'
if target.exists(): raise FileExistsError(f'Refusing to replace {target}')
with tarfile.open(target,'w:gz') as archive:
    for p in paths: archive.add(p,arcname='grasp_workflow/'+str(p.relative_to(ROOT)))
(ROOT/'reports/bundle_manifest.json').write_text(json.dumps({'files':manifest,
    'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'includes_models_or_data':False},indent=2))
print(target)
