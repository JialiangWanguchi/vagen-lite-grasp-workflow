"""Read-only progress report; safe to run after a client/SSH disconnect."""
import json
import re
import subprocess
from experiment_config import ROOT, CFG

report={'profile':str(ROOT/'reports/experiment_profile.json'),'phases':{}}
for phase in ('workflow_suite','report_finalization','base_test','sft_train','sft_test','grpo_train','grpo_test','sft_grpo_train','sft_grpo_test'):
    prefix=ROOT/'reports/logs'/phase
    log=prefix.with_suffix('.log')
    code=ROOT/'reports/logs'/f'{phase}_exit_code.txt'
    state='not_started'
    if log.exists(): state='started_no_exit_record'
    if code.exists(): state='passed' if code.read_text().strip()=='0' else 'failed'
    result={'state':state}
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-60000));text=stream.read().decode(errors='replace')
        clean=re.sub(r'\x1b\[[0-9;]*m','',text)
        progress=[line for line in clean.splitlines() if 'step:' in line or '"progress":' in line]
        result['last_progress']=progress[-1][-450:] if progress else None
    report['phases'][phase]=result
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(report,indent=2))
