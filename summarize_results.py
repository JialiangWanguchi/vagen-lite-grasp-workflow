"""Create an evidence-only experiment summary; fail if required runs are absent."""
import csv
import datetime
import json
import re
import hashlib
import math
from pathlib import Path
from grasp_common import ROOT, load_rows
from experiment_config import CFG

results={}
sample_sets={}
for arm in ('base','sft','grpo','sft_grpo'):
    path=ROOT/f'results/{arm}/test_metrics.json'
    results[arm]=json.loads(path.read_text())
    expected={task:sum(r['task_id']==task for r in load_rows('test')) for task in ('A2','P1')}
    if CFG['evaluation']['per_task']:
        expected={k:min(v,CFG['evaluation']['per_task']) for k,v in expected.items()}
    assert results[arm]['n']==sum(expected.values())
    assert all(results[arm]['metrics'][task]['n']==n for task,n in expected.items())
    assert results[arm]['decoding']==results['base']['decoding']
    predictions=[json.loads(line) for line in (path.parent/'test_predictions.jsonl').read_text().splitlines()]
    selected_rows=[]
    for row in load_rows('test'):
        if sum(x['task_id']==row['task_id'] for x in selected_rows)<expected[row['task_id']]:
            selected_rows.append(row)
    from task_contract import ground_truth
    assert len(selected_rows)==len(predictions)
    for i,(pred,row) in enumerate(zip(predictions,selected_rows)):
        assert pred['index']==i and pred['task_id']==row['task_id'] and pred['truth']==ground_truth(row)
    # Baseline may have started with the earlier metadata-only schema. Its
    # ordered row IDs are reconstructed from the unchanged split and profile.
    inferred=[hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for row in selected_rows]
    sample_sets[arm]=[r.get('sample_sha256',digest) for r,digest in zip(predictions,inferred)]
    results[arm]['row_identity_source']='recorded' if all('sample_sha256' in r for r in predictions) else 'reconstructed from fixed split, profile, index, task and truth'
    assert sample_sets[arm]==sample_sets['base'], 'Different evaluation rows across arms'
    assert len(predictions)==results[arm]['n']
    for task in ('all','A2','P1'):
        selected=[r for r in predictions if task=='all' or r['task_id']==task]
        for metric in ('exact_match','format_valid'):
            assert abs(sum(r[metric] for r in selected)/len(selected)-results[arm]['metrics'][task][metric])<1e-12
resources={}
for phase in ('base_test','sft_train','sft_test','grpo_train','grpo_test','sft_grpo_train','sft_grpo_test'):
    logs=ROOT/'reports/logs'
    assert (logs/f'{phase}_exit_code.txt').read_text().strip()=='0',f'{phase} failed or unfinished'
    begin=datetime.datetime.fromisoformat((logs/f'{phase}_started.txt').read_text().strip().replace('Z','+00:00'))
    end=datetime.datetime.fromisoformat((logs/f'{phase}_finished.txt').read_text().strip().replace('Z','+00:00'))
    with (logs/f'{phase}_gpu.csv').open() as file:
        rows=list(csv.DictReader(file))
    usage=[float(row[' memory.used [MiB]'].strip().split()[0]) for row in rows]
    resources[phase]={'elapsed_seconds':(end-begin).total_seconds(),'sampled_peak_gpu_MiB':max(usage)}
summary={'results':results,'resources':resources,'status':'all required training and tests completed',
         'profile':CFG,
         'limitations':['Single seed, small script-level workflow test; not an efficacy experiment',
                        'Exact images are split-disjoint, cases and neighboring frames are not',
                        'Only workflow feasibility and descriptive comparison; no clinical or case-generalization claim']}
base=results['base']['metrics']['all']['exact_match']
summary['delta_vs_base']={arm:{'accuracy_difference':r['metrics']['all']['exact_match']-base,
    'relative_change':(r['metrics']['all']['exact_match']-base)/base if base else None}
    for arm,r in results.items() if arm!='base'}
from audit_adapter import audit,locate
summary['artifacts']={arm:audit(locate(ROOT/'runs'/arm)) for arm in ('sft','grpo')}
summary['artifacts']['sft_grpo']=audit(locate(ROOT/'runs/sft_grpo/grpo_stage'),locate(ROOT/'runs/sft_grpo/sft_stage'))
assert len({x['parameters'] for x in summary['artifacts'].values()})==1,'LoRA capacity differs across arms'
summary['training_metrics']={}
for phase in ('sft_train','grpo_train','sft_grpo_train'):
    log=(ROOT/f'reports/logs/{phase}.log').read_text(errors='replace')
    clean=re.sub(r'\x1b\[[0-9;]*m','',log)
    entries=[]
    for line in clean.splitlines():
        if 'step:' not in line: continue
        metrics={k:float(v) for k,v in re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',line)}
        if metrics:entries.append(metrics)
    summary['training_metrics'][phase]=entries
    assert entries, f'No optimizer-step metrics in {phase}'
    if phase in ('grpo_train','sft_grpo_train'):
        policy_steps=[m for m in entries if 'actor/grad_norm' in m]
        assert policy_steps and any(m['actor/grad_norm']>0 for m in policy_steps), f'{phase}: no nonzero policy gradient'
        assert any(m.get('custom_metrics/train/reward_variance',0)>0 for m in policy_steps), f'{phase}: no group reward variation'
        assert all(math.isfinite(m['actor/grad_norm']) for m in policy_steps)
        if CFG['grpo']['steps'] is not None:
            assert max(m['step'] for m in policy_steps)==CFG['grpo']['steps']
    if phase in ('sft_train','sft_grpo_train'):
        supervised_steps=[m for m in entries if 'train/loss' in m]
        assert supervised_steps and all(math.isfinite(m['train/loss']) and math.isfinite(m['train/grad_norm']) for m in supervised_steps)
        if CFG['sft']['steps'] is not None:
            assert max(m['step'] for m in supervised_steps)==CFG['sft']['steps']
(ROOT/'reports/final_results.json').write_text(json.dumps(summary,indent=2))
lines=['# 实际运行结果','', '| 模型 | A2 正确率 | P1 正确率 | 总正确率 | JSON 合法率 |',
       '|---|---:|---:|---:|---:|']
for arm,r in results.items():
    m=r['metrics']
    lines.append(f"| {arm} | {m['A2']['exact_match']:.1%} | {m['P1']['exact_match']:.1%} | {m['all']['exact_match']:.1%} | {m['all']['format_valid']:.1%} |")
lines+=['',f'每个任务的测试样本数：{expected}。所有指标来自保存的逐条预测，不是训练集 reward。',
        '小样本且病例未隔离，不能由这些结果推断临床可靠性、跨病例泛化或算法优越性。','',
        '| 阶段 | 用时（秒） | 采样峰值显存（MiB） |','|---|---:|---:|']
for phase,r in resources.items():lines.append(f"| {phase} | {r['elapsed_seconds']:.0f} | {r['sampled_peak_gpu_MiB']:.0f} |")
(ROOT/'reports/results_summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
