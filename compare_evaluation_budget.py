"""Recompute paired 512/expanded-budget metrics without publishing private text."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from experiment_config import ROOT
from task_contract import evaluate_answer

ARMS=('base','sft','grpo','sft_grpo')

def load(path):
    return json.loads(path.read_text())

def predictions(directory):
    rows=[json.loads(line) for line in (directory/'test_predictions.jsonl').read_text().splitlines() if line.strip()]
    for record in rows:
        computed=evaluate_answer(record['prediction'],{'task_id':record['task_id'],'gt_answer':record['truth']})
        for key,value in computed.items():
            assert record[key]==value, f'Prediction metric mismatch: {key}'
    return rows

def counts(rows):
    return {task:{'n':len(group),'correct':sum(r['exact_match'] for r in group),
                  'format_valid':sum(r['format_valid'] for r in group),
                  'truncated':sum(r['finish_reason']=='length' for r in group),
                  'output_tokens_max':max(r['output_tokens'] for r in group)}
            for task in ('all','A2','P1')
            if (group:=[r for r in rows if task=='all' or r['task_id']==task])}

def elapsed(logs,name):
    start=datetime.fromisoformat((logs/f'{name}_started.txt').read_text().strip().replace('Z','+00:00'))
    end=datetime.fromisoformat((logs/f'{name}_finished.txt').read_text().strip().replace('Z','+00:00'))
    peak=0
    with (logs/f'{name}_gpu.csv').open() as stream:
        for row in list(csv.reader(stream))[1:]:
            if len(row)>1 and (match:=re.search(r'\d+',row[1])):
                peak=max(peak,int(match.group()))
    return {'elapsed_seconds':(end-start).total_seconds(),'sampled_peak_gpu_MiB':peak}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.prefix)
    out=ROOT/'reports'/args.prefix
    if (out/'budget_comparison.json').exists():
        raise FileExistsError('Comparison already exists; do not overwrite evidence')
    preflight=load(out/'preflight.json')
    original_artifacts=load(ROOT/'reports/final_results.json')['artifacts']
    report={'created_at':datetime.now(timezone.utc).isoformat(),'prefix':args.prefix,
            'scope':'Paired evaluation only; no retraining or changed prompts/rewards.',
            'n':preflight['n'],'new_max_output_tokens':preflight['max_output_tokens'],
            'new_max_model_len':preflight['max_model_len'],'arms':{},
            'limitations':['Single seed and 8 held-out rows; not efficacy or clinical validation.',
                           '512-token base row identity was reconstructed; other rows have recorded SHA.',
                           'Generation budget and supporting context limit changed together; greedy GPU execution is not guaranteed bitwise identical.']}
    for arm in ARMS:
        old_dir=ROOT/'results'/arm
        new_dir=ROOT/'results'/f'{args.prefix}_{arm}'
        old,new=predictions(old_dir),predictions(new_dir)
        assert len(old)==len(new)==preflight['n']
        old_cfg,new_cfg=load(old_dir/'experiment_profile.json'),load(new_dir/'experiment_profile.json')
        old_budget=old_cfg['evaluation']['response_tokens']
        old_context=old_cfg['evaluation']['max_model_len']
        for cfg in (old_cfg,new_cfg):
            for key in ('response_tokens','max_model_len'):
                del cfg['evaluation'][key]
        assert old_cfg==new_cfg, f'{arm}: unrelated configuration changed'
        for index,(a,b) in enumerate(zip(old,new)):
            for key in ('index','task_id','truth','prompt_tokens'):
                assert a[key]==b[key], f'{arm}/{index}: changed {key}'
            assert b['sample_sha256']==preflight['sample_sha256'][index]
            if 'sample_sha256' in a:
                assert a['sample_sha256']==b['sample_sha256']
        metrics=load(new_dir/'test_metrics.json')
        new_counts=counts(new)
        for task,values in new_counts.items():
            saved=metrics['metrics'][task]
            assert saved['n']==values['n']
            assert saved['exact_match']==values['correct']/values['n']
            assert saved['format_valid']==values['format_valid']/values['n']
            assert saved['truncated']==values['truncated']
        phase=f'{args.prefix}_{arm}_test'
        exit_code=int((ROOT/'reports/logs'/f'{phase}_exit_code.txt').read_text())
        assert exit_code==0
        item={'old_budget':old_budget,'old_context':old_context,'old':counts(old),'new':new_counts,
              'exit_code':exit_code,'resources':elapsed(ROOT/'reports/logs',phase),
              'previously_truncated':{'n':sum(r['finish_reason']=='length' for r in old),
                   'now_finished':sum(a['finish_reason']=='length' and b['finish_reason']!='length' for a,b in zip(old,new)),
                   'now_format_valid':sum(a['finish_reason']=='length' and b['format_valid'] for a,b in zip(old,new)),
                   'now_correct':sum(a['finish_reason']=='length' and b['exact_match'] for a,b in zip(old,new))},
              'exact_text_prefix_matches':sum(b['prediction'].startswith(a['prediction']) for a,b in zip(old,new))}
        old_all,new_all=item['old']['all'],item['new']['all']
        item['accuracy_delta_pp']=100*(new_all['correct']-old_all['correct'])/len(new)
        item['accuracy_relative_change']=(new_all['correct']/old_all['correct']-1) if old_all['correct'] else None
        if arm!='base':
            adapter=Path(metrics['adapter']).resolve()
            assert adapter==Path(original_artifacts[arm]['path']).resolve()
            sha=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
            assert sha==original_artifacts[arm]['sha256'],f'{arm}: checkpoint changed'
            item['unchanged_adapter_sha256']=sha
        report['arms'][arm]=item
    (out/'budget_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PAIRED_EVALUATION_COMPARISON_PASS')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
