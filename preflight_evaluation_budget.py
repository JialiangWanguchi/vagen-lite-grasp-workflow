"""Read-only checks of paired evaluation inputs; writes only a new run manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from experiment_config import CFG, ROOT, save_profile, split_path
from grasp_common import load_rows, messages_for, processor
from task_contract import ground_truth

def selected_rows():
    rows = load_rows('test')
    limit = CFG['evaluation']['per_task']
    if not limit:
        return rows
    counts, selected = {}, []
    for row in rows:
        task = row['task_id']
        if counts.get(task, 0) < limit:
            selected.append(row)
            counts[task] = counts.get(task, 0) + 1
    return selected

def row_digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.prefix), 'Unsafe result prefix'
    out = ROOT/'reports'/args.prefix
    if out.exists():
        raise FileExistsError('Use a fresh prefix; existing evidence is protected')
    rows = selected_rows()
    assert rows
    for arm in ('base','sft','grpo','sft_grpo'):
        assert not (ROOT/'results'/f'{args.prefix}_{arm}').exists()
        old = [json.loads(line) for line in (ROOT/'results'/arm/'test_predictions.jsonl').read_text().splitlines() if line.strip()]
        assert len(old)==len(rows), f'{arm}: different evaluation size'
        for index,(row,record) in enumerate(zip(rows,old)):
            assert record['index']==index and record['task_id']==row['task_id']
            assert record['truth']==ground_truth(row)
            if record.get('sample_sha256'):
                assert record['sample_sha256']==row_digest(row)
    proc=processor(); lengths=[]
    for row in rows:
        messages,images=messages_for(row)
        prompt=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        length=int(proc(text=[prompt],images=images,return_tensors='pt')['input_ids'].shape[-1])
        assert length+CFG['evaluation']['response_tokens']<=CFG['evaluation']['max_model_len']
        lengths.append(length)
    save_profile(out)
    report={'prefix':args.prefix,'n':len(rows),'input_tokens':lengths,
            'sample_sha256':[row_digest(row) for row in rows],
            'split_sha256':hashlib.sha256(split_path('test').read_bytes()).hexdigest(),
            'max_output_tokens':CFG['evaluation']['response_tokens'],
            'max_model_len':CFG['evaluation']['max_model_len'],
            'baseline_identity_note':'Original base rows lack SHA; checked fixed split, row order, index, task and truth.'}
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EVALUATION_BUDGET_PREFLIGHT_PASS',json.dumps({'n':len(rows),'input_tokens':lengths,'max_output_tokens':CFG['evaluation']['response_tokens']}),flush=True)

if __name__=='__main__':
    main()
