"""Validate the real data/profile before any costly GPU job; no training."""
import collections
import hashlib
import json
from experiment_config import CFG, ROOT, split_path, save_profile
from grasp_common import MODEL, load_rows, messages_for, processor
from task_contract import ground_truth, sft_target

def main():
    model_config=json.loads((MODEL/'config.json').read_text())
    assert model_config['model_type']=='qwen3_vl', 'This tested bridge supports dense Qwen3-VL; other architectures need a new compatibility test'
    proc=processor(); report={'model':str(MODEL),'profile':CFG,'splits':{}}
    for split in ('train','val','test'):
        rows=load_rows(split)
        counts=collections.Counter(row['task_id'] for row in rows)
        assert counts['A2'] and counts['P1'], f'{split} must contain both tasks'
        lengths=[]; image_counts=[]
        for row in rows:
            ground_truth(row)
            messages,images=messages_for(row)
            assert 0<len(images)<=CFG['vision']['max_images']
            prompt=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            n=int(proc(text=[prompt],images=images,return_tensors='pt')['input_ids'].shape[-1])
            full=proc.apply_chat_template(messages+[{'role':'assistant','content':sft_target(row)}],
                                          tokenize=False,add_generation_prompt=False)
            total=int(proc(text=[full],images=images,return_tensors='pt')['input_ids'].shape[-1])
            assert n<=CFG['grpo']['prompt_tokens'], f'{split} prompt {n} exceeds GRPO limit'
            assert total<=CFG['sft']['max_length'], f'{split} SFT tokens exceed limit'
            assert n+CFG['evaluation']['response_tokens']<=CFG['evaluation']['max_model_len']
            lengths.append(n);image_counts.append(len(images))
        report['splits'][split]={'n':len(rows),'tasks':dict(counts),'prompt_min':min(lengths),
            'prompt_max':max(lengths),'image_min':min(image_counts),'image_max':max(image_counts),
            'sha256':hashlib.sha256(split_path(split).read_bytes()).hexdigest()}
    train_n=report['splits']['train']['n']
    for stage in ('sft','grpo'):
        c=CFG[stage];available=(train_n//c['batch_size'])*c['epochs']
        assert available>0, f'{stage} batch exceeds dataset'
        assert c['steps'] is None or c['steps']<=available, f'{stage} steps exceed available batches/epochs'
    save_profile(ROOT/'reports')
    (ROOT/'reports/profile_preflight.json').write_text(json.dumps(report,indent=2))
    print('PROFILE_PREFLIGHT_PASS',json.dumps(report['splits']),flush=True)

if __name__=='__main__': main()
