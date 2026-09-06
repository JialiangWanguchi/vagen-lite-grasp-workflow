"""Shared held-out evaluation for base and all arms; vLLM only."""
import argparse
import collections
import json
import time
import hashlib
from pathlib import Path
from grasp_common import ROOT, MODEL, SEED, load_rows, messages_for, processor
from task_contract import judge_answer, ground_truth
from experiment_config import CFG, save_profile

def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--adapter')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    rows=load_rows(a.split)
    if a.limit:
        rows=rows[:a.limit]
    elif CFG['evaluation']['per_task']:
        counts=collections.Counter(); selected=[]
        for row in rows:
            if counts[row['task_id']] < CFG['evaluation']['per_task']:
                selected.append(row);counts[row['task_id']]+=1
        rows=selected
    if not rows: raise ValueError('Evaluation split is empty')
    e,v=CFG['evaluation'],CFG['vision']
    proc=processor()
    llm=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=CFG['hardware']['tensor_parallel'],
            max_model_len=e['max_model_len'],max_num_batched_tokens=e['max_model_len'],max_num_seqs=e['max_num_seqs'],
            gpu_memory_utilization=e['gpu_memory_fraction'],enforce_eager=e['enforce_eager'],seed=SEED,
            compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4]},
            limit_mm_per_prompt={'image':v['max_images'],'video':0},enable_lora=bool(a.adapter),max_lora_rank=CFG['lora']['rank'],
            mm_processor_kwargs={'min_pixels':v['min_pixels'],'max_pixels':v['max_pixels']})
    request=LoRARequest(a.name,1,str(Path(a.adapter).resolve())) if a.adapter else None
    out=ROOT/'results'/a.name;out.mkdir(parents=True,exist_ok=True)
    save_profile(out)
    predictions=[];start=time.monotonic()
    with (out/f'{a.split}_predictions.jsonl').open('w') as stream:
        for i,row in enumerate(rows):
            messages,images=messages_for(row)
            prompt=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            output=llm.generate([{'prompt':prompt,'multi_modal_data':{'image':images}}],
                SamplingParams(temperature=0,max_tokens=e['response_tokens'],seed=SEED),lora_request=request,use_tqdm=False)[0]
            completion=output.outputs[0]
            text=completion.text
            judgement=judge_answer(text,row,finish_reason=completion.finish_reason,
                                    output_tokens=len(completion.token_ids),max_tokens=e['response_tokens'])
            record={'index':i,'task_id':row['task_id'],'prediction':text,'truth':ground_truth(row),
                    'sample_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                    'finish_reason':completion.finish_reason,
                    'prompt_tokens':len(output.prompt_token_ids),'output_tokens':len(completion.token_ids),
                    'max_output_tokens':e['response_tokens'],**judgement}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            predictions.append(record)
            print(json.dumps({'progress':i+1,'total':len(rows),'task':row['task_id'],
                              'strict_correct':record['exact_match'],
                              'accepted_correct':record['accepted_match'],
                              'mode':record['match_mode']}),flush=True)
    summary={'arm':a.name,'split':a.split,'adapter':a.adapter,'elapsed_seconds':time.monotonic()-start,
             'n':len(rows),'decoding':{'temperature':0,'max_tokens':e['response_tokens'],'seed':SEED},'metrics':{}}
    for task in ('all','A2','P1'):
        group=[r for r in predictions if task=='all' or r['task_id']==task]
        if group:
            summary['metrics'][task]={
                'n':len(group),
                'strict_exact_match':sum(r['exact_match'] for r in group)/len(group),
                # Backward-compatible alias: length hard negatives never count as exact.
                'exact_match':sum(r['exact_match'] for r in group)/len(group),
                'accepted_match':sum(r['accepted_match'] for r in group)/len(group),
                'format_valid':sum(r['format_valid'] for r in group)/len(group),
                'fallback_correct':sum(r['match_mode']=='fallback_correct' for r in group),
                'hard_negative_length':sum(r['hard_negative_length'] for r in group),
                'review_required':sum(r['review_required'] for r in group),
                'truncated':sum(r['finish_reason']=='length' for r in group),
            }
    (out/f'{a.split}_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
