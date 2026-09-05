"""One JSON profile shared by all independent entrypoints and Ray workers."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('GRASP_ROOT', Path(__file__).resolve().parent)).resolve()
DEFAULTS = {
    'seed': 20260905,
    'model': 'models/Qwen3-VL-2B-Instruct-224',
    'data': {'prepared_dir': 'prepared', 'image_root': 'dataset/data/GraSP',
             'train': 'train.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'},
    'vision': {'max_side': 224, 'min_pixels': 4096, 'max_pixels': 50176, 'max_images': 16},
    'lora': {'rank': 8, 'alpha': 16, 'target_modules': 'all-linear', 'exclude_modules': '.*visual.*'},
    'hardware': {'gpus_per_node': 1, 'nodes': 1, 'tensor_parallel': 1, 'ray_cpus': 4,
                 'object_store_bytes': 536870912},
    'sft': {'batch_size': 4, 'micro_batch_size': 1, 'max_length': 3072,
            'lr': 1e-4, 'epochs': 3, 'steps': None},
    'grpo': {'batch_size': 4, 'mini_batch_size': 4, 'micro_batch_size': 1,
             'lr': 1e-5, 'epochs': 1, 'steps': 35, 'group_size': 4, 'temperature': 1.0,
             'prompt_tokens': 1280, 'response_tokens': 512, 'max_num_seqs': 4,
             'gpu_memory_fraction': 0.50, 'param_offload': False, 'optimizer_offload': False,
             'agent_workers': 1, 'enforce_eager': False},
    'evaluation': {'per_task': 0, 'gpu_memory_fraction': 0.80, 'max_model_len': 4096,
                   'max_num_seqs': 1, 'response_tokens': 2048, 'enforce_eager': False},
    'sft_overrides': {}, 'grpo_overrides': {},
}

def merge(base, custom, prefix=''):
    result = copy.deepcopy(base)
    for key, value in custom.items():
        if key not in base:
            raise ValueError(f'Unknown profile key: {prefix}{key}')
        if key.endswith('_overrides'):
            result[key] = value
        elif isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{prefix}{key} must be an object')
            result[key] = merge(base[key], value, prefix + key + '.')
        else:
            result[key] = value
    return result

PROFILE_PATH = Path(os.environ.get('GRASP_CONFIG', ROOT / 'profiles/workflow_3060.json')).resolve()
CFG = merge(DEFAULTS, json.loads(PROFILE_PATH.read_text(encoding='utf-8')))
os.environ['GRASP_CONFIG'] = str(PROFILE_PATH)
os.environ['GRASP_ROOT'] = str(ROOT)

def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p

def split_path(split):
    return resolve(CFG['data']['prepared_dir']) / CFG['data'][split]

def save_profile(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'experiment_profile.json').write_text(json.dumps(CFG, indent=2) + '\n')

def validate():
    h, s, g = CFG['hardware'], CFG['sft'], CFG['grpo']
    world = h['gpus_per_node'] * h['nodes']
    if min(world, h['tensor_parallel'], s['batch_size'], g['batch_size'], g['mini_batch_size']) <= 0:
        raise ValueError('GPU and batch counts must be positive')
    if world % h['tensor_parallel'] or s['batch_size'] % world or g['mini_batch_size'] % world:
        raise ValueError('Batch/world-size/tensor-parallel divisibility mismatch')
    if s['micro_batch_size'] != 1:
        raise ValueError('Pinned visual SFT bridge requires micro_batch_size=1; increase effective batch instead')
    if g['group_size'] < 2 or g['batch_size'] % g['mini_batch_size']:
        raise ValueError('GRPO needs group_size >= 2 and batch_size divisible by mini_batch_size')
    for section in (s, g):
        if section['steps'] is not None and section['steps'] < 1:
            raise ValueError('Training steps must be positive or null')
    if CFG['vision']['min_pixels'] > CFG['vision']['max_pixels']:
        raise ValueError('Invalid image pixel limits')

validate()
if __name__ == '__main__':
    import sys
    value = CFG
    for key in sys.argv[1].split('.') if len(sys.argv) > 1 else []:
        value = value[key]
    print(json.dumps(value) if isinstance(value, (dict, list)) else value)
