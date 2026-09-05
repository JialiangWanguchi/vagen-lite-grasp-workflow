"""One visual input contract shared by SFT, VAGEN rollout and vLLM testing."""
import json
import os
from pathlib import Path
from PIL import Image
from task_contract import prompt_parts
from experiment_config import ROOT, CFG, resolve, split_path

MODEL = resolve(CFG['model'])
SEED = CFG['seed']
SYSTEM = 'You are a visual reasoning assistant. Follow the requested answer format exactly.'

def load_rows(split):
    path = Path(split)
    if not path.is_file():
        path = split_path(split)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def load_visual(row):
    content, images = [], []
    for part in prompt_parts(row):
        if part['type'] == 'text':
            content.append(part)
        else:
            relative = part['image_path'].split('data/GraSP/', 1)[-1]
            path = resolve(CFG['data']['image_root']) / relative
            with Image.open(path) as source:
                img = source.convert('RGB')
                # Preserve aspect ratio and every frame. Qwen processor rounds to 32-pixel units.
                side = CFG['vision']['max_side']
                img.thumbnail((side, side), Image.Resampling.LANCZOS)
                images.append(img.copy())
            content.append({'type': 'image'})
    return content, images

def messages_for(row):
    content, images = load_visual(row)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}], images

def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(MODEL), min_pixels=CFG['vision']['min_pixels'],
                                        max_pixels=CFG['vision']['max_pixels'])
