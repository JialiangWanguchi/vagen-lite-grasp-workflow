"""Independent SFT-only LoRA entrypoint on the pinned VERL/FSDP engine."""
import argparse
import os
from pathlib import Path
from grasp_common import ROOT, MODEL, SEED
from experiment_config import CFG, split_path, save_profile

def run_sft(output, smoke=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    from grasp_sft_dataset import install_visual_sft_adapter
    install_visual_sft_adapter()
    from verl.trainer.sft_trainer import run_sft as verl_run_sft
    config_dir = str(Path(verl.__file__).parent / 'trainer/config')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name='sft_trainer_engine')
    updates = {
        'model.path': str(MODEL), 'model.lora_rank': 8, 'model.lora_alpha': 16,
        'model.external_lib': 'grasp_seed',
        'model.target_modules': 'all-linear', 'model.exclude_modules': '.*visual.*',
        'model.use_remove_padding': False, 'model.use_fused_kernels': False,
        'model.enable_gradient_checkpointing': True,
        'model.override_config._attn_implementation': 'sdpa',
        'engine.model_dtype': 'bf16', 'engine.use_torch_compile': False,
        'engine.strategy': 'fsdp', 'engine.use_orig_params': False,
        'data.train_files': str(ROOT/'prepared/train.jsonl'),
        'data.val_files': None, 'data.train_batch_size': 4,
        'data.micro_batch_size_per_gpu': 1, 'data.use_dynamic_bsz': True,
        'data.max_token_len_per_gpu': 1024,
        'data.max_length': 3072, 'data.pad_mode': 'no_padding',
        'data.custom_cls.path': str(ROOT/'grasp_sft_dataset.py'),
        'data.custom_cls.name': 'GraSPSFTDataset',
        'optim.lr': 1e-4, 'optim.weight_decay': 0.0,
        'trainer.total_epochs': 3, 'trainer.total_training_steps': 1 if smoke else None,
        'trainer.default_local_dir': str(output), 'trainer.logger': ['console'],
        'trainer.seed': SEED, 'trainer.save_freq': -1, 'trainer.test_freq': -1,
        'trainer.resume_mode': 'disable',
        'checkpoint.save_contents': ['optimizer', 'extra'],
    }
    s, l = CFG['sft'], CFG['lora']
    updates.update({'model.lora_rank':l['rank'], 'model.lora_alpha':l['alpha'],
        'model.target_modules':l['target_modules'], 'model.exclude_modules':l['exclude_modules'],
        'data.train_files':str(split_path('train')), 'data.train_batch_size':s['batch_size'],
        'data.micro_batch_size_per_gpu':s['micro_batch_size'], 'data.max_length':s['max_length'],
        'data.max_token_len_per_gpu':s['max_length'],
        'optim.lr':s['lr'], 'trainer.total_epochs':s['epochs'],
        'trainer.total_training_steps':1 if smoke else s['steps']})
    updates.update(CFG['sft_overrides'])
    for key,value in updates.items():
        OmegaConf.update(cfg,key,value,force_add=True)
    Path(output).mkdir(parents=True,exist_ok=True)
    save_profile(output)
    OmegaConf.save(cfg, str(Path(output)/'resolved_config.yaml'))
    if os.environ.get('GRASP_DRY_CONFIG') == '1':
        print('SFT_CONFIG_PASS', flush=True)
        return
    verl_run_sft(cfg)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'runs/sft'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run_sft(args.output,args.smoke)
