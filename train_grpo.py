"""Independent GRPO-only entrypoint: VAGEN-Lite one-turn env + vLLM rollout."""
import argparse
import json
import os
from pathlib import Path
from grasp_common import ROOT, MODEL, SEED, load_rows
from experiment_config import CFG, save_profile

def run_grpo(output, adapter=None, smoke=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    from vagen.main_ppo import run_ppo
    import vagen
    config_dir = str(Path(vagen.__file__).parent/'configs')
    verl_config = str(Path(verl.__file__).parent/'trainer/config')
    with initialize_config_dir(config_dir=config_dir,version_base=None):
        cfg=compose(config_name='vagen_multiturn',overrides=[f'hydra.searchpath=[file://{verl_config}]'])
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        rows=load_rows(split)
        specs=[]
        for task in ('A2','P1'):
            indices=[i for i,r in enumerate(rows) if r['task_id']==task]
            if not indices:
                continue
            specs.append({'name':'GraSP','n_envs':len(indices),'data_source':task,
                          'config':{'split':split,'rollout_log':str(output.resolve()/f'{split}_rollouts.jsonl')},'seed_list':indices+[indices[0]],
                          'max_turns':1,'response_length_per_turn':CFG['grpo']['response_tokens']})
        OmegaConf.save(OmegaConf.create({'envs':specs}),str(output/f'{split}_envs.yaml'))
    updates={
        'env_registry.GraSP':'grasp_env.GraSPEnv',
        'data.train_files':str(output/'train_envs.yaml'), 'data.val_files':str(output/'val_envs.yaml'),
        'data.train_batch_size':4, 'data.max_prompt_length':1280,'data.max_response_length':512,
        'data.base_seed':SEED, 'data.seed':SEED,
        'data.custom_cls.path':str(ROOT/'VAGEN-vagen-lite/vagen/gym_agent_dataset.py'),
        'algorithm.adv_estimator':'grpo','algorithm.kl_ctrl.kl_coef':0.0,
        'actor_rollout_ref.model.path':str(MODEL),
        'actor_rollout_ref.model.external_lib':'grasp_seed',
        'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
        'actor_rollout_ref.model.target_modules':'all-linear',
        'actor_rollout_ref.model.exclude_modules':'.*visual.*',
        'actor_rollout_ref.model.use_remove_padding':False,
        'actor_rollout_ref.model.use_fused_kernels':False,
        'actor_rollout_ref.model.enable_gradient_checkpointing':True,
        'actor_rollout_ref.model.override_config.attn_implementation':'sdpa',
        'actor_rollout_ref.actor.strategy':'fsdp', 'actor_rollout_ref.actor.optim.lr':1e-5,
        'actor_rollout_ref.actor.ppo_mini_batch_size':4,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.actor.use_dynamic_bsz':False,
        'actor_rollout_ref.actor.use_torch_compile':False,
        'actor_rollout_ref.actor.use_kl_loss':False,
        'actor_rollout_ref.actor.entropy_coeff':0.0,
        'actor_rollout_ref.actor.fsdp_config.param_offload':False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':False,
        'actor_rollout_ref.actor.fsdp_config.model_dtype':'bf16',
        'actor_rollout_ref.actor.fsdp_config.use_orig_params':False,
        'actor_rollout_ref.rollout.name':'vllm', 'actor_rollout_ref.rollout.mode':'async',
        'actor_rollout_ref.rollout.load_format':'safetensors',
        'actor_rollout_ref.rollout.engine_kwargs.vllm.seed':SEED,
        'actor_rollout_ref.rollout.n':4, 'actor_rollout_ref.rollout.temperature':1.0,
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.gpu_memory_utilization':0.50,
        'actor_rollout_ref.rollout.enforce_eager':False,
        'actor_rollout_ref.rollout.layered_summon':True,
        'actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config':json.dumps({
            'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4]}),
        'actor_rollout_ref.rollout.free_cache_engine':True,
        'actor_rollout_ref.rollout.max_model_len':2048,
        'actor_rollout_ref.rollout.max_num_batched_tokens':2048,
        'actor_rollout_ref.rollout.max_num_seqs':4,
        'actor_rollout_ref.rollout.disable_log_stats':False,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.rollout.multi_turn.enable':True,
        'actor_rollout_ref.rollout.agent.num_workers':1,
        'actor_rollout_ref.rollout.agent.agent_loop_config_path':str(ROOT/'VAGEN-vagen-lite/vagen/configs/agent.yaml'),
        'actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt':json.dumps({'image':16,'video':0}),
        'actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs':json.dumps({'min_pixels':4096,'max_pixels':50176}),
        'actor_rollout_ref.rollout.val_kwargs.do_sample':False,
        'actor_rollout_ref.rollout.val_kwargs.n':1,
        'actor_rollout_ref.actor.checkpoint.save_contents':['optimizer','extra'],
        'trainer.n_gpus_per_node':1,'trainer.nnodes':1,'trainer.logger':['console'],
        'trainer.project_name':'grasp_qwen3_vagen_lite','trainer.experiment_name':output.name,
        'trainer.default_local_dir':str(output),'trainer.total_epochs':1,
        'trainer.total_training_steps':int(os.environ.get('GRASP_SMOKE_STEPS','1')) if smoke else 35,
        'trainer.val_before_train':False,'trainer.test_freq':-1,
        'trainer.save_freq':int(os.environ.get('GRASP_SMOKE_STEPS','1')) if smoke else 35,
        'trainer.resume_mode':'disable',
        'ray_kwargs.ray_init.num_cpus':4,'ray_kwargs.ray_init.object_store_memory':536870912,
    }
    g, l, h, v = CFG['grpo'], CFG['lora'], CFG['hardware'], CFG['vision']
    steps = int(os.environ.get('GRASP_SMOKE_STEPS','2')) if smoke else g['steps']
    updates.update({
        'data.train_batch_size':g['batch_size'], 'data.max_prompt_length':g['prompt_tokens'],
        'data.max_response_length':g['response_tokens'],
        'data.custom_cls.path':str(Path(vagen.__file__).parent/'gym_agent_dataset.py'),
        'actor_rollout_ref.model.lora_rank':l['rank'], 'actor_rollout_ref.model.lora_alpha':l['alpha'],
        'actor_rollout_ref.model.target_modules':l['target_modules'],
        'actor_rollout_ref.model.exclude_modules':l['exclude_modules'],
        'actor_rollout_ref.actor.optim.lr':g['lr'],
        'actor_rollout_ref.actor.ppo_mini_batch_size':g['mini_batch_size'],
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':g['micro_batch_size'],
        'actor_rollout_ref.actor.fsdp_config.param_offload':g['param_offload'],
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':g['optimizer_offload'],
        'actor_rollout_ref.rollout.n':g['group_size'], 'actor_rollout_ref.rollout.temperature':g['temperature'],
        'actor_rollout_ref.rollout.tensor_model_parallel_size':h['tensor_parallel'],
        'actor_rollout_ref.rollout.gpu_memory_utilization':g['gpu_memory_fraction'],
        'actor_rollout_ref.rollout.enforce_eager':g['enforce_eager'],
        'actor_rollout_ref.rollout.max_model_len':g['prompt_tokens']+g['response_tokens'],
        'actor_rollout_ref.rollout.max_num_batched_tokens':g['prompt_tokens']+g['response_tokens'],
        'actor_rollout_ref.rollout.max_num_seqs':g['max_num_seqs'],
        'actor_rollout_ref.rollout.agent.num_workers':g['agent_workers'],
        'actor_rollout_ref.rollout.agent.agent_loop_config_path':str(Path(vagen.__file__).parent/'configs/agent.yaml'),
        'actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt':json.dumps({'image':v['max_images'],'video':0}),
        'actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs':json.dumps({'min_pixels':v['min_pixels'],'max_pixels':v['max_pixels']}),
        'trainer.n_gpus_per_node':h['gpus_per_node'], 'trainer.nnodes':h['nodes'],
        'trainer.total_epochs':g['epochs'], 'trainer.total_training_steps':steps,
        'trainer.save_freq':steps if steps else 100,
        'ray_kwargs.ray_init.num_cpus':h['ray_cpus'],
        'ray_kwargs.ray_init.object_store_memory':h['object_store_bytes'],
    })
    updates.update(CFG['grpo_overrides'])
    if adapter:
        updates['actor_rollout_ref.model.lora_adapter_path']=str(adapter)
    for key,value in updates.items():
        OmegaConf.update(cfg,key,value,force_add=True)
    OmegaConf.save(cfg,str(output/'resolved_config.yaml'))
    save_profile(output)
    from verl.utils.config import omega_conf_to_dataclass
    omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    if os.environ.get('GRASP_DRY_CONFIG') == '1':
        print('GRPO_CONFIG_PASS',flush=True)
        return
    run_ppo(cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'runs/grpo'))
    p.add_argument('--adapter');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run_grpo(a.output,a.adapter,a.smoke)
