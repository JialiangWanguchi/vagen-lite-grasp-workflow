"""Multimodal dataset for VERL's engine-based SFTTrainer; no HF Trainer."""
import torch
from torch.utils.data import Dataset
from grasp_common import load_rows, messages_for, processor
from task_contract import sft_target

class GraSPSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [row for path in paths for row in load_rows(path)]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.processor = processor()
        self.max_length = config.max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        messages, images = messages_for(row)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(
            messages + [{'role': 'assistant', 'content': sft_target(row)}],
            tokenize=False, add_generation_prompt=False)
        prompt_ids = self.processor(text=[prompt], images=images, return_tensors='pt')['input_ids'][0]
        encoded = self.processor(text=[full], images=images, return_tensors='pt')
        ids = encoded['input_ids'][0]
        assert torch.equal(ids[:len(prompt_ids)], prompt_ids), 'Assistant masking boundary mismatch'
        if len(ids) > self.max_length:
            raise ValueError(f'No silent image/text truncation: {len(ids)} > {self.max_length}')
        mask = torch.zeros_like(ids)
        mask[len(prompt_ids):] = 1
        from verl.models.transformers.qwen3_vl import get_rope_index
        mrope = get_rope_index(self.processor, ids, image_grid_thw=encoded['image_grid_thw'])
        return {'input_ids': ids, 'position_ids': torch.arange(len(ids)), 'loss_mask': mask,
                'mrope_positions': mrope.transpose(0, 1),
                'pixel_values': encoded['pixel_values'], 'image_grid_thw': encoded['image_grid_thw']}

def install_visual_sft_adapter():
    """Retain VERL optimizer/loss/FSDP; adapt its text-only input bridge for VL.

    The existing collator preserves all tensors as jagged tensors. At microbatch=1,
    restore dense visual tensors and the pinned VERL Qwen image-aware MRoPE positions.
    Packing is disabled: resetting position IDs across samples would be incorrect.
    """
    from verl.utils import model as model_utils
    original_selector = model_utils.get_hf_auto_model_class

    class StrictQwen3Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            from transformers import Qwen3VLForConditionalGeneration
            model, info = Qwen3VLForConditionalGeneration.from_pretrained(
                *args, **kwargs, output_loading_info=True)
            for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
                assert not info.get(field), f'Invalid base model load: {field}: {info[field]}'
            print('STRICT_QWEN3_BASE_LOAD_PASS',flush=True)
            return model

    def select(hf_config):
        return StrictQwen3Loader if hf_config.model_type == 'qwen3_vl' else original_selector(hf_config)

    model_utils.get_hf_auto_model_class = select
    from verl.workers.engine.fsdp import transformer_impl
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    native_micro_batches = transformer_impl.prepare_micro_batches

    def visual_micro_batches(data, **kwargs):
        if 'pixel_values' in data:
            # One sample per microbatch regardless of token lengths. Keep native
            # jagged indexing, ordering restoration and distributed normalization.
            kwargs['min_num_micro_batch'] = int(data.batch_size[0])
        return native_micro_batches(data=data, **kwargs)

    transformer_impl.prepare_micro_batches = visual_micro_batches
    original = FSDPEngineWithLMHead.prepare_model_inputs

    def prepare(self, micro_batch):
        inputs, output_args = original(self, micro_batch)
        if 'pixel_values' in micro_batch:
            assert micro_batch.batch_size[0] == 1, 'Visual adapter requires microbatch 1'
            assert not self.use_remove_padding, 'Visual SFT requires unpacked SDPA'
            inputs['pixel_values'] = micro_batch['pixel_values'].values()
            inputs['image_grid_thw'] = micro_batch['image_grid_thw'].values()
            inputs['position_ids'] = micro_batch['mrope_positions'].values().transpose(0, 1).unsqueeze(1)
        return inputs, output_args

    FSDPEngineWithLMHead.prepare_model_inputs = prepare
    # The pinned SFT engine has no LoRA export, unlike its PPO worker.
    # Export adapters with PEFT's canonical naming after the native checkpoint.
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    native_lora = FSDPEngine._build_lora_module

    def build_lora(self, module):
        adapted = native_lora(self, module)
        trainable = [(k,v) for k,v in adapted.named_parameters() if v.requires_grad]
        count = sum(v.numel() for _,v in trainable)
        assert count > 0, 'No trainable LoRA parameters'
        assert all('lora_' in k and 'visual' not in k for k,_ in trainable)
        print(f'TRAINABLE_CAPACITY_VERIFIED {count}',flush=True)
        return adapted

    FSDPEngine._build_lora_module = build_lora
    native_save = FSDPEngine.save_checkpoint

    def save(self, local_path, *args, **kwargs):
        # Pinned engine passes checkpoint_contents=, but this manager expects
        # checkpoint_config=. Honor the actual config instead of its fallback.
        self.checkpoint_manager.checkpoint_save_contents = list(self.checkpoint_config.save_contents)
        native_save(self, local_path, *args, **kwargs)
        from pathlib import Path
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from peft.utils.save_and_load import get_peft_model_state_dict
        from safetensors.torch import save_file
        adapter = Path(local_path) / 'lora_adapter'
        peft_model = getattr(self.module, '_fsdp_wrapped_module', self.module)
        with FSDP.summon_full_params(self.module, writeback=False):
            # Do not call model.state_dict(): FSDP would clone the entire frozen
            # 4GB base on GPU before PEFT filters it, exhausting 12GB VRAM.
            trainable = {k.replace('_fsdp_wrapped_module.', ''):v.detach().cpu().contiguous()
                         for k,v in peft_model.named_parameters() if 'lora_' in k}
            state = get_peft_model_state_dict(peft_model,state_dict=trainable,save_embedding_layers=False)
        assert state and all('lora_' in k for k in state)
        if torch.distributed.get_rank() == 0:
            adapter.mkdir(parents=True, exist_ok=True)
            save_file(state, str(adapter/'adapter_model.safetensors'))
            peft_model.peft_config['default'].save_pretrained(adapter)
            print(f'SFT_LORA_EXPORTED {adapter} tensors={len(state)}',flush=True)
        torch.distributed.barrier()

    FSDPEngine.save_checkpoint = save
