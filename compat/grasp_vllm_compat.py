"""Correct upstream multimodal prefixes; do not apply LoRA to the vision tower."""
def register():
    import vllm
    assert vllm.__version__=='0.11.0','Re-audit compatibility patch for other vLLM versions'
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.model_executor.models.module_mapping import MultiModelKeys

    def mapping(self):
        # This version has self.visual, not self.model.visual. The upstream
        # model.visual prefix failed to exclude vision layers from LoRA wrapping.
        return MultiModelKeys.from_string_field(
            language_model='language_model',connector='visual.merger',tower_model='visual.')

    Qwen3VLForConditionalGeneration.get_mm_mapping=mapping
    print('GRASP_QWEN3_VLLM_MM_PREFIX_PATCH',flush=True)
