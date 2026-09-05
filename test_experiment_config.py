"""Fast config regression checks; no model weights or GPU required."""
import copy
import unittest
import experiment_config as ec

class ProfileTests(unittest.TestCase):
    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError): ec.merge(ec.DEFAULTS, {'grpo': {'stepz': 4}})

    def test_partial_merge_preserves_common_contract(self):
        cfg=ec.merge(ec.DEFAULTS, {'sft': {'steps': 4}, 'grpo': {'steps': 4}})
        self.assertEqual(cfg['vision'], ec.DEFAULTS['vision'])
        self.assertEqual(cfg['lora'], ec.DEFAULTS['lora'])
        self.assertEqual(cfg['sft']['lr'], 1e-4)

    def test_override_is_not_mutating_defaults(self):
        cfg=ec.merge(ec.DEFAULTS, {'lora': {'rank': 16, 'alpha': 32}})
        self.assertEqual(cfg['lora']['rank'],16)
        self.assertEqual(ec.DEFAULTS['lora']['rank'],8)

    def check_invalid(self, changes):
        original=ec.CFG
        try:
            ec.CFG=ec.merge(ec.DEFAULTS, changes)
            with self.assertRaises(ValueError): ec.validate()
        finally: ec.CFG=original

    def test_visual_microbatch_guard(self): self.check_invalid({'sft': {'micro_batch_size': 2}})
    def test_grpo_group_guard(self): self.check_invalid({'grpo': {'group_size': 1}})
    def test_world_batch_guard(self): self.check_invalid({'hardware': {'gpus_per_node': 3}})
    def test_positive_steps_guard(self): self.check_invalid({'sft': {'steps': 0}})
    def test_pixel_guard(self): self.check_invalid({'vision': {'max_pixels': 100}})

if __name__=='__main__': unittest.main()
