"""CPU-only invariants for the paired output-budget experiment."""
import copy
import json
from pathlib import Path
import unittest
from experiment_config import DEFAULTS, merge

class EvaluationBudgetTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/'profiles'
        self.old=json.loads((Path(__file__).parent/'docs/results/2026-09-05/experiment_profile.json').read_text())
        self.new=merge(DEFAULTS,json.loads((root/'evaluation_2048_3060.json').read_text()))

    def test_four_times_output_budget(self):
        self.assertEqual(self.new['evaluation']['response_tokens'],4*self.old['evaluation']['response_tokens'])

    def test_context_covers_prompt_and_output(self):
        self.assertGreaterEqual(self.new['evaluation']['max_model_len'],self.new['grpo']['prompt_tokens']+self.new['evaluation']['response_tokens'])

    def test_only_evaluation_limits_changed(self):
        for cfg in (self.old,self.new):
            for key in ('response_tokens','max_model_len'):
                del cfg['evaluation'][key]
        self.assertEqual(self.old,self.new)

    def test_historical_evidence_keeps_original_settings(self):
        self.assertEqual(self.old['evaluation']['response_tokens'],512)
        self.assertEqual(self.old['evaluation']['max_model_len'],2048)
        self.assertEqual(self.new['evaluation']['per_task'],4)

    def test_current_default_uses_validation_calibrated_budget(self):
        self.assertEqual(DEFAULTS['evaluation']['response_tokens'],1024)
        self.assertEqual(DEFAULTS['evaluation']['max_model_len'],4096)

if __name__=='__main__':
    unittest.main()
