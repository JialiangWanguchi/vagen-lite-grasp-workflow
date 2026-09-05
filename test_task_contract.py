"""CPU-only regression tests; synthetic strings, no private images or labels."""
import unittest
from task_contract import evaluate_answer, ground_truth, prompt_parts, sft_target

class TaskContractTests(unittest.TestCase):
    def test_a2_target(self):
        row = {'task_id': 'A2', 'gt_answer': 'C'}
        self.assertEqual(evaluate_answer(sft_target(row), row)['reward'], 1)

    def test_p1_target(self):
        row = {'task_id': 'P1', 'gt_answer': '["B","C","A"]'}
        self.assertEqual(ground_truth(row), ['B', 'C', 'A'])
        self.assertEqual(evaluate_answer(sft_target(row), row)['reward'], 1)

    def test_valid_but_wrong_has_no_reward(self):
        result = evaluate_answer('{"think":"","answer":"B"}', {'task_id':'A2','gt_answer':'A'})
        self.assertTrue(result['format_valid'])
        self.assertEqual(result['reward'], 0)

    def test_duplicate_key_rejected(self):
        result = evaluate_answer('{"think":"","answer":"A","answer":"A"}', {'task_id':'A2','gt_answer':'A'})
        self.assertFalse(result['format_valid'])

    def test_fences_and_extra_keys_rejected(self):
        row = {'task_id':'A2','gt_answer':'A'}
        for text in ('```json\n'+sft_target(row)+'\n```', '{"think":"","answer":"A","extra":0}'):
            self.assertFalse(evaluate_answer(text, row)['format_valid'])

    def test_illegal_permutations_rejected(self):
        row = {'task_id':'P1','gt_answer':['A','B','C']}
        for answer in ('["A","A","C"]', '"ABC"', '["A","B"]', '["A","B",3]'):
            self.assertFalse(evaluate_answer('{"think":"","answer":'+answer+'}', row)['format_valid'])

    def test_prompt_replaces_filename(self):
        path = 'data/GraSP/example/frame.jpg'
        row = {'question': 'Before '+path+' After', 'clips':[{'clip_label':'B','frames':[{'image_path':path}]}]}
        parts = prompt_parts(row)
        self.assertEqual([p['type'] for p in parts], ['text','image','text'])
        self.assertNotIn(path, ''.join(p['text'] for p in parts if p['type']=='text'))
        self.assertIn('Clip B, Frame 1', parts[0]['text'])

    def test_missing_image_slot_rejected(self):
        row = {'question':'No image reference', 'clips':[{'clip_label':'A','frames':[{'image_path':'data/GraSP/example/f.jpg'}]}]}
        with self.assertRaises(ValueError): prompt_parts(row)

if __name__ == '__main__':
    unittest.main()
