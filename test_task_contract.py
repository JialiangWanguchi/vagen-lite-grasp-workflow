"""CPU-only regression tests; synthetic strings, no private images or labels."""
import unittest
from task_contract import (FALLBACK_REWARD, STRICT_REWARD, evaluate_answer,
                           generation_hit_limit, ground_truth, judge_answer,
                           prompt_parts, sft_target)

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

    def test_strict_judge_reward(self):
        row = {'task_id':'A2','gt_answer':'A'}
        result = judge_answer('{"think":"ok","answer":"A"}', row, finish_reason='stop')
        self.assertEqual(result['reward'], STRICT_REWARD)
        self.assertEqual(result['match_mode'], 'strict_correct')

    def test_a2_high_precision_fallbacks(self):
        row = {'task_id':'A2','gt_answer':'A'}
        for text in ('a', 'Option a', 'Clip A', 'answer is a', 'the first clip',
                     '{"think":"ok","answer":"a"}',
                     '```json\n{"think":"ok","answer":"a"}\n```'):
            with self.subTest(text=text):
                result = judge_answer(text, row, finish_reason='stop')
                self.assertEqual(result['reward'], FALLBACK_REWARD)
                self.assertTrue(result['accepted_match'])

    def test_a2_none_description_fallback(self):
        row = {'task_id':'A2','gt_answer':'E'}
        result = judge_answer('All four clips belong to the same surgical phase.', row)
        self.assertEqual(result['reward'], FALLBACK_REWARD)

    def test_p1_high_precision_fallbacks(self):
        row = {'task_id':'P1','gt_answer':['B','C','A']}
        for text in ('B,C,A', 'B > C > A', 'Clip B then Clip C then Clip A',
                     '{"answer":["b","c","a"],"think":"ok"}'):
            with self.subTest(text=text):
                result = judge_answer(text, row, finish_reason='stop')
                self.assertEqual(result['reward'], FALLBACK_REWARD)
                self.assertEqual(result['normalized_answer'], ['B','C','A'])

    def test_length_is_hard_negative_even_if_answer_is_correct(self):
        row = {'task_id':'A2','gt_answer':'A'}
        text = '{"think":"ok","answer":"A"}'
        for metadata in ({'finish_reason':'length'}, {'truncated':True},
                         {'output_tokens':512,'max_tokens':512}):
            with self.subTest(metadata=metadata):
                result = judge_answer(text, row, **metadata)
                self.assertEqual(result['reward'], 0)
                self.assertFalse(result['exact_match'])
                self.assertTrue(result['hard_negative_length'])
                self.assertTrue(result['content_exact_before_length'])

    def test_stop_at_exact_budget_is_not_inferred_as_truncation(self):
        self.assertFalse(generation_hit_limit(finish_reason='stop', output_tokens=512,
                                              max_tokens=512))

    def test_malformed_or_ambiguous_output_is_not_rescued(self):
        row = {'task_id':'A2','gt_answer':'A'}
        for text in ('{"think":"unfinished","answer":"A"',
                     '```json\n{"think":"","answer":"A"}',
                     'The answer could be A or B.'):
            with self.subTest(text=text):
                self.assertEqual(judge_answer(text, row)['reward'], 0)

    def test_closed_json_with_invalid_think_can_salvage_unique_answer(self):
        row = {'task_id':'A2','gt_answer':'A'}
        text = '{"think":"an "unescaped" quote","answer":"a"}'
        result = judge_answer(text, row, finish_reason='stop')
        self.assertEqual(result['reward'], FALLBACK_REWARD)
        self.assertEqual(result['normalized_answer'], 'A')

    def test_answer_like_text_inside_malformed_think_is_not_salvaged(self):
        row = {'task_id':'A2','gt_answer':'A'}
        text = '{"think":"I could write \'answer\':\'A\' here" BROKEN}'
        self.assertEqual(judge_answer(text, row, finish_reason='stop')['reward'], 0)

    def test_reasoning_body_is_not_mined_for_answer(self):
        row = {'task_id':'A2','gt_answer':'A'}
        result = judge_answer('I considered A, B, C and D. The images are difficult.', row)
        self.assertEqual(result['reward'], 0)
        self.assertTrue(result['review_required'])

    def test_fallback_wrong_is_still_wrong(self):
        row = {'task_id':'A2','gt_answer':'A'}
        result = judge_answer('clip b', row)
        self.assertTrue(result['fallback_applied'])
        self.assertFalse(result['accepted_match'])
        self.assertEqual(result['reward'], 0)

if __name__ == '__main__':
    unittest.main()
