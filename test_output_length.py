import unittest
from calibrate_output_length import build_report, recommend_cap


class OutputLengthTests(unittest.TestCase):
    def test_user_example(self):
        self.assertEqual(recommend_cap([100, 1135], 512), 1536)

    def test_strictly_larger_at_boundary(self):
        self.assertEqual(recommend_cap([512], 512), 1024)

    def test_length_outputs_are_excluded(self):
        rows = [
            {'_arm':'base','task_id':'A2','output_tokens':100,'finish_reason':'stop'},
            {'_arm':'base','task_id':'P1','output_tokens':2048,'finish_reason':'length'},
            {'_arm':'sft','task_id':'A2','output_tokens':300,'finish_reason':'stop'},
        ]
        report = build_report(rows, 512, 'val')
        self.assertEqual(report['excluded_length_limited'], 1)
        self.assertEqual(report['normal_outputs']['mean'], 200)
        self.assertEqual(report['normal_outputs']['max'], 300)
        self.assertEqual(report['recommended_max_output_tokens'], 512)

    def test_test_split_cannot_tune_cap(self):
        with self.assertRaisesRegex(ValueError, 'held-out test'):
            build_report([], 512, 'test')

    def test_quantum_is_power_of_two(self):
        with self.assertRaisesRegex(ValueError, 'power of two'):
            recommend_cap([1], 500)


if __name__ == '__main__':
    unittest.main()
