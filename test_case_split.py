import tempfile
import unittest
from pathlib import Path
from case_split import audit, case_ids, materialize, plan_case_manifest


def row(row_id, task, cases):
    return {'id':row_id, 'task_id':task,
            'clips':[{'frames':[{'image_path':f'data/GraSP/frames/{case}/1.jpg'}]}
                     for case in cases]}


class CaseSplitTests(unittest.TestCase):
    def test_extracts_cases(self):
        self.assertEqual(case_ids(row('x','A2',['CASE001','CASE002'])), {'CASE001','CASE002'})

    def test_connected_a2_graph_can_make_existing_rows_unsplittable(self):
        rows = [row('a','A2',['CASE001','CASE002']), row('b','A2',['CASE002','CASE003'])]
        report = audit(rows)
        self.assertEqual(report['case_components'], 1)
        self.assertFalse(report['strict_three_way_split_of_existing_rows_possible'])

    def test_plan_uses_five_four_four_for_thirteen_cases(self):
        rows = [row(f'p{i}','P1',[f'CASE{i:03d}']) for i in range(1,14)]
        manifest = plan_case_manifest(rows, 7)
        self.assertEqual(manifest['counts'], {'train':5,'val':4,'test':4})

    def test_materializer_rejects_cross_partition_row(self):
        rows = [row('x','A2',['CASE001','CASE002'])]
        manifest = {'case_to_split':{'CASE001':'train','CASE002':'test'}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'cross case partitions'):
                materialize(rows, manifest, Path(tmp))


if __name__ == '__main__':
    unittest.main()
