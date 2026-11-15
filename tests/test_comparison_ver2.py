"""User-run guards for ver2 aggregation, separate from legacy/cached results."""
from copy import deepcopy
import unittest

from scripts.analyze.compare_sampling_ver2 import comparison_rows_ver2


def reports_ver2():
    result = []
    for method in ('uniform','random_window','d_optimal'):
        for seed in (range(5) if method=='random_window' else (0,)):
            result.append({'protocol_version':2,'mode':'full','success':True,'candidate_space':'raw_window',
                'score_anchor':'window_start','comparison_signature':'same','dataset_key':'ucf',
                'sampling':{'method':method,'budget':.5,'seed':seed},'end_to_end_seconds':10,
                'videos':[{'video':'v','input_sha256':'hash','reference_tokens':2,'evaluation_frames':32,'selected_tokens':1}],
                'metrics':{'test_videos':1,'primary':{'metric':'roc_auc','value':.8},
                           'branch1':{},'branch2':{},'fine_grained':{'mean_map':.1}}})
    return result


class ComparisonVer2Tests(unittest.TestCase):
    def test_statistics(self):
        rows = comparison_rows_ver2(reports_ver2())
        row = next(r for r in rows if r['method']=='random_window')
        self.assertEqual(row['n_seeds'],5)
        self.assertEqual(row['primary_std_pp'],0)
        self.assertEqual(row['uniform_delta_pp'],0)

    def test_refuse_mixed_duplicate_missing_seeds_and_k(self):
        base = reports_ver2()
        for key,value in (('protocol_version',1),('comparison_signature','other'),('candidate_space','fixed_reference_grid')):
            data = deepcopy(base); data[-1][key]=value
            with self.assertRaises(ValueError): comparison_rows_ver2(data)
        with self.assertRaises(ValueError): comparison_rows_ver2(base+[base[0]])
        with self.assertRaises(ValueError): comparison_rows_ver2(base[:-2])
        data = deepcopy(base); data[-1]['videos'][0]['selected_tokens']=2
        with self.assertRaises(ValueError): comparison_rows_ver2(data)


if __name__=='__main__':
    unittest.main()
