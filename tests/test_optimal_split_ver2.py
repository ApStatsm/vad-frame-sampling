"""User-run checks for optimal-only reuse and GPU-independent comparison."""
from copy import deepcopy
import unittest

from scripts.analyze.compare_optimal_ver2 import optimal_rows_ver2
from scripts.run_optimal_split_ver2 import reusable_ver2


def fixture_ver2():
    inputs = {k:'hash' for k in ('experiment_config','dataset_config','setup_manifest','local_config',
              'clip_checkpoint','vadclip_checkpoint','gt','segment_gt','class_gt','fine_grained_source',
              'official_list','tokenizer_vocabulary')}
    videos = [{'video':f'v{i}', 'official_csv_index':i, 'input_sha256':'video-hash',
               'reference_tokens':8,'evaluation_frames':128,'gt_offset':128*i,
               'selected_tokens':8,'encoder_input_images':8} for i in range(290)]
    metrics = {'test_videos':290,'primary':{'branch':'branch1','metric':'roc_auc','value':.87},
               'branch1':{'roc_auc':.87,'ap':.3,'ano_auc':.7},'branch2':{},
               'fine_grained':{'mean_map':.1,'map_by_iou':{'0.1':.2}}}
    base = {'success':True,'mode':'full','finished_at':'2026-11-14T00:00:00+09:00',
            'protocol_version':2,'dataset_key':'ucf','dataset':'UCF-Crime','dataset_identity_sha256':'identity',
            'candidate_space':'raw_window','score_anchor':'window_start','sampling':{'method':'d_optimal','budget':1.,'seed':0},
            'experiment':{'sampling':{'method':'d_optimal','budget':1.,'seed':0},'selection_spec':{'version':2},'protocol':{'restoration':'nearest'},'model_seed':234},
            'input_sha256':inputs,'package_sha256':{'module':'hash'},'vendored_source_sha256':{'vendored':'hash'},
            'entrypoint_sha256':'entry','environment':{'device':'cuda','torch':'same','gpu_uuid':'gpu0'},
            'projection':{'q_sha256':'q'},'prompts':['normal'],'model_options':{},'extraction_config':{},
            'videos':videos,'metrics':metrics,'encoder_input_images':290*8,'end_to_end_seconds':100}
    manifest = {'dataset_identity_sha256':'identity','videos':[{
        'filename':v['video'],'official_csv_index':v['official_csv_index'],'sha256':v['input_sha256'],
        'reference_tokens':8,'evaluation_frames':128,'gt_offset':v['gt_offset']} for v in videos]}
    context = {'dataset':'ucf','manifest':manifest,'inputs':{k:v for k,v in inputs.items() if k!='experiment_config'},
               'package':base['package_sha256'],'entrypoint':'entry','vendored':base['vendored_source_sha256']}
    reports = []
    for method in ('d_optimal','a_optimal','e_optimal'):
        for budget in (1.,.5,.25,.125):
            r = deepcopy(base)
            r['sampling'].update(method=method,budget=budget)
            k = round(8*budget)
            for v in r['videos']:
                v.update(selected_tokens=k,encoder_input_images=k)
            r['encoder_input_images']=k*290
            reports.append(r)
    return base,context,reports


class OptimalSplitVer2Tests(unittest.TestCase):
    def test_reuse_compatible_and_gpu_change(self):
        report,context,_ = fixture_ver2()
        self.assertTrue(reusable_ver2(report,report['experiment'],'hash',context))
        report['environment']['gpu_uuid']='gpu1'
        self.assertTrue(reusable_ver2(report,report['experiment'],'hash',context))

    def test_reject_partial_failed_source_config_and_input_changes(self):
        report,context,_ = fixture_ver2()
        for key,value in (('success',False),('finished_at',None),('mode','smoke'),('package_sha256',{}),('videos',[])):
            changed = deepcopy(report); changed[key]=value
            self.assertFalse(reusable_ver2(changed,report['experiment'],'hash',context))
        self.assertFalse(reusable_ver2(report,{'different':'config'},'hash',context))
        self.assertFalse(reusable_ver2(report,report['experiment'],'other-hash',context))
        changed = deepcopy(report); changed['input_sha256']['clip_checkpoint']='other'
        self.assertFalse(reusable_ver2(changed,report['experiment'],'hash',context))

    def test_compare_all_optimal_without_baselines_across_gpus(self):
        _,_,reports = fixture_ver2()
        for index,r in enumerate(reports):
            r['environment'].update(gpu_uuid=f'gpu{index%2}',CUDA_VISIBLE_DEVICES=str(index%2))
        rows = optimal_rows_ver2(reports)
        self.assertEqual(len(rows),12)
        self.assertEqual({r['method'] for r in rows},{'d_optimal','a_optimal','e_optimal'})
        self.assertTrue(all(r['d_delta_pp']==0 for r in rows))

    def test_reject_baseline_missing_duplicate_and_frozen_change(self):
        _,_,reports = fixture_ver2()
        with self.assertRaises(ValueError): optimal_rows_ver2(reports[:-1])
        with self.assertRaises(ValueError): optimal_rows_ver2(reports+[reports[0]])
        for key,value in (('candidate_space','fixed_reference_grid'),('package_sha256',{})):
            changed=deepcopy(reports); changed[-1][key]=value
            with self.assertRaises(ValueError): optimal_rows_ver2(changed)
        changed=deepcopy(reports); changed[-1]['sampling']['method']='random_window'
        with self.assertRaises(ValueError): optimal_rows_ver2(changed)
        changed=deepcopy(reports); changed[-1]['environment']['torch']='different'
        with self.assertRaises(ValueError): optimal_rows_ver2(changed)

    def test_reject_bad_k_and_timeline(self):
        _,_,reports = fixture_ver2()
        changed=deepcopy(reports); changed[-1]['videos'][0]['selected_tokens']=8
        with self.assertRaises(ValueError): optimal_rows_ver2(changed)
        changed=deepcopy(reports); changed[-1]['videos'][0]['gt_offset']=1
        with self.assertRaises(ValueError): optimal_rows_ver2(changed)


if __name__=='__main__':
    unittest.main()
