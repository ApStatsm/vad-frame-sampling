"""User-run checks for window constraints, objectives, and legacy equivalence."""
import unittest
from unittest.mock import patch

import numpy as np

from vadfs.samplers.random_window import select_frames
from vadfs.window_sampling_ver2 import (collect_ver2, feature_row_ver2, project_ver2,
    projection_ver2, select_ver2, windows_ver2)


class WindowSamplingVer2Tests(unittest.TestCase):
    def test_tail_and_budget(self):
        for n,frames in ((10,160),(10,165),(11,165),(1,7)):
            for ratio in (1.,.5,.25,.125):
                bounds = windows_ver2(n,frames,ratio)
                self.assertEqual(len(bounds),int(np.ceil(n*ratio)))
                self.assertEqual(bounds[-1][1],min(frames,n*16))
                self.assertTrue(all(a<b for a,b in bounds))
        with self.assertRaises(ValueError):
            windows_ver2(10,16,.5)

    def test_random_matches_legacy(self):
        for ratio in (1.,.5,.25,.125):
            bounds = windows_ver2(11,165,ratio)
            for seed in range(5):
                result,_ = select_ver2(None,bounds,'random_window',seed)
                self.assertEqual(result,select_frames([a for a,_ in bounds],int(16/ratio),165,11,seed))

    def test_feature_definition(self):
        black = np.zeros((40,50,3),dtype=np.uint8)
        row,previous = feature_row_ver2(black)
        np.testing.assert_array_equal(row,np.zeros(8))
        white = np.full_like(black,255)
        row,_ = feature_row_ver2(white,previous)
        np.testing.assert_allclose(row,[1,1,1,0,0,0,0,1],atol=1e-12)

    def test_projection_and_no_input_mutation(self):
        q = projection_ver2()
        np.testing.assert_allclose(q.T@q,np.eye(8),atol=1e-12)
        x = np.arange(64,dtype=float).reshape(8,8)
        copy = x.copy()
        self.assertEqual(project_ver2(x,1,q).shape,(8,1))
        project_ver2(x,8,q)
        np.testing.assert_array_equal(x,copy)

    def test_all_methods_constraints_determinism_and_objectives(self):
        rows = np.random.default_rng(17).random((128,8))
        bounds = windows_ver2(8,128,1.)
        for method in ('d_optimal','a_optimal','e_optimal'):
            chosen,detail = select_ver2(rows,bounds,method)
            self.assertEqual(chosen,select_ver2(rows,bounds,method)[0])
            self.assertEqual(len(chosen),8)
            self.assertTrue(all(a<=i<b for i,(a,b) in zip(chosen,bounds)))
            self.assertLessEqual(len(detail['changed_windows_per_sweep']),5)
            before,after = detail['initial_objectives'],detail['final_objectives']
            if method=='d_optimal':
                self.assertGreaterEqual(after['logdet'],before['logdet']-1e-10)
            elif method=='a_optimal':
                self.assertLessEqual(after['inverse_trace'],before['inverse_trace']+1e-10)
            else:
                self.assertGreaterEqual(after['minimum_eigenvalue'],before['minimum_eigenvalue']-1e-7)

    def test_one_window_matches_direct_best_d_and_a(self):
        rows = np.random.default_rng(9).random((16,8))
        z = project_ver2(rows,1,projection_ver2())[:,0]
        # For one scalar observation both objectives maximize squared magnitude.
        expected = int(np.argmax(z*z))
        for method in ('d_optimal','a_optimal'):
            self.assertEqual(select_ver2(rows,[(0,16)],method)[0],[expected])

    def test_full_budget_does_not_bypass_selection(self):
        rows = np.zeros((16,8)); rows[7,0] = 1
        self.assertEqual(select_ver2(rows,windows_ver2(1,16,1.),'d_optimal')[0],[7])

    def test_zero_and_nonfinite(self):
        for method in ('d_optimal','a_optimal','e_optimal'):
            chosen,detail = select_ver2(np.zeros((32,8)),[(0,16),(16,32)],method)
            self.assertEqual(chosen,[0,16])
            self.assertEqual(detail['fallback'],'zero_projection_uniform')
        with self.assertRaises(ValueError):
            select_ver2(np.full((16,8),np.nan),[(0,16)],'e_optimal')

    def test_collector_decode_failure(self):
        class Capture:
            def isOpened(self): return True
            def get(self,key):
                import cv2
                return 32 if key==cv2.CAP_PROP_FRAME_COUNT else 30
            def read(self): return False,None
            def release(self): pass
        with patch('vadfs.window_sampling_ver2.cv2.VideoCapture',return_value=Capture()):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                collect_ver2('unused.mp4',2)


if __name__=='__main__':
    unittest.main()
