"""User-run synthetic CPU checks: selected-only encoding, Full identity and tails.

No real video, checkpoint, CLIP forward, ffmpeg subprocess or GPU is used.
These checks were written but not run during authoring.
"""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from vadfs.extract import crop_image, extract_reference


class FakeCapture:
    def __init__(self, frames):
        self.count = frames
        self.position = 0

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_FRAME_WIDTH: 224, cv2.CAP_PROP_FRAME_HEIGHT: 224,
                cv2.CAP_PROP_POS_MSEC: self.position * 1000 / 30}[key]

    def getBackendName(self):
        return 'synthetic'

    def read(self):
        if self.position >= self.count:
            return False, None
        frame = np.full((224, 224, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        pass


class FakeEncoder:
    visual = SimpleNamespace(conv1=SimpleNamespace(weight=torch.zeros(1)))

    def encode_image(self, images):
        return images.mean(dim=(1, 2, 3))[:, None].repeat(1, 512)


class ExtractionSelectionTests(unittest.TestCase):
    def test_xd_crop_keeps_large_frames_at_native_size(self):
        image = np.random.default_rng(234).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        crop = crop_image(image, 0, 'shortside_min256')
        np.testing.assert_array_equal(crop, image[68:292, 208:432])

    def test_xd_crop_upscales_small_frames_before_cropping(self):
        image = np.random.default_rng(234).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        enlarged = cv2.resize(image, None, fx=256 / 240, fy=256 / 240, interpolation=cv2.INTER_LINEAR)
        crop = crop_image(image, 0, 'shortside_min256')
        np.testing.assert_array_equal(crop, enlarged[16:240, 58:282])

    def extract(self, frames, tokens, selected=None, *, frame_selector=None):
        config = {'temporal_mode': 'single_frame', 'frame_offset': 0,
                  'color_conversion': 'opencv_bgr', 'image_geometry': 'shortside_fxfy',
                  'tail_policy': 'match_official_rows', 'crop_type': 5,
                  'encoder_batch_size': 2, 'feature_l2_normalize': False}
        def preprocess(image):
            return torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1).float()
        with patch('vadfs.extract.cv2.VideoCapture', return_value=FakeCapture(frames)):
            return extract_reference(Path('synthetic.mp4'), FakeEncoder(), preprocess,
                                     'cpu', config, np.float32, tokens, selected,
                                     frame_selector=frame_selector)

    def test_full_selection_matches_legacy_default_path(self):
        old, old_selection, _ = self.extract(48, 3)
        new, new_selection, _ = self.extract(48, 3, [0, 16, 32])
        np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(old_selection['selected_frame_indices'], new_selection['selected_frame_indices'])

    def test_sparse_path_encodes_only_requested_frames(self):
        features, selection, metadata = self.extract(48, 3, [0, 32])
        self.assertEqual(features.shape, (2, 512))
        self.assertEqual(metadata['decoded_frames'], 48)
        self.assertEqual(metadata['encoder_input_images'], 2)
        self.assertEqual(metadata['candidate_frames'], 3)
        np.testing.assert_array_equal(selection['token_start_frames'], [0, 32])

    def test_floor_and_ceil_tail_follow_reference_metadata(self):
        floor, _, _ = self.extract(17, 1)
        ceil, selected, _ = self.extract(17, 2)
        self.assertEqual(len(floor), 1)
        self.assertEqual(len(ceil), 2)
        np.testing.assert_array_equal(selected['selected_frame_indices'], [0, 16])

    def test_frame_selector_encodes_off_grid_frames_and_keeps_real_indices(self):
        features, selected, metadata = self.extract(33, 3, frame_selector=lambda count: [15, count - 1])
        np.testing.assert_array_equal(features[:, 0], [15, 32])
        np.testing.assert_array_equal(selected['selected_frame_indices'], [15, 32])
        self.assertEqual(metadata['candidate_frames'], 3)
        self.assertEqual(metadata['tail_frames_encoded'], 1)


if __name__ == '__main__':
    unittest.main()
