"""Compatibility locations for legacy UCF scripts; new runs use datasets/ucf."""

from .paths import ROOT
from .datasets.ucf import LABELS


TEST_ROOT = ROOT / 'data/UCF-Crime/test'
MANIFEST = TEST_ROOT / 'manifest.json'
INPUT_AUDIT = ROOT / '.local/setup/ucf/ucf_baseline_input_check.json'
RESOURCE_AUDIT = ROOT / '.local/setup/ucf/ucf_resource_check.json'
LOCAL_FEATURE_LIST = ROOT / '.local/setup/ucf/ucf_CLIP_rgbtest.local.csv'
SMOKE_REPORT = ROOT / '.local/setup/ucf/ucf_smoke_check.json'
FRAME_GT = ROOT / 'VadCLIP/list/gt_ucf.npy'
SEGMENT_GT = ROOT / 'VadCLIP/list/gt_segment_ucf.npy'
CLASS_GT = ROOT / 'VadCLIP/list/gt_label_ucf.npy'
DETECTION_MAP_SOURCE = ROOT / 'VadCLIP/src/utils/ucf_detectionMAP.py'
OFFICIAL_TEST_SOURCE = ROOT / 'VadCLIP/src/ucf_test.py'
OPTIONS_SOURCE = ROOT / 'VadCLIP/src/ucf_option.py'
