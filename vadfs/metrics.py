"""Evaluation metrics from saved or fresh scores; no model imports.

Fine-grained evaluation uses the bundled VadCLIP evaluators and original
segment/class ground truth with excludeNormal=False.
"""

import importlib.util

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .datasets import xd
from .datasets.base import validate_manifest
from .timeline import to_frames
from .ucf import LABELS


IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def compute_ucf_metrics(gt, branch1, branch2, videos, manifest):
    """Pool every frame of anomaly videos for Ano-AUC, including normal intervals."""
    gt = np.asarray(gt)
    scores = [np.asarray(branch1), np.asarray(branch2)]
    if gt.ndim != 1 or not np.isin(gt, [0, 1]).all() or np.unique(gt).size != 2:
        raise ValueError('GT must be a binary frame array containing both classes')
    for score in scores:
        if score.shape != gt.shape or not np.isfinite(score).all():
            raise ValueError('Frame scores must be finite and have the GT shape')
        if np.any(score < 0) or np.any(score > 1):
            raise ValueError('Frame scores must be in [0, 1]')

    declared = manifest['videos']
    if len(videos) != 290 or len(declared) != 290:
        raise ValueError('Ano-AUC requires the complete 290-video UCF test run')
    by_name = {record['filename']: record for record in declared}
    if len(by_name) != len(declared):
        raise ValueError('Manifest contains duplicate video names')
    anomaly_mask = np.zeros(len(gt), dtype=bool)
    seen = set()
    offset = 0
    anomaly_count = 0
    for index, record in enumerate(videos):
        name = record['video']
        if name in seen or name not in by_name:
            raise ValueError(f'Duplicate or unknown run video: {name}')
        seen.add(name)
        declared_video = by_name[name]
        if (record['official_csv_index'] != index
                or declared_video['official_csv_index'] != index
                or record.get('prefix_only') is not False
                or record.get('case') != 'full_video'):
            raise ValueError(f'Run order or full-video scope mismatch: {name}')
        count = record['feature_timeline_frames']
        if (not isinstance(count, int) or count <= 0
                or count != 16 * record['token_count']
                or record['gt_offset'] != offset or offset + count > len(gt)):
            raise ValueError(f'Invalid or non-contiguous frame segment: {name}')
        is_anomaly = declared_video['is_anomaly']
        if (not isinstance(is_anomaly, bool)
                or is_anomaly != (declared_video['category'] != 'Normal')):
            raise ValueError(f'Invalid manifest anomaly classification: {name}')
        if is_anomaly:
            anomaly_mask[offset:offset + count] = True
            anomaly_count += 1
        elif np.any(gt[offset:offset + count] != 0):
            raise ValueError(f'Normal-only video contains positive GT: {name}')
        offset += count
    if offset != len(gt) or anomaly_count != 140:
        raise ValueError('Expected complete frame coverage and 140 anomaly videos')
    anomaly_gt = gt[anomaly_mask]
    if np.unique(anomaly_gt).size != 2:
        raise ValueError('Ano-AUC requires both normal and anomaly frames in anomaly videos')
    metrics = {
        'evaluation_frames': len(gt), 'test_videos': len(videos), 'includes_normal': True,
        'anomaly_videos': anomaly_count,
        'anomaly_evaluation_frames': int(anomaly_mask.sum()),
        'ano_auc_scope': 'Pooled frames of manifest.is_anomaly videos, including normal intervals',
    }
    for name, score in zip(('branch1', 'branch2'), scores):
        metrics[name] = {
            'roc_auc': float(roc_auc_score(gt, score)),
            'ap': float(average_precision_score(gt, score)),
            'ano_auc': float(roc_auc_score(anomaly_gt, score[anomaly_mask])),
        }
    return metrics


def compute_official_fine_grained(class_scores, videos, manifest, class_names,
                                 gt_segments, gt_labels, source_path):
    """Call the unmodified bundled evaluator with its original all-video protocol."""
    if (tuple(class_names) != LABELS or len(class_scores) != 290
            or len(videos) != 290 or len(gt_segments) != 290 or len(gt_labels) != 290):
        raise ValueError('Original UCF fine-grained inputs must cover 290 videos/14 classes')
    by_name = {video['filename']: video for video in manifest['videos']}
    if len(by_name) != 290:
        raise ValueError('Expected 290 unique manifest videos')
    predictions = []
    unmapped = set()
    for index, (record, probabilities) in enumerate(zip(videos, class_scores)):
        video = by_name[record['video']]
        if (record['official_csv_index'] != index or video['official_csv_index'] != index
                or record.get('prefix_only') is not False or record.get('case') != 'full_video'):
            raise ValueError('Original fine-grained video order/scope mismatch')
        probabilities = _class_probabilities(probabilities, record['token_count'], len(LABELS))
        segments = [[int(start), int(end)] for start, end in gt_segments[index]]
        labels = [str(label) for label in gt_labels[index]]
        expected = video['intervals_raw'] if video['is_anomaly'] else [[0, record['feature_timeline_frames']]]
        if segments != expected or len(labels) != len(segments):
            raise ValueError(f'Bundled segment GT does not match run/manifest: {record["video"]}')
        if video['is_anomaly']:
            if any(label != video['category'] for label in labels):
                raise ValueError(f'Bundled class GT differs from manifest: {record["video"]}')
        elif any(label not in ('Normal', 'A') for label in labels):
            # The bundled GT generator uses 'A' for normal videos; do not remap it.
            raise ValueError('Unexpected bundled normal GT label')
        unmapped.update(label for label in labels if label not in LABELS)
        frames = to_frames(probabilities)
        if len(frames) != record['feature_timeline_frames']:
            raise ValueError('Original fine-grained score timeline mismatch')
        predictions.append(frames)
    return _official_detection_map('ucf', predictions, LABELS, gt_segments, gt_labels,
                                   source_path, unmapped, 'gt_segment_ucf.npy and gt_label_ucf.npy')


def compute_xd_fine_grained(class_scores, videos, manifest, gt_segments, gt_labels, source_path):
    """Evaluate the complete official XD split in CSV order using seven prompts."""
    declared = validate_manifest(manifest, xd.SPEC)
    video_count = len(declared)
    if any(len(items) != video_count for items in (class_scores, videos, gt_segments, gt_labels)):
        raise ValueError('Original XD fine-grained inputs must cover the complete 800-video test split')

    predictions = []
    offset = 0
    anomaly_labels = {'B1', 'B2', 'B4', 'B5', 'B6', 'G'}
    for index, (record, video, probabilities) in enumerate(zip(videos, declared, class_scores)):
        name = video['filename']
        tokens = video['reference_tokens']
        frames = video['evaluation_frames']
        if (record['video'] != name or record['official_csv_index'] != index
                or record.get('prefix_only') is not False or record.get('case') != 'full_video'
                or record['reference_tokens'] != tokens or record['evaluation_frames'] != frames
                or record['gt_offset'] != offset or video['gt_offset'] != offset):
            raise ValueError(f'XD fine-grained video order, scope or timeline mismatch: {name}')
        segments = [[int(start), int(end)] for start, end in gt_segments[index]]
        labels = [str(label) for label in gt_labels[index]]
        if len(labels) != len(segments) or any(start < 0 or end <= start for start, end in segments):
            raise ValueError(f'Invalid bundled XD segment GT: {name}')
        if video['is_anomaly']:
            if not labels or not set(labels) <= anomaly_labels:
                raise ValueError(f'Invalid bundled XD anomaly labels: {name}')
        elif segments != [[0, frames]] or labels != ['A']:
            raise ValueError(f'Bundled XD normal GT differs from the evaluation timeline: {name}')
        predictions.append(to_frames(_class_probabilities(probabilities, tokens, len(xd.PROMPTS))))
        offset += frames

    return _official_detection_map('xd', predictions, xd.PROMPTS, gt_segments, gt_labels,
                                   source_path, (), 'gt_segment.npy and gt_label.npy')


def _class_probabilities(scores, token_count, class_count):
    scores = np.asarray(scores)
    if (scores.shape != (token_count, class_count) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)
            or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Invalid model class probabilities')
    return scores


def _official_detection_map(dataset, predictions, class_names, gt_segments, gt_labels,
                            source_path, unmapped, gt_names):
    spec = importlib.util.spec_from_file_location(f'vadclip_original_{dataset}_detection_map', source_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Could not locate the original {dataset} fine-grained evaluator')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    values, thresholds = upstream.getDetectionMAP(predictions, gt_segments, gt_labels, excludeNormal=False)
    values = np.asarray(values, dtype=float)
    if (tuple(thresholds) != IOU_THRESHOLDS or values.shape != (5,)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 100)):
        raise ValueError('Invalid original fine-grained metric output')
    values = values / 100
    return {
        'score_branch': 'branch2', 'protocol': f'original_{dataset}_test_excludeNormal_false',
        'unit': 'fraction', 'evaluation_videos': len(predictions), 'evaluation_classes': list(class_names),
        'map_by_iou': {f'{threshold:.1f}': float(value) for threshold, value in zip(thresholds, values)},
        'mean_map': float(values.mean()),
        'gt_source': f'Unmodified bundled {gt_names}; order validated against manifest',
        'unmapped_gt_labels_preserved': sorted(unmapped),
        'implementation': f'Unmodified VadCLIP/src/utils/{dataset}_detectionMAP.py, excludeNormal=False',
        'comparison_status': 'Original-code reproduction; agreement with published values still requires measurement',
    }


def metric_comparison(raw, official):
    """Percentage-point differences of a run against the official-feature reference."""
    differences = {}
    for branch in ('branch1', 'branch2'):
        differences[branch] = {key: 100 * (raw[branch][key] - official[branch][key])
                               for key in ('roc_auc', 'ap', 'ano_auc')}
    differences['fine_grained'] = {
        'map_by_iou': {key: 100 * (value - official['fine_grained']['map_by_iou'][key])
                       for key, value in raw['fine_grained']['map_by_iou'].items()},
        'mean_map': 100 * (raw['fine_grained']['mean_map'] - official['fine_grained']['mean_map'])}
    return {'unit': 'percentage_points', 'raw_minus_official': differences,
            'reproduction_status': 'Not judged automatically: compare against the tolerance fixed before the run'}
