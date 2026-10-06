"""Frame metrics for full official-feature UCF runs; no model imports."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
