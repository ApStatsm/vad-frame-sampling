"""Evaluate restored reference-grid scores; model/sampler output lengths stay separate."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .datasets import get_dataset
from .paths import project_path


def evaluate(dataset_config, manifest, records, branch1, branch2, class_tokens):
    adapter = get_dataset(dataset_config['key'])
    gt = np.load(project_path(dataset_config['frame_gt']), allow_pickle=False)
    frames1, frames2 = np.concatenate(branch1), np.concatenate(branch2)
    if (frames1.shape != gt.shape or frames2.shape != gt.shape
            or not np.isfinite(frames1).all() or not np.isfinite(frames2).all()):
        raise ValueError('Predictions must cover the full fixed official evaluation timeline')
    if adapter.SPEC.key == 'ucf':
        from .metrics import compute_official_fine_grained, compute_ucf_metrics

        # The old evaluator accepts dense reference tokens, not sparse sampler K.
        # Preserve its complete 290-video/order/GT checks and 16-fold convention.
        dense_records = [{**record, 'token_count': record['reference_tokens'],
                          'feature_timeline_frames': record['reference_tokens'] * 16}
                         for record in records]
        metrics = compute_ucf_metrics(gt, frames1, frames2, dense_records, manifest)
        metrics['fine_grained'] = compute_official_fine_grained(
            class_tokens, dense_records, manifest, adapter.SPEC.prompts,
            np.load(project_path(dataset_config['segment_gt']), allow_pickle=True),
            np.load(project_path(dataset_config['class_gt']), allow_pickle=True),
            project_path(dataset_config['fine_grained_source']))
    else:
        metrics = {'evaluation_frames': len(gt), 'test_videos': len(records), 'includes_normal': True}
        for name, scores in (('branch1', frames1), ('branch2', frames2)):
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError('Probability outside [0,1]')
            metrics[name] = {'roc_auc': float(roc_auc_score(gt, scores)),
                             'ap': float(average_precision_score(gt, scores))}
        # XD primary AP is supported; temporal mAP awaits separate protocol validation.
        metrics['fine_grained'] = {'status': 'not_implemented_or_validated_for_xd'}
    metrics['primary'] = {'branch': adapter.SPEC.primary_branch, 'metric': adapter.SPEC.primary_metric,
                          'value': metrics[adapter.SPEC.primary_branch][adapter.SPEC.primary_metric]}
    return metrics, frames1, frames2
