"""Original VadCLIP UCF temporal mAP for the full model evaluation.

Use the unmodified bundled evaluator and segment/class GT, with the upstream
ucf_test.py setting excludeNormal=False (290 videos and 14 classes).
Validation checks alignment without changing GT labels or evaluator behavior.
"""

import importlib.util

import numpy as np


CLASS_NAMES = ('Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
               'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
               'Shoplifting', 'Stealing', 'Vandalism')
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def compute_official_fine_grained(class_scores, videos, manifest, class_names,
                                 gt_segments, gt_labels, source_path):
    """Call the unmodified bundled evaluator with its original all-video protocol."""
    if (tuple(class_names) != CLASS_NAMES or len(class_scores) != 290
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
        probabilities = np.asarray(probabilities)
        if (probabilities.shape != (record['token_count'], len(CLASS_NAMES))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError('Invalid model class probabilities')
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
        unmapped.update(label for label in labels if label not in CLASS_NAMES)
        frames = np.repeat(probabilities, 16, axis=0)
        if len(frames) != record['feature_timeline_frames']:
            raise ValueError('Original fine-grained score timeline mismatch')
        predictions.append(frames)
    spec = importlib.util.spec_from_file_location('vadclip_original_ucf_detection_map', source_path)
    if spec is None or spec.loader is None:
        raise ValueError('Could not locate the original UCF fine-grained evaluator')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    values, thresholds = upstream.getDetectionMAP(predictions, gt_segments, gt_labels, excludeNormal=False)
    values = np.asarray(values, dtype=float)
    if (tuple(thresholds) != IOU_THRESHOLDS or values.shape != (5,)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 100)):
        raise ValueError('Invalid original fine-grained metric output')
    values = values / 100
    return {
        'score_branch': 'branch2', 'protocol': 'original_ucf_test_excludeNormal_false',
        'unit': 'fraction', 'evaluation_videos': 290, 'evaluation_classes': list(CLASS_NAMES),
        'map_by_iou': {f'{threshold:.1f}': float(value) for threshold, value in zip(thresholds, values)},
        'mean_map': float(values.mean()),
        'gt_source': 'Unmodified bundled gt_segment_ucf.npy and gt_label_ucf.npy; order validated against manifest',
        'unmapped_gt_labels_preserved': sorted(unmapped),
        'implementation': 'Unmodified VadCLIP/src/utils/ucf_detectionMAP.py, excludeNormal=False',
        'comparison_status': 'Original-code reproduction; agreement with published values still requires measurement',
    }
