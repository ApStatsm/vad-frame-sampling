"""Stage A: Full -> layout-restored Full scores -> actual ver3 sparse scores."""

import argparse
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vadfs.context_ver4 import SPEC_VER4, load_ver3, restore_full_layout, score_arrays
from vadfs.paths import sha256
from vadfs.runs import create_run_dir, record_provenance, write_json
from vadfs.timeline import restore_tokens, to_frames
from scripts.analyze.compare_context_ver3 import fpr_at_recall, save_csv


def run(sweep, output_root):
    _, output = create_run_dir('xd_decomposition_ver4', 'full', output_root=output_root)
    status = {'success': False, 'error': None, 'protocol': SPEC_VER4, 'rows': [],
              'input_sha256': {}, 'command': sys.argv,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'sklearn': sklearn.__version__}}
    try:
        record_provenance(status)
        groups, manifest, gt, provenance = load_ver3(sweep)
        status['inputs'] = provenance
        status['source_sha256'] = {'entrypoint': sha256(Path(__file__)),
                                 'helper': sha256(ROOT / 'vadfs/context_ver4.py'),
                                 'timeline': sha256(ROOT / 'vadfs/timeline.py'),
                                 'metric_helper': sha256(ROOT / 'scripts/analyze/compare_context_ver3.py')}
        ref_path, ref = groups[('reference', 0)]
        with np.load(ref_path.parent / 'frame_scores.npz', allow_pickle=False) as z:
            full = {b: z[f'branch{b}'].copy() for b in (1, 2)}
        full_ap, targets = {}, {}
        for b in (1, 2):
            if full[b].shape != gt.shape or not np.isfinite(full[b]).all():
                raise ValueError('Invalid Full aggregate scores')
            full_ap[b] = float(average_precision_score(gt, full[b]))
            if abs(full_ap[b] - ref['metrics'][f'branch{b}']['ap']) > 1e-10:
                raise ValueError('Full AP does not reproduce')
            targets[b] = float(np.mean(full[b][gt == 1] >= 0.5))
        score_inputs = {}
        for (method, seed), (path, report) in sorted(groups.items()):
            restored = {b: [] for b in (1, 2)}
            actual = {b: [] for b in (1, 2)}
            for video in manifest['videos']:
                frames, grid, values, source = score_arrays(path, report, video)
                rf, rg, rv, ref_source = score_arrays(ref_path, ref, video)
                score_inputs[source['file']] = source['sha256']
                score_inputs[ref_source['file']] = ref_source['sha256']
                if not np.array_equal(rf, grid) or not np.array_equal(rg, grid):
                    raise ValueError('Full/sparse coordinate alignment failed')
                start, stop = video['gt_offset'], video['gt_offset'] + video['evaluation_frames']
                for b in (1, 2):
                    if not np.array_equal(to_frames(rv[b]), full[b][start:stop]):
                        raise ValueError('Full direct and aggregate scores disagree')
                    restored[b].append(restore_full_layout(rv[b], frames, grid))
                    actual[b].append(to_frames(restore_tokens(values[b], frames, grid)))
            with np.load(path.parent / 'frame_scores.npz', allow_pickle=False) as z:
                score_inputs[str(path.parent / 'frame_scores.npz')] = sha256(path.parent / 'frame_scores.npz')
                for b in (1, 2):
                    prediction = np.concatenate(actual[b])
                    if not np.array_equal(prediction, z[f'branch{b}']):
                        raise ValueError('Actual direct/restored aggregate mismatch')
                    sparse_ap = float(average_precision_score(gt, prediction))
                    if abs(sparse_ap - report['metrics'][f'branch{b}']['ap']) > 1e-10:
                        raise ValueError('Sparse AP does not reproduce')
                    counterfactual = np.concatenate(restored[b])
                    restored_ap = float(average_precision_score(gt, counterfactual))
                    rfpr, rfpr_actual, rrecall = fpr_at_recall(gt, counterfactual, targets[b])
                    afpr, afpr_actual, arecall = fpr_at_recall(gt, prediction, targets[b])
                    status['rows'].append({
                        'method': method, 'seed': seed, 'branch': b,
                        'full_ap_percent': 100 * full_ap[b],
                        'layout_restored_full_ap_percent': 100 * restored_ap,
                        'actual_ap_percent': 100 * sparse_ap,
                        'full_to_layout_pp': 100 * (restored_ap - full_ap[b]),
                        'layout_to_actual_pp': 100 * (sparse_ap - restored_ap),
                        'target_recall_percent': 100 * targets[b],
                        'layout_fpr_interpolated_percent': 100 * rfpr,
                        'actual_fpr_interpolated_percent': 100 * afpr,
                        'layout_fpr_attainable_percent': 100 * rfpr_actual,
                        'actual_fpr_attainable_percent': 100 * afpr_actual,
                        'layout_attainable_recall_percent': 100 * rrecall,
                        'actual_attainable_recall_percent': 100 * arecall})
            print(f'Decomposed {method} seed{seed}', flush=True)
        aggregate = []
        for method in sorted({r['method'] for r in status['rows']}):
            for b in (1, 2):
                subset = [r for r in status['rows'] if r['method'] == method and r['branch'] == b]
                row = {'method': method, 'branch': b, 'n_runs': len(subset)}
                for field in subset[0]:
                    if field in ('method', 'branch', 'seed'):
                        continue
                    values = [r[field] for r in subset]
                    row[field + '_mean'] = statistics.mean(values)
                    row[field + '_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
                aggregate.append(row)
        status['score_input_sha256'] = score_inputs
        status['aggregates'] = aggregate
        save_csv(output / 'decomposition_runs_ver4.csv', status['rows'])
        save_csv(output / 'decomposition_comparison_ver4.csv', aggregate)
        status['success'] = True
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(output / 'summary_ver4.json', status)
        print(f'Report: {output / "summary_ver4.json"}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-ver3', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/analyses')
    args = parser.parse_args()
    run(args.sweep_ver3, args.output_root)


if __name__ == '__main__':
    main()
