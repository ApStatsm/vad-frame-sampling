"""D/A/E-only ver2 performance comparison, allowing different GPU assignments."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.paths import ROOT, sha256
from vadfs.runs import create_run_dir, write_json

METHODS = ('d_optimal', 'a_optimal', 'e_optimal')
BUDGETS = (1., .5, .25, .125)


def performance_identity_ver2(report):
    """Ignore GPU UUID/visibility, retain frozen inference and evaluation inputs."""
    inputs = report['input_sha256']
    env = report['environment']
    return {
        'dataset': report['dataset_key'], 'dataset_identity': report['dataset_identity_sha256'],
        'recipe': report['extraction_config'], 'model_options': report['model_options'],
        'prompts': report['prompts'], 'model_seed': report['experiment']['model_seed'],
        'selection_spec': report['experiment']['selection_spec'],
        'protocol': report['experiment']['protocol'], 'projection': report['projection']['q_sha256'],
        'package': report['package_sha256'], 'vendored': report['vendored_source_sha256'],
        'entrypoint': report['entrypoint_sha256'],
        'inputs': {k: inputs[k] for k in ('dataset_config', 'setup_manifest', 'clip_checkpoint',
                   'vadclip_checkpoint', 'gt', 'segment_gt', 'class_gt', 'fine_grained_source',
                   'official_list', 'tokenizer_vocabulary')},
        'software': {k: env.get(k) for k in ('python', 'torch', 'torch_cuda', 'torchvision',
                    'numpy', 'opencv', 'pillow', 'device', 'torch_threads', 'opencv_threads',
                    'cuda_allow_tf32', 'cudnn_allow_tf32', 'cudnn_benchmark', 'ffmpeg')},
        'primary': {k: report['metrics']['primary'][k] for k in ('branch', 'metric')},
    }


def video_identity_ver2(report):
    return [(v['video'], v['official_csv_index'], v['input_sha256'], v['reference_tokens'],
             v['evaluation_frames'], v['gt_offset']) for v in report['videos']]


def optimal_rows_ver2(reports):
    if not reports:
        raise ValueError('No summaries')
    groups = {}
    for report in reports:
        if (report.get('protocol_version') != 2 or report.get('mode') != 'full'
                or not report.get('success') or not report.get('finished_at')
                or report.get('candidate_space') != 'raw_window'
                or report.get('score_anchor') != 'window_start' or not report.get('metrics')):
            raise ValueError('Only completed ver2 raw-window full runs are supported')
        sampling = report['sampling']
        key = (sampling['method'], sampling['budget'])
        if key[0] not in METHODS or key[1] not in BUDGETS or sampling['seed'] != 0:
            raise ValueError('Only D/A/E seed0 at the four declared budgets are supported')
        if key in groups:
            raise ValueError('Duplicate method/budget; timing repeats are separate')
        videos = report['videos']
        if (report['metrics']['test_videos'] != len(videos)
                or len(videos) != {'ucf': 290, 'xd': 800}[report['dataset_key']]
                or len({v['video'] for v in videos}) != len(videos)):
            raise ValueError('Need all unique test videos')
        inverse = round(1 / key[1])
        if any(v['selected_tokens'] != (v['reference_tokens'] + inverse - 1) // inverse
               or v['encoder_input_images'] != v['selected_tokens'] for v in videos):
            raise ValueError('Per-video budget/encoder count mismatch')
        groups[key] = report
    if set(groups) != {(m,b) for m in METHODS for b in BUDGETS}:
        raise ValueError('Need all 12 D/A/E x budget runs; no Uniform/Random required')
    first = reports[0]
    identity, videos = performance_identity_ver2(first), video_identity_ver2(first)
    if any(performance_identity_ver2(r) != identity or video_identity_ver2(r) != videos for r in reports):
        raise ValueError('Model/data/selection/evaluation/software identity differs')
    rows = []
    for budget in BUDGETS:
        values = {m: groups[m,budget]['metrics']['primary']['value'] * 100 for m in METHODS}
        for method in METHODS:
            report = groups[method,budget]
            row = {'dataset': report['dataset_key'], 'method': method, 'budget': budget,
                   'primary_metric': report['metrics']['primary']['metric'],
                   'primary_percent': values[method],
                   'd_delta_pp': values[method] - values['d_optimal'],
                   'a_delta_pp': values[method] - values['a_optimal'],
                   'e_delta_pp': values[method] - values['e_optimal'],
                   'encoder_input_images': report['encoder_input_images'],
                   'end_to_end_seconds': report['end_to_end_seconds'],
                   'gpu_name': report['environment'].get('gpu_name'),
                   'gpu_uuid': report['environment'].get('gpu_uuid'),
                   'visible_devices': report['environment'].get('CUDA_VISIBLE_DEVICES'),
                   'timing_scope': 'single run; concurrent/historical conditions; no speedup claim'}
            for branch in ('branch1','branch2'):
                for metric in ('roc_auc','ap','ano_auc'):
                    value = report['metrics'][branch].get(metric)
                    row[f'{branch}_{metric}_percent'] = None if value is None else value * 100
            fine = report['metrics']['fine_grained']
            row['mean_map_percent'] = fine['mean_map'] * 100
            for iou,value in fine['map_by_iou'].items():
                row[f'map_iou_{iou}_percent'] = value * 100
            rows.append(row)
    return rows


def write_comparison_ver2(paths, output_root):
    paths = [Path(p).resolve() for p in paths]
    reports = [read_json(p) for p in paths]
    rows = optimal_rows_ver2(reports)
    _,folder = create_run_dir(f'{reports[0]["dataset_key"]}_optimal_only_ver2', 'full', output_root=output_root)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (folder/'comparison_optimal_ver2.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r['method'],r['budget']): r for r in rows}
    lines = ['# D/A/E optimal ver2 performance comparison', '',
             f'Dataset: {reports[0]["dataset"]}. Metric: {rows[0]["primary_metric"]} (%).', '',
             '| Method | 100% | 50% | 25% | 12.5% |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        cells = [f'{lookup[method,b]["primary_percent"]:.2f}' for b in BUDGETS]
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines += ['', 'No Uniform/Random runs are required. Deterministic sampler seed is 0.',
              'GPU assignment differences are allowed for performance comparison and recorded per run.',
              'Concurrent/shared-resource times are not final efficiency or speedup evidence.',
              'This remains an exploratory test-set comparison.']
    (folder/'evaluation_optimal_ver2.md').write_text('\n'.join(lines)+'\n')
    write_json(folder/'summary_optimal_ver2.json', {
        'protocol_version':2, 'scope':'optimal_only_performance', 'speedup_eligible':False,
        'frozen_identity':performance_identity_ver2(reports[0]), 'rows':rows,
        'environments':[{'run_id':r['run_id'],'environment':r['environment']} for r in reports],
        'inputs':[{'summary':str(p),'sha256':sha256(p)} for p in paths]})
    print(f'Comparison: {folder / "comparison_optimal_ver2.csv"}', flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/analyses')
    args = parser.parse_args()
    write_comparison_ver2(args.summaries, args.output_root)


if __name__ == '__main__':
    main()
