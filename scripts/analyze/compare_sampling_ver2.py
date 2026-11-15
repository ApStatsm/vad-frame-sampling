"""Compare complete ver2 raw-window runs; reject cached/legacy/mixed inputs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vadfs.config import read_json
from vadfs.paths import ROOT, sha256
from vadfs.runs import create_run_dir, write_json
from vadfs.samplers import budget_count


def comparison_rows_ver2(reports):
    if not reports:
        raise ValueError('No reports')
    first = reports[0]
    groups = defaultdict(list)
    seen = set()
    for report in reports:
        if (report.get('protocol_version')!=2 or report.get('mode')!='full' or not report.get('success')
                or report.get('candidate_space')!='raw_window' or report.get('score_anchor')!='window_start'
                or report.get('comparison_signature')!=first.get('comparison_signature')
                or not report.get('comparison_signature') or report.get('dataset_key')!=first.get('dataset_key')
                or not report.get('metrics') or report['metrics']['test_videos']!=len(report['videos'])):
            raise ValueError('Need successful same-signature ver2 full runs')
        identity = [(v['video'],v['input_sha256'],v['reference_tokens'],v['evaluation_frames']) for v in report['videos']]
        if identity != [(v['video'],v['input_sha256'],v['reference_tokens'],v['evaluation_frames']) for v in first['videos']]:
            raise ValueError('Video identity/order/timeline differs')
        sampling = report['sampling']
        if sampling['method'] not in ('uniform','random_window','d_optimal','a_optimal','e_optimal'):
            raise ValueError('Unknown ver2 sampler')
        if sampling['budget'] not in (1.,.5,.25,.125):
            raise ValueError('Unknown ver2 budget')
        if any(v['selected_tokens'] != budget_count(v['reference_tokens'],sampling['budget']) for v in report['videos']):
            raise ValueError('Invalid per-video budget')
        key = (sampling['method'],sampling['budget'],sampling['seed'])
        if key in seen:
            raise ValueError('Duplicate method/budget/seed; time repeats must be separate')
        seen.add(key)
        groups[key[:2]].append(report)
    rows = []
    for (method,budget),group in sorted(groups.items()):
        seeds = sorted(r['sampling']['seed'] for r in group)
        if seeds != (list(range(5)) if method=='random_window' else [0]):
            raise ValueError('Expected all five Random seeds or one deterministic seed0')
        uniform = groups.get(('uniform',budget))
        random_group = groups.get(('random_window',budget))
        if not uniform or not random_group or len(random_group)!=5:
            raise ValueError('Uniform and five Random-window seeds required at every budget')
        ks = [[v['selected_tokens'] for v in r['videos']] for r in group+uniform+random_group]
        if any(k!=ks[0] for k in ks):
            raise ValueError('Per-video K differs')
        values = [r['metrics']['primary']['value']*100 for r in group]
        mean = statistics.mean(values)
        row = {'dataset':first['dataset_key'],'method':method,'budget':budget,
               'primary_metric':first['metrics']['primary']['metric'],'primary_mean_percent':mean,
               'primary_std_pp':statistics.stdev(values) if len(values)>1 else None,'n_seeds':len(group),
               'uniform_delta_pp':mean-uniform[0]['metrics']['primary']['value']*100,
               'random_delta_pp':mean-statistics.mean(r['metrics']['primary']['value']*100 for r in random_group),
               'encoder_input_images':sum(ks[0]),
               'end_to_end_mean_seconds':statistics.mean(r['end_to_end_seconds'] for r in group),
               'timing_note':'one execution per seed; not repeated timing evidence'}
        for branch in ('branch1','branch2'):
            for metric in ('roc_auc','ap','ano_auc'):
                values = [r['metrics'][branch].get(metric) for r in group]
                if all(v is not None for v in values):
                    row[f'{branch}_{metric}_percent'] = statistics.mean(values)*100
        row['mean_map_percent'] = statistics.mean(r['metrics']['fine_grained']['mean_map'] for r in group)*100
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries',nargs='+',type=Path)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/analyses')
    args = parser.parse_args()
    reports = [read_json(p) for p in args.summaries]
    rows = comparison_rows_ver2(reports)
    _,folder = create_run_dir(f'{reports[0]["dataset_key"]}_comparison_ver2','full',output_root=args.output_root)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (folder/'comparison_ver2.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    write_json(folder/'summary_ver2.json',{'protocol_version':2,'rows':rows,
               'comparison_signature':reports[0]['comparison_signature'],
               'inputs':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in args.summaries]})
    print(f'Comparison: {folder / "comparison_ver2.csv"}',flush=True)


if __name__=='__main__':
    main()
