"""Sequential ver2 raw-window sweep: 36 runs per dataset, no cached features."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_random_windows import run_and_get_summary
from vadfs.paths import sha256
from vadfs.runs import create_run_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('ucf','xd'),required=True)
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--setup-manifest',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda','mps'),default='cuda')
    parser.add_argument('--mode',choices=('smoke','full'),default='smoke')
    parser.add_argument('--methods',nargs='+',choices=('uniform','random_window','d_optimal','a_optimal','e_optimal'),
                        default=['uniform','random_window','d_optimal','a_optimal','e_optimal'])
    parser.add_argument('--budgets',nargs='+',choices=('100','50','25','12p5'),default=['100','50','25','12p5'])
    args = parser.parse_args()
    if len(set(args.methods))!=len(args.methods) or len(set(args.budgets))!=len(args.budgets):
        parser.error('Duplicate methods/budgets')
    _,folder = create_run_dir(f'{args.dataset}_sweep_ver2',args.mode,output_root=ROOT/'outputs/analyses')
    sweep = {'protocol_version':2,'complete':False,'mode':args.mode,'runs':[]}
    try:
        for method in args.methods:
            for budget in args.budgets:
                for seed in (range(5) if method=='random_window' else (0,)):
                    command = [sys.executable,'-u',str(ROOT/'scripts/run_experiment_ver2.py'),
                               '--config',str(ROOT/f'configs/experiments/{args.dataset}_{method}_{budget}_ver2.json'),
                               '--local',str(args.local.resolve()),'--setup-manifest',str(args.setup_manifest.resolve()),
                               '--device',args.device,'--mode',args.mode,'--seed',str(seed)]
                    summary = run_and_get_summary(command)
                    sweep['runs'].append({'method':method,'budget':budget,'seed':seed,'summary':summary,'sha256':sha256(Path(summary))})
                    write_json(folder/'sweep_ver2.json',sweep)
        if args.mode=='full' and {'uniform','random_window'} <= set(args.methods):
            subprocess.run([sys.executable,str(ROOT/'scripts/analyze/compare_sampling_ver2.py'),
                            *[r['summary'] for r in sweep['runs']]],cwd=ROOT,check=True)
        sweep['complete'] = True
    finally:
        write_json(folder/'sweep_ver2.json',sweep)
        print(f'Sweep: {folder / "sweep_ver2.json"}',flush=True)


if __name__=='__main__':
    main()
