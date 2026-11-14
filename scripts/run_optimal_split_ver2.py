"""Resume D/A/E-only full runs across GPUs; never launches Uniform/Random."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze.compare_optimal_ver2 import METHODS, write_comparison_ver2
from vadfs.config import load_experiment, read_json, resource_path
from vadfs.paths import package_hashes, project_path, sha256
from vadfs.runs import create_run_dir, now, write_json

BUDGET_LABELS = ('100','50','25','12p5')


def expected_context_ver2(dataset, local, manifest_path):
    first_config = ROOT/f'configs/experiments/{dataset}_d_optimal_100_ver2.json'
    _,data,resources,_,data_path = load_experiment(first_config,local)
    manifest = read_json(manifest_path)
    if data['key'] != dataset or manifest.get('dataset_key', dataset) != dataset:
        raise ValueError('Manifest/dataset mismatch')
    if len(manifest['videos']) != {'ucf':290,'xd':800}[dataset]:
        raise ValueError('Full manifest required')
    inputs = {'dataset_config':sha256(data_path),'local_config':sha256(local),
              'setup_manifest':sha256(manifest_path),'gt':sha256(project_path(data['frame_gt']))}
    for key in ('clip_checkpoint','vadclip_checkpoint'):
        inputs[key] = sha256(resource_path(resources,key))
    for key in ('segment_gt','class_gt','fine_grained_source','official_list'):
        inputs[key] = sha256(project_path(data[key]))
    inputs['tokenizer_vocabulary'] = sha256(ROOT/'VadCLIP/src/clip/bpe_simple_vocab_16e6.txt.gz')
    return {'dataset':dataset,'manifest':manifest,'inputs':inputs,'package':package_hashes(),
            'entrypoint':sha256(ROOT/'scripts/run_experiment_ver2.py'),
            'vendored':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted((ROOT/'VadCLIP/src').rglob('*.py'))},
            'output_root':resource_path(resources,'output_root')}


def reusable_ver2(report, config, config_hash, context):
    """Reuse completed inference only; does not reuse raw speedup evidence."""
    if (not report.get('success') or report.get('mode') != 'full' or not report.get('finished_at')
            or report.get('protocol_version') != 2 or report.get('dataset_key') != context['dataset']
            or report.get('candidate_space') != 'raw_window' or report.get('score_anchor') != 'window_start'
            or report.get('experiment') != config or report.get('package_sha256') != context['package']
            or report.get('vendored_source_sha256') != context['vendored']
            or report.get('entrypoint_sha256') != context['entrypoint']
            or report.get('dataset_identity_sha256') != context['manifest']['dataset_identity_sha256']
            or not report.get('metrics')):
        return False
    inputs = report.get('input_sha256',{})
    if inputs.get('experiment_config') != config_hash or any(inputs.get(k)!=v for k,v in context['inputs'].items()):
        return False
    saved, expected = report.get('videos',[]), context['manifest']['videos']
    if len(saved)!=len(expected) or report['metrics'].get('test_videos')!=len(expected):
        return False
    q = round(1/config['sampling']['budget'])
    for old,video in zip(saved,expected):
        if (old.get('video')!=video['filename'] or old.get('official_csv_index')!=video['official_csv_index']
                or old.get('input_sha256')!=video['sha256'] or old.get('reference_tokens')!=video['reference_tokens']
                or old.get('evaluation_frames')!=video['evaluation_frames'] or old.get('gt_offset')!=video['gt_offset']
                or old.get('selected_tokens')!=(video['reference_tokens']+q-1)//q
                or old.get('encoder_input_images')!=old.get('selected_tokens')):
            return False
    return True


def find_existing_ver2(jobs, roots, context):
    """Choose latest compatible success per case. Partial/failed summaries are ignored."""
    found = {}
    catalog = []
    for root in roots:
        for path in sorted(root.glob(f'{context["dataset"]}_*_ver2_full_*/summary.json')):
            try:
                report = read_json(path)
            except (OSError,ValueError):
                continue
            sampling = report.get('sampling',{})
            if sampling.get('method') in METHODS and report.get('success'):
                catalog.append((path,report))
    for job in jobs:
        config = read_json(job['config'])
        digest = sha256(job['config'])
        matches = [(p,r) for p,r in catalog if reusable_ver2(r,config,digest,context)]
        if matches:
            path,report = max(matches,key=lambda item:item[1]['finished_at'])
            found[job['key']] = {'summary':str(path.resolve()),'sha256':sha256(path),
                                 'reused':True,'gpu':report['environment'].get('CUDA_VISIBLE_DEVICES')}
    return found


def existing_processes_ver2(dataset):
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not any(Path(a).name in ('run_optimal_ver2.py','run_experiment_ver2.py','run_optimal_split_ver2.py') for a in args):
                continue
            if (proc/'cwd').resolve()!=ROOT:
                continue
            explicit_dataset = args[args.index('--dataset')+1] if '--dataset' in args else None
            config = args[args.index('--config')+1] if '--config' in args else ''
            if explicit_dataset==dataset or Path(config).name.startswith(dataset+'_'):
                result.append({'pid':int(proc.name),'command':' '.join(args)})
        except (OSError,ValueError,IndexError):
            continue
    return result


def stop_children_ver2(active):
    for item in active.values():
        process = item['process']
        if process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
    for item in active.values():
        process = item['process']
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        item['handle'].close()


def report_from_log_ver2(log):
    summary = None
    with log.open(encoding='utf-8',errors='replace') as handle:
        for line in handle:
            if line.startswith('Report: '):
                summary = Path(line.removeprefix('Report: ').strip())
    if summary is None or not summary.is_file():
        raise RuntimeError(f'No saved report in {log}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('ucf','xd'),required=True)
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--setup-manifest',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',default=['0','1'])
    parser.add_argument('--mode',choices=('full',),default='full')
    parser.add_argument('--reuse-root',type=Path,action='append',default=[])
    parser.add_argument('--plan-only',action='store_true',help='User-run pending/reuse plan; no inference')
    args = parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or any(not g.isdigit() for g in args.gpus):
        parser.error('Use unique physical GPU indices, e.g. --gpus 0 1')
    local,manifest = project_path(args.local),project_path(args.setup_manifest)
    # This imports no project model, torch or decoder; children do inference.
    context = expected_context_ver2(args.dataset,local,manifest)
    jobs = [{'key':f'{method}_{label}','method':method,'budget':label,
             'config':ROOT/f'configs/experiments/{args.dataset}_{method}_{label}_ver2.json'}
            for label in BUDGET_LABELS for method in METHODS]
    for job in jobs:
        config = read_json(job['config'])
        if config['sampling']['method'] not in METHODS or config['sampling']['seed']!=0:
            raise ValueError('Only deterministic D/A/E configs are allowed')
    analyses = ROOT/'outputs/analyses'
    analyses.mkdir(parents=True,exist_ok=True)
    lock = (analyses/f'.{args.dataset}_optimal_split_ver2.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another split sweep is already active') from None
    roots = list(dict.fromkeys([context['output_root'],*[project_path(p) for p in args.reuse_root]]))
    found = find_existing_ver2(jobs,roots,context)
    pending = [j for j in jobs if j['key'] not in found]
    print(f'{args.dataset}: {len(found)}/12 compatible completed cases reused; {len(pending)} pending.',flush=True)
    for job in pending:
        print(f'Pending: {job["key"]}',flush=True)
    running = existing_processes_ver2(args.dataset)
    if running:
        for item in running:
            print(f'Existing PID {item["pid"]}: {item["command"]}',flush=True)
        if not args.plan_only:
            raise RuntimeError('Stop the old dataset sweep first; no new jobs were launched')
    if args.plan_only:
        return 0
    _,folder = create_run_dir(f'{args.dataset}_optimal_split_ver2','full',('logs',),analyses)
    state = {'protocol_version':2,'dataset':args.dataset,'complete':False,'started_at':now().isoformat(),
             'gpus':args.gpus,'scope':'D/A/E performance only; concurrent timing not benchmark',
             'scheduler_sha256':sha256(Path(__file__)),'runs':found,'active':{}}
    active = {}
    def save():
        state['active'] = {gpu:{'job':item['job']['key'],'pid':item['process'].pid,'log':str(item['log'])}
                           for gpu,item in active.items()}
        write_json(folder/'sweep_optimal_ver2.json',state)
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        while pending or active:
            for gpu in args.gpus:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                log = folder/'logs'/f'{job["key"]}_gpu{gpu}_ver2.log'
                handle = log.open('w',encoding='utf-8')
                command = [sys.executable,'-u',str(ROOT/'scripts/run_experiment_ver2.py'),
                           '--config',str(job['config']),'--local',str(local),'--setup-manifest',str(manifest),
                           '--device','cuda','--mode','full','--seed','0']
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu
                try:
                    process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    handle.close()
                    raise
                active[gpu] = {'process':process,'handle':handle,'log':log,'job':job}
                print(f'Start GPU {gpu}: {job["key"]}; log={log}',flush=True)
                save()
            for gpu,item in list(active.items()):
                status = item['process'].poll()
                if status is None:
                    continue
                item['handle'].close()
                if status!=0:
                    raise RuntimeError(f'GPU {gpu} {item["job"]["key"]} failed ({status}); inspect {item["log"]}')
                summary = report_from_log_ver2(item['log'])
                report = read_json(summary)
                config = read_json(item['job']['config'])
                if not reusable_ver2(report,config,sha256(item['job']['config']),context):
                    raise RuntimeError(f'Finished report failed identity/completeness checks: {summary}')
                state['runs'][item['job']['key']] = {'summary':str(summary),'sha256':sha256(summary),'reused':False,'gpu':gpu,'log':str(item['log'])}
                del active[gpu]
                print(f'Done {len(state["runs"])}/12: {item["job"]["key"]} on GPU {gpu}',flush=True)
                save()
            if active:
                time.sleep(2)
        paths = [state['runs'][job['key']]['summary'] for job in jobs]
        comparison = write_comparison_ver2(paths,analyses)
        state.update(complete=True,comparison_folder=str(comparison))
        return 0
    except KeyboardInterrupt:
        state['error'] = 'Interrupted; successful cases remain reusable'
        print(state['error'],flush=True)
        return 130
    except Exception as error:
        state['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stop_children_ver2(active)
        active.clear()
        state['finished_at'] = now().isoformat()
        save()
        print(f'Sweep: {folder / "sweep_optimal_ver2.json"}',flush=True)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
