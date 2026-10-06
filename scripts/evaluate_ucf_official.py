"""User-run smoke/full VadCLIP evaluation on official UCF test features.

Retains upstream process_split, declared window lengths, mask behavior, prompts,
two score branches and 16-fold repetition. Smoke prefixes are execution checks,
not benchmark results. Cached-feature timings are not raw-video efficiency.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
LABELS = ('Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
          'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
          'Shoplifting', 'Stealing', 'Vandalism')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def synchronize(torch, device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def upstream_lengths(torch, token_count, window):
    """Mirror ucf_test.py exactly, including integer-multiple window behavior."""
    remaining = token_count
    lengths = torch.zeros(int(remaining / window) + 1)
    for index in range(len(lengths)):
        if index == 0 and remaining < window:
            lengths[index] = remaining
        elif index == 0 and remaining > window:
            lengths[index] = window
            remaining -= window
        elif remaining > window:
            lengths[index] = window
            remaining -= window
        else:
            lengths[index] = remaining
    return lengths.to(torch.int64)


def infer_feature(np, torch, model, feature, window, prompts, device):
    from utils.tools import process_split, get_batch_mask

    split, token_count = process_split(feature, window)
    visual = torch.tensor(split)
    if token_count < window:
        visual = visual.unsqueeze(0)
    lengths = upstream_lengths(torch, token_count, window)
    if visual.ndim != 3 or visual.shape[0] != len(lengths) or visual.shape[1] != window:
        raise ValueError('Feature split and upstream declared lengths are inconsistent')
    mask = get_batch_mask(lengths, window).to(device)
    visual = visual.to(device)
    synchronize(torch, device)
    started = time.perf_counter()
    with torch.no_grad():
        _, logits1, logits2 = model(visual, mask, prompts, lengths)
        score1 = torch.sigmoid(logits1.reshape(-1, logits1.shape[-1])[:token_count].squeeze(-1))
        score2 = 1 - logits2.reshape(-1, logits2.shape[-1])[:token_count].softmax(dim=-1)[:, 0]
    synchronize(torch, device)
    elapsed = time.perf_counter() - started
    score1 = score1.detach().cpu().numpy()
    score2 = score2.detach().cpu().numpy()
    for score in (score1, score2):
        if score.shape != (token_count,) or not np.isfinite(score).all():
            raise ValueError('Invalid length or non-finite valid-token scores')
        if np.any(score < 0) or np.any(score > 1):
            raise ValueError('Probability scores fall outside [0, 1]')
    record = {'token_count': int(token_count), 'window_count': int(visual.shape[0]),
              'declared_lengths': lengths.tolist(),
              'physical_valid_lengths': [max(0, min(window, token_count - i * window))
                                         for i in range(visual.shape[0])],
              'feature_timeline_frames': int(token_count) * 16,
              'branch1_range': [float(score1.min()), float(score1.max())],
              'branch2_range': [float(score2.min()), float(score2.max())],
              'model_forward_seconds': elapsed, 'valid_scores_finite': True}
    return score1, score2, record


def prepare_cases(mode, audited_videos, window):
    if mode == 'full':
        return [(record, None, 'full_video') for record in audited_videos]
    anomaly = next(record for record in audited_videos if not record['video'].startswith('Normal_'))
    normals = [record for record in audited_videos if record['video'].startswith('Normal_')]
    normal = min(normals, key=lambda record: record['feature_length'])
    long_video = next(record for record in audited_videos if record['feature_length'] >= 2 * window)
    cases = [(anomaly, None, 'anomaly_video'), (normal, None, 'short_normal_video'),
             (long_video, None, 'long_video')]
    cases.extend((long_video, count, f'boundary_prefix_{count}')
                 for count in (window - 1, window, window + 1, 2 * window))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ucf_resources.json')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    run_id = f'ucf_official_{args.mode}_{args.device}_{stamp}'
    output = ROOT / 'outputs/baseline' / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scores').mkdir()
    report = {'run_id': run_id, 'mode': args.mode, 'dataset': 'UCF-Crime',
              'input_type': 'official_features', 'device': args.device,
              'sampling': 'not_applied', 'budget': None,
              'success': False, 'videos': [], 'metrics': None,
              'input_sha256': {}, 'output_dir': str(output),
              'timing_scope': 'Cached-feature execution only; not raw-video efficiency',
              'smoke_outputs_are_benchmark_results': False}
    try:
        provenance_path = ROOT / 'VadCLIP/UPSTREAM.json'
        provenance = json.loads(provenance_path.read_text())
        report['upstream_repository'] = provenance['repository']
        report['upstream_commit'] = provenance['commit']
        report['input_sha256']['upstream_provenance'] = sha256(provenance_path)
        # Vendored VadCLIP inherits the research repository; never label its HEAD
        # as the original upstream commit. Source hashes remain the run identity.
        report['research_commit'] = None
        report['research_dirty'] = None
        try:
            revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=False)
            status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain',
                                     '--untracked-files=no'],
                                    capture_output=True, text=True, check=False)
            if revision.returncode == 0:
                report['research_commit'] = revision.stdout.strip()
            if status.returncode == 0:
                report['research_dirty'] = bool(status.stdout.strip())
        except FileNotFoundError:
            pass
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score, average_precision_score

        sys.path.insert(0, str(ROOT / 'VadCLIP/src'))
        from model import CLIPVAD
        import ucf_option
        from utils.tools import get_prompt_text

        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; select an available device explicitly')
        if args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValueError('MPS is unavailable; select an available device explicitly')
        config_path = args.config.expanduser().resolve()
        config = json.loads(config_path.read_text())
        audit_path = ROOT / 'outputs/setup/ucf_baseline_input_check.json'
        audit = json.loads(audit_path.read_text())
        if not audit.get('ready_for_small_inference') or not audit.get('data_checks_passed') or audit.get('issues'):
            raise ValueError('Resolve baseline input audit issues first')
        local_csv = ROOT / 'outputs/setup/ucf_CLIP_rgbtest.local.csv'
        manifest = ROOT / 'UCF-Crime/test/manifest.json'
        gt_path = ROOT / 'VadCLIP/list/gt_ucf.npy'
        hashes = {name: sha256(path) for name, path in
                  [('config', config_path), ('local_csv', local_csv), ('manifest', manifest), ('gt', gt_path)]}
        if any(value != audit['input_sha256'].get(name) for name, value in hashes.items()):
            raise ValueError('Config, list, manifest or GT changed since the input audit')
        report['input_sha256'].update(hashes)
        report['input_sha256'].update({
            'script': sha256(Path(__file__)), 'input_audit': sha256(audit_path),
            'model_source': sha256(ROOT / 'VadCLIP/src/model.py'),
            'layers_source': sha256(ROOT / 'VadCLIP/src/utils/layers.py'),
            'tools_source': sha256(ROOT / 'VadCLIP/src/utils/tools.py'),
            'official_test_source': sha256(ROOT / 'VadCLIP/src/ucf_test.py')})
        with local_csv.open(newline='') as handle:
            local_rows = list(csv.DictReader(handle))
        if len(local_rows) != len(audit['videos']) or len(local_rows) != 290:
            raise ValueError('Expected 290 audited test videos')
        if any(project_path(row['path']) != project_path(record['path'])
               for row, record in zip(local_rows, audit['videos'])):
            raise ValueError('Local list order differs from audited feature order')
        defaults = ucf_option.parser.parse_args([])
        clip_path, ucf_path = project_path(config['clip_checkpoint']), project_path(config['ucf_checkpoint'])
        resource_audit = json.loads((ROOT / 'outputs/setup/ucf_resource_check.json').read_text())
        for name, path in [('clip', clip_path), ('ucf', ucf_path)]:
            value = sha256(path)
            report['input_sha256'][f'{name}_checkpoint'] = value
            if value != resource_audit['checkpoints'][name]['sha256']:
                raise ValueError(f'{name} checkpoint changed since the resource audit')
        if args.mode == 'full':
            smoke_path = ROOT / 'outputs/setup/ucf_smoke_check.json'
            smoke = json.loads(smoke_path.read_text())
            if not smoke.get('success') or smoke.get('device') != args.device or smoke.get('mode') != 'smoke':
                raise ValueError('Run a successful smoke check on the selected device first')
            for key in ('config', 'local_csv', 'manifest', 'gt', 'script', 'model_source',
                        'layers_source', 'tools_source', 'clip_checkpoint', 'ucf_checkpoint',
                        'upstream_provenance'):
                if report['input_sha256'][key] != smoke['input_sha256'].get(key):
                    raise ValueError(f'{key} changed since the smoke check; rerun smoke first')
            report['input_sha256']['passed_smoke_report'] = sha256(smoke_path)
        torch.manual_seed(defaults.seed)
        model = CLIPVAD(defaults.classes_num, defaults.embed_dim, defaults.visual_length,
                        defaults.visual_width, defaults.visual_head, defaults.visual_layers,
                        defaults.attn_window, defaults.prompt_prefix, defaults.prompt_postfix,
                        args.device, clip_checkpoint=str(clip_path))
        model.load_state_dict(torch.load(ucf_path, map_location='cpu', weights_only=True), strict=True)
        model.to(args.device).eval()
        prompts = get_prompt_text(dict(zip(LABELS, LABELS)))
        report.update(environment={'python': sys.version, 'executable': sys.executable,
                                   'torch': torch.__version__, 'numpy': np.__version__,
                                   'platform': platform.platform(), 'machine': platform.machine(),
                                   'torch_cpu_threads': torch.get_num_threads(),
                                   'cuda_device_name': torch.cuda.get_device_name() if args.device == 'cuda' else None},
                      precision={'visual_tokens': 'model casts to float32',
                                 'vad_parameters': str(model.classifier.weight.dtype),
                                 'clip_parameters': str(model.clipmodel.token_embedding.weight.dtype)},
                      model_options={key: getattr(defaults, key) for key in
                                     ('classes_num', 'embed_dim', 'visual_length', 'visual_width', 'visual_head',
                                      'visual_layers', 'attn_window', 'prompt_prefix', 'prompt_postfix', 'seed')},
                      prompt_text=prompts,
                      rules={'crop_variant': '__5.npy', 'window_policy': 'upstream process_split and declared lengths',
                             'padding_mask': 'upstream encode_video passes None to temporal',
                             'score_branch1': 'sigmoid(logits1)', 'score_branch2': '1 - softmax(logits2)[:, 0]',
                             'score_restore': 'repeat each official feature score 16 times',
                             'branch_combination': None, 'smoothing': None})
        gt = np.load(gt_path, mmap_mode='r', allow_pickle=False)
        branch1_frames, branch2_frames = [], []
        cases = prepare_cases(args.mode, audit['videos'], defaults.visual_length)
        run_started = time.perf_counter()
        for case_index, (audited, prefix, case_name) in enumerate(cases):
            path = project_path(audited['path'])
            feature = np.load(path, allow_pickle=False)
            if list(feature.shape) != audited['shape'] or str(feature.dtype) != audited['dtype']:
                raise ValueError(f'Feature shape/dtype changed: {path.name}')
            feature_sha = sha256(path)
            if prefix is not None:
                feature = feature[:prefix]
            score1, score2, details = infer_feature(np, torch, model, feature, defaults.visual_length, prompts, args.device)
            frame1, frame2 = np.repeat(score1, 16), np.repeat(score2, 16)
            offset = audited['gt_offset']
            gt_segment = np.asarray(gt[offset:offset + len(frame1)])
            if len(gt_segment) != len(frame1):
                raise ValueError('Restored scores and GT segment lengths differ')
            score_file = output / 'scores' / f'{case_index:03d}_{Path(audited["video"]).stem}_{case_name}.npz'
            np.savez_compressed(score_file, branch1_feature_scores=score1,
                                branch2_feature_scores=score2, gt=gt_segment)
            report['videos'].append({'official_csv_index': audited['index'], 'video': audited['video'],
                                     'case': case_name, 'prefix_only': prefix is not None,
                                     'feature_sha256': feature_sha, 'gt_offset': offset,
                                     'score_file': str(score_file), **details})
            if args.mode == 'full':
                branch1_frames.append(frame1)
                branch2_frames.append(frame2)
            print(f'{case_index + 1}/{len(cases)}: {audited["video"]} [{case_name}] OK', flush=True)
        report['cached_feature_loop_seconds'] = time.perf_counter() - run_started
        if args.mode == 'full':
            frame1, frame2 = np.concatenate(branch1_frames), np.concatenate(branch2_frames)
            if len(frame1) != len(gt) or len(frame2) != len(gt):
                raise ValueError('Full restored score length does not equal the full GT length')
            report['metrics'] = {
                'evaluation_frames': len(gt), 'test_videos': len(cases), 'includes_normal': True,
                'branch1': {'roc_auc': float(roc_auc_score(gt, frame1)),
                            'ap': float(average_precision_score(gt, frame1))},
                'branch2': {'roc_auc': float(roc_auc_score(gt, frame2)),
                            'ap': float(average_precision_score(gt, frame2))}}
            np.savez_compressed(output / 'frame_scores.npz', branch1=frame1, branch2=frame2)
            print(json.dumps(report['metrics'], indent=2))
        report['success'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        summary_path = output / 'summary.json'
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if args.mode == 'smoke':
            (ROOT / 'outputs/setup/ucf_smoke_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'Report: {summary_path}', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
