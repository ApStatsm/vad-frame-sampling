"""Fixed pretrained VadCLIP: model loading and upstream-equivalent test inference.

Retains upstream process_split, declared window lengths, mask behavior, prompts,
two score branches and 16-fold score restoration (VadCLIP/src/ucf_test.py).
Known upstream quirks are kept on purpose (do not fix without re-running both
reference runs): process_split adds an extra padded window when the token count
is an exact multiple of the window, and encode_video passes None instead of the
padding mask to the temporal module.
"""

import time

import numpy as np
import torch

from .paths import ROOT, add_vadclip_src, sha256

add_vadclip_src()
import ucf_option  # noqa: E402
from model import CLIPVAD  # noqa: E402
from utils.tools import get_batch_mask, get_prompt_text, process_split  # noqa: E402


MODEL_OPTION_KEYS = ('classes_num', 'embed_dim', 'visual_length', 'visual_width', 'visual_head',
                     'visual_layers', 'attn_window', 'prompt_prefix', 'prompt_postfix', 'seed')
FRAMES_PER_TOKEN = 16
SOURCES = {'model_source': 'VadCLIP/src/model.py', 'layers_source': 'VadCLIP/src/utils/layers.py',
           'tools_source': 'VadCLIP/src/utils/tools.py'}


def source_hashes():
    return {key: sha256(ROOT / relative) for key, relative in SOURCES.items()}


def check_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select an available device explicitly')
    if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise ValueError('MPS is unavailable; select an available device explicitly')


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def default_options(seed=None):
    options = ucf_option.parser.parse_args([])
    if seed is not None:
        options.seed = seed
    return options


def load_model(options, device, clip_checkpoint, vadclip_checkpoint):
    """Strict-load the dataset checkpoint; the caller seeds RNGs before this call."""
    model = CLIPVAD(options.classes_num, options.embed_dim, options.visual_length,
                    options.visual_width, options.visual_head, options.visual_layers,
                    options.attn_window, options.prompt_prefix, options.prompt_postfix,
                    device, clip_checkpoint=str(clip_checkpoint))
    model.load_state_dict(torch.load(vadclip_checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def prompt_text(labels):
    return get_prompt_text(dict(zip(labels, labels)))


def upstream_lengths(token_count, window):
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


def infer_feature(model, feature, window, prompts, device):
    """[T,512] features -> branch1/branch2 token scores and [T,classes] probabilities."""
    split, token_count = process_split(feature, window)
    visual = torch.tensor(split)
    if token_count < window:
        visual = visual.unsqueeze(0)
    lengths = upstream_lengths(token_count, window)
    if visual.ndim != 3 or visual.shape[0] != len(lengths) or visual.shape[1] != window:
        raise ValueError('Feature split and upstream declared lengths are inconsistent')
    mask = get_batch_mask(lengths, window).to(device)
    visual = visual.to(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        _, logits1, logits2 = model(visual, mask, prompts, lengths)
        score1 = torch.sigmoid(logits1.reshape(-1, logits1.shape[-1])[:token_count].squeeze(-1))
        class_scores = logits2.reshape(-1, logits2.shape[-1])[:token_count].softmax(dim=-1)
        score2 = 1 - class_scores[:, 0]
    synchronize(device)
    elapsed = time.perf_counter() - started
    score1 = score1.detach().cpu().numpy()
    score2 = score2.detach().cpu().numpy()
    class_scores = class_scores.detach().cpu().numpy()
    if (class_scores.shape != (token_count, len(prompts))
            or not np.isfinite(class_scores).all()
            or not np.allclose(class_scores.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Invalid class probability shape or normalization')
    for score in (score1, score2):
        if score.shape != (token_count,) or not np.isfinite(score).all():
            raise ValueError('Invalid length or non-finite valid-token scores')
        if np.any(score < 0) or np.any(score > 1):
            raise ValueError('Probability scores fall outside [0, 1]')
    record = {'token_count': int(token_count), 'window_count': int(visual.shape[0]),
              'declared_lengths': lengths.tolist(),
              'physical_valid_lengths': [max(0, min(window, token_count - i * window))
                                         for i in range(visual.shape[0])],
              'feature_timeline_frames': int(token_count) * FRAMES_PER_TOKEN,
              'branch1_range': [float(score1.min()), float(score1.max())],
              'branch2_range': [float(score2.min()), float(score2.max())],
              'model_forward_seconds': elapsed, 'valid_scores_finite': True}
    return score1, score2, class_scores, record


def restore_frames(token_scores):
    """Repeat each token score over its 16-frame block on the evaluation timeline."""
    return np.repeat(token_scores, FRAMES_PER_TOKEN, axis=0)
