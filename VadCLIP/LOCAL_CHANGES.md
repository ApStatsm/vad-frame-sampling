# Local modifications for VAD Frame Sampling

Upstream: https://github.com/nwpu-zxr/VadCLIP  
Base commit: `c41067f07d252efcda18008bea367886070c33b0`  
Recorded: 2026-10-06

This directory contains the upstream tracked files and the following local modifications. The upstream README, LICENSE, evaluation lists, GT arrays, CLIP tokenizer vocabulary and other source files are preserved. The research repository tracks this as an ordinary directory.

| File | Local modification | Validation evidence |
|---|---|---|
| `src/model.py` | Add optional `clip_checkpoint=None` to select the local CLIP archive. Existing calls retain the upstream named-model default. | User CPU strict loading and MPS smoke passed before the later comment-only notice addition. |
| `src/utils/layers.py` | Replace hard-coded CUDA placement in `DistanceAdj` with the parameter device. Preserve the distance, exponential expression and checkpoint parameters. | User MPS smoke passed before the later comment-only notice addition. CPU/CUDA forward equivalence remains unverified. |

The modified source files carry prominent local change notices. These notices change file hashes without changing model calculations. Run the current-version smoke before full evaluation; the old smoke does not validate the new hashes.

Research code under `../vadfs/` and `../scripts/` is a separate addition. Its evaluators mirror the upstream scoring/window behavior and record both the research commit and the fixed upstream commit from `UPSTREAM.json`. No retraining or upstream inference-policy change is part of the GitHub preparation.

The previous nested `.git` history and the pre-integration modification patch are backed up under the ignored project-local `.local-backups/vadclip-before-integration/`. That backup is machine-local and is not a dependency for a fresh clone.

## Separate refactor copy — 2026-11-08

`src/clip/clip.py` imports the public `packaging.version` module directly instead
of `pkg_resources.packaging`. CLIP computations and version comparison are
unchanged in source; setuptools is no longer a CLIP runtime dependency. The old
setuptools 57.4.0 pin is archived for provenance, not used for installation.
No numerical equivalence/import test was executed during this refactor.
Reproduce the official and raw baselines in the target environment before using
new sampler results. Model weights, temporal/window quirks and upstream GT files
were not edited. All current source hashes are captured by the common runner.
