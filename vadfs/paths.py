"""Project-root path resolution and file hashing."""

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VADCLIP_SRC = ROOT / 'VadCLIP/src'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def audited_path(value):
    """Rebase an audited absolute path whose project folder was later renamed.

    The input audit stores absolute paths (e.g. the former '03_인공지능 프로젝트'
    folder). Keep the audit file unchanged; map the part after the project
    directory name onto the current ROOT. Callers still verify SHA-256.
    """
    path = project_path(value)
    if path.exists():
        return path
    parts = Path(value).parts
    if ROOT.name in parts:
        marker = len(parts) - 1 - parts[::-1].index(ROOT.name)
        rebased = ROOT.joinpath(*parts[marker + 1:])
        if rebased.exists():
            return rebased
    raise FileNotFoundError(f'Audited path not found, also after project-root rebasing: {value}')


def add_vadclip_src():
    """Make the vendored upstream modules (model, clip, crop, utils, ucf_option) importable."""
    if str(VADCLIP_SRC) not in sys.path:
        sys.path.insert(0, str(VADCLIP_SRC))


def package_hashes():
    """SHA-256 of every vadfs module, recorded with each run."""
    return {path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted((ROOT / 'vadfs').rglob('*.py'))}
