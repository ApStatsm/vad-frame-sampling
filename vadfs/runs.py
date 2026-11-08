"""Run directories, provenance records and summary writing."""

from datetime import datetime
import json
import subprocess
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from .paths import ROOT, sha256


TIMEZONE = ZoneInfo('Asia/Seoul')


def now():
    return datetime.now(TIMEZONE)


def create_run_dir(prefix, mode, subdirs=(), output_root=None):
    """outputs/baseline/<prefix>_<mode>_<YYYYMMDD_HHMMSS>[_n]; never reuse an existing run."""
    base = f'{prefix}_{mode}_{now().strftime("%Y%m%d_%H%M%S")}'
    number = 1
    while True:
        run_id = base if number == 1 else f'{base}_{number}'
        base_root = Path(output_root) if output_root is not None else ROOT / 'outputs/baseline'
        output = base_root / run_id
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            number += 1
            continue
        for name in subdirs:
            (output / name).mkdir()
        return run_id, output


def record_provenance(report):
    """Upstream VadCLIP commit plus the research repository commit and dirty state."""
    provenance_path = ROOT / 'VadCLIP/UPSTREAM.json'
    provenance = json.loads(provenance_path.read_text())
    report['input_sha256']['upstream_provenance'] = sha256(provenance_path)
    report.update(upstream_repository=provenance['repository'], upstream_commit=provenance['commit'])
    snapshot = ROOT / 'SOURCE_SNAPSHOT.json'
    if snapshot.exists():
        report['source_snapshot'] = json.loads(snapshot.read_text())
    # Vendored VadCLIP inherits the research repository; never label its HEAD
    # as the original upstream commit. Source hashes remain the run identity.
    for key, command in (('research_commit', ['rev-parse', 'HEAD']),
                         ('research_dirty', ['status', '--porcelain', '--untracked-files=normal'])):
        try:
            top = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'],
                                 capture_output=True, text=True, check=False)
            if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT:
                report[key] = None
                continue
            result = subprocess.run(['git', '-C', str(ROOT), *command], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            report[key] = None
            continue
        if result.returncode != 0:
            report[key] = None
        else:
            report[key] = bool(result.stdout.strip()) if key == 'research_dirty' else result.stdout.strip()


def write_json(path, data):
    """Atomic replace so interrupted saves do not leave a truncated summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
