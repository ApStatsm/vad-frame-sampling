"""Run directories, provenance records and summary writing."""

from datetime import datetime
import json
import subprocess
from zoneinfo import ZoneInfo

from .paths import ROOT, sha256


TIMEZONE = ZoneInfo('Asia/Seoul')


def now():
    return datetime.now(TIMEZONE)


def create_run_dir(prefix, mode, subdirs=()):
    """outputs/baseline/<prefix>_<mode>_<YYYYMMDD_HHMMSS>[_n]; never reuse an existing run."""
    base = f'{prefix}_{mode}_{now().strftime("%Y%m%d_%H%M%S")}'
    number = 1
    while True:
        run_id = base if number == 1 else f'{base}_{number}'
        output = ROOT / 'outputs/baseline' / run_id
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
    # Vendored VadCLIP inherits the research repository; never label its HEAD
    # as the original upstream commit. Source hashes remain the run identity.
    for key, command in (('research_commit', ['rev-parse', 'HEAD']),
                         ('research_dirty', ['status', '--porcelain', '--untracked-files=normal'])):
        try:
            result = subprocess.run(['git', '-C', str(ROOT), *command], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            report[key] = None
            continue
        if result.returncode != 0:
            report[key] = None
        else:
            report[key] = bool(result.stdout.strip()) if key == 'research_dirty' else result.stdout.strip()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
