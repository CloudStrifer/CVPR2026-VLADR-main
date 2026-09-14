"""Clone a pending first-stage checkpoint for the reviewed evaluation-only fix.

Training source/environment checks stay strict. Run after stopping the old
process; original checkpoint and logs are never edited.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train_category_progressive import ProgressiveRun
from reid.utils.progressive_checkpoint import atomic_json, atomic_save, exact_runtime, file_digest


# Populated with the reviewed sources, normalized only for CRLF/LF transfer.
APPROVED_TARGETS = {
    'reid/evaluation/category_oracle.py': '570dc67449b1ba5d87f69a0a12522d57c7083e31cf965a0e9c2606f5e76a75d2',
    'reid/evaluation/category_progressive.py': 'ea996d47212b82b785a54bb3ab20916f7a0ace20f3bc8eff804d09079084ba11',
}


def validate_upgrade(state, current):
    if (state.get('kind') != 'ecpm_pgca_continuous' or state.get('schema_version') != 1
            or state.get('phase') != 'between_stages' or state.get('stage_index') != 1
            or state.get('pending_evaluation') != 0 or state.get('evaluations') != []
            or state.get('trainer') is not None or state.get('candidate') is not None):
        raise ValueError('upgrade requires the committed first stage with its first evaluation still pending')
    old = state['runtime']
    differences = [key for key in set(old) | set(current)
                   if key != 'source_sha256' and old.get(key) != current.get(key)]
    if differences:
        raise ValueError('software/device/thread environment changed: ' + ', '.join(sorted(differences)))
    before, after = old['source_sha256'], current['source_sha256']
    if set(before) != set(after):
        raise ValueError('source file inventory changed; restore the original training source set')
    changed = {name: dict(before=before[name], after=after[name])
               for name in before if before[name] != after[name]}
    forbidden = [name for name in changed if name.replace('\\', '/') not in APPROVED_TARGETS]
    if forbidden:
        raise ValueError('non-evaluation source changed: ' + ', '.join(sorted(forbidden)))
    if not changed:
        raise ValueError('checkpoint already matches this code; use ordinary --resume')
    return changed


def upgrade(run_dir, output_dir):
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError('migration output must not exist: ' + str(output))
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('source and destination must be separate, non-nested run directories')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('run directory contains symlinks; migration requires ordinary local files')
    for name, expected in APPROVED_TARGETS.items():
        actual = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('evaluation source is not the reviewed fix: ' + name)
    checkpoint = source / 'latest.pt'
    original_digest = file_digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    runtime = exact_runtime(state['settings']['device'])
    changed = validate_upgrade(state, runtime)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.upgrade-', dir=output.parent))
    print(json.dumps(dict(event='evaluation_upgrade_copy', source=str(source), destination=str(output)),
                     ensure_ascii=False), flush=True)
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        if file_digest(checkpoint) != original_digest or file_digest(staging / 'latest.pt') != original_digest:
            raise ValueError('checkpoint changed during copy; stop the training process before upgrading')
        upgraded = copy.deepcopy(state)
        upgraded['runtime'] = runtime
        atomic_save(upgraded, staging / 'latest.pt')
        # Full normal loader validates reference, stream, ECPM, cursor and log
        # prefixes. Any recovered speculative log tails belong to the copy.
        restored = ProgressiveRun(staging, resume=True)
        restored.save()
        config_path = staging / 'run_config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding='utf-8'))
            config['runtime'] = runtime
            atomic_json(config, config_path)
        audit = dict(kind='evaluation_only_upgrade', source=str(source), destination=str(output),
                     original_checkpoint_sha256=original_digest,
                     upgraded_checkpoint_sha256=file_digest(staging / 'latest.pt'),
                     source_runtime=state['runtime'], target_runtime=runtime, changed_sources=changed,
                     stage_index=state['stage_index'], total_updates=state['total_updates'],
                     pending_evaluation=state['pending_evaluation'],
                     note='Training state retained. Pending evaluation restarts using the reviewed faster implementation.')
        atomic_json(audit, staging / 'evaluation_upgrade.json')
        if output.exists():
            raise FileExistsError(output)
        if staging.resolve().parent != output.parent:
            raise ValueError('staging directory escaped the requested destination parent')
        staging.rename(output)
    except BaseException:
        print('Migration not published; original untouched. Diagnostic copy: ' + str(staging), file=sys.stderr)
        raise
    print(json.dumps(dict(event='evaluation_upgrade_complete', output_dir=str(output),
                          total_updates=state['total_updates'], pending_evaluation=0), ensure_ascii=False), flush=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    return upgrade(args.run_dir, args.output_dir)


if __name__ == '__main__':
    main()
