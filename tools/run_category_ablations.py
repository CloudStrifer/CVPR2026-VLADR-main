"""Generate or execute matched-budget continuous ECPM/PGCA ablation runs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_category_progressive import DEFAULTS, ProgressiveRun
from reid.evaluation.category_progressive import lifelong_summary
from reid.models.category_adapter_bank import build_category_model
from reid.utils.progressive_checkpoint import atomic_json


ABLATIONS = {
    'full': {},
    'persistent_baseline': dict(consistency='off', init_mode='default', control_summary='identity_mean', routing_summary='identity_mean'),
    'fixed_consistency': dict(consistency='fixed'),
    'default_initialization': dict(init_mode='default'),
    'random_historical_source': dict(init_mode='random'),
    'global_only_initialization': dict(alpha=1.),
    'mean_control_only': dict(control_summary='identity_mean'),
    'mean_routing_only': dict(routing_summary='identity_mean'),
    'mean_control_and_routing': dict(control_summary='identity_mean', routing_summary='identity_mean'),
}


def run_suite(base, output, names=None, execute=False, model_factory=build_category_model):
    names = list(ABLATIONS) if names is None else list(names)
    if not names or len(names) != len(set(names)) or set(names) - set(ABLATIONS):
        raise ValueError('unknown or duplicate ablation name')
    if set(base) - (set(DEFAULTS) | {'stream_config'}):
        raise ValueError('base config contains unknown training options')
    if any(base.get(k, DEFAULTS[k]) != DEFAULTS[k] for k in
           ('consistency', 'init_mode', 'control_summary', 'routing_summary', 'transfer_source')):
        raise ValueError('base config must use full-method defaults; ablations apply their own overrides')
    base = dict(base, stream_config=str(Path(base['stream_config']).resolve()), evaluate=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(base=base, experiments={name: dict(base, **ABLATIONS[name]) for name in names},
                note='same data, seed and training budget; each checkpoint evaluates oracle and prototype')
    manifest = output / 'ablation_plan.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != plan:
        raise ValueError('ablation plan differs from existing experiment directory')
    atomic_json(plan, manifest)
    results = {}
    if execute:
        for name, settings in plan['experiments'].items():
            directory = output / name
            run = ProgressiveRun(directory, settings, resume=(directory / 'latest.pt').exists(), model_factory=model_factory)
            run.run()
            results[name] = lifelong_summary(run.evaluations)
            atomic_json(dict(results=results, completed=list(results), planned=names), output / 'ablation_results.json')
    return plan, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', required=True, help='JSON object of continuous-run settings')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--experiments', nargs='+', choices=tuple(ABLATIONS))
    parser.add_argument('--execute', action='store_true', help='otherwise only write the reviewable experiment plan')
    args = parser.parse_args(argv)
    return run_suite(json.loads(Path(args.base_config).read_text(encoding='utf-8')), args.output_dir, args.experiments, args.execute)


if __name__ == '__main__':
    main()
