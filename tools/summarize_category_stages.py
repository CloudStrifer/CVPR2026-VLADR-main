"""Print/export completed stage results from existing JSON, without rerunning evaluation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid.evaluation.stage_reporting import format_stage_result, write_stage_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', help='defaults to run-dir; writes only derived stage result files')
    parser.add_argument('--stage', help='print only this stage; exported files still contain all completed stages')
    parser.add_argument('--include-oracle', action='store_true')
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    summary = json.loads((run / 'evaluation_summary.json').read_text(encoding='utf-8-sig'))
    history = summary['stages']
    if args.stage and args.stage not in {r['stage_id'] for r in history}:
        parser.error('stage has not completed evaluation: ' + args.stage)
    results = write_stage_results(history, args.output_dir or run)
    if not results:
        print('暂无已完成评估。', flush=True)
    for result in results:
        if args.stage is None or result['stage_id'] == args.stage:
            print(format_stage_result(result, include_oracle=args.include_oracle), flush=True)
    return results


if __name__ == '__main__':
    main()
