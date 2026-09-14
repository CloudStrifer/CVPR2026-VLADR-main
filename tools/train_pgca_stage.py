"""Single-stage ECPM + both PGCA branches (default: drift and similarity)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid.models.category_adapter_bank import build_category_model
from tools.train_pgca_recurring_stage import main as _run_stage


def main(argv=None, model_factory=build_category_model):
    return _run_stage(argv, model_factory=model_factory, default_init_mode='similarity')


if __name__ == '__main__':
    main()
