#!/usr/bin/env python
"""FlyBrain experiment: smoke test.

Wraps :mod:`flybrain.experiments.smoke_test` so the experiment can be run directly
from a checkout without installing the package.

    python experiments/smoke_test.py
    python experiments/smoke_test.py --use-dataset --mode subset
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.experiments.smoke_test import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
