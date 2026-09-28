#!/usr/bin/env python
"""FlyBrain experiment: silencing (stimulate A, silence B, compare).

    python experiments/silencing_test.py --mode subset --silence mn9
    python experiments/silencing_test.py --mode subset           # screens for target B
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.experiments.silencing_test import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
