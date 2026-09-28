#!/usr/bin/env python
"""FlyBrain experiment: baseline activity with no stimulation.

    python experiments/baseline_activity.py --mode subset --duration-ms 1000
    python experiments/baseline_activity.py --mode whole-brain --duration-ms 100
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.experiments.baseline_activity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
