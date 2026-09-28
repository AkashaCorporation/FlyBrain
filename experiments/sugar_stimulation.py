#!/usr/bin/env python
"""FlyBrain experiment: sugar GRN stimulation.

    python experiments/sugar_stimulation.py --mode subset --duration-ms 1000
    python experiments/sugar_stimulation.py --mode whole-brain --rate-hz 100 \
        --duration-ms 1000 --compare-published
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.experiments.sugar_stimulation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
