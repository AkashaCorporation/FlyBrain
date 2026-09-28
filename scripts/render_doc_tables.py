#!/usr/bin/env python
"""Regenerate the generated sections of the docs.

    python scripts/render_doc_tables.py          # write
    python scripts/render_doc_tables.py --check  # fail if stale (used by tests)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.model.docgen import update_document  # noqa: E402

TARGETS = [
    ROOT / "docs" / "MODEL_ASSUMPTIONS.md",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any target is out of date")
    args = ap.parse_args()

    stale = []
    for p in TARGETS:
        if not p.is_file():
            print(f"missing: {p}")
            stale.append(p)
            continue
        current = update_document(p, check=args.check)
        status = "up to date" if current else ("STALE" if args.check else "updated")
        print(f"{status}: {p}")
        if not current:
            stale.append(p)

    if args.check and stale:
        print(
            "\nRun `python scripts/render_doc_tables.py` to regenerate.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
