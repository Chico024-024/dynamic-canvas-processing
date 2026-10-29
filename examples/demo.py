"""Run the complete DCP Core demonstration.

Usage:
    python examples/demo.py [--out output/demo] [--duration 16] [--transition 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcp.demo import run_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/demo")
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--transition", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = run_demo(
        output_dir=args.out,
        duration_s=args.duration,
        transition_s=args.transition,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    return 0 if result.baseline_report.passed and result.dcp_report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
