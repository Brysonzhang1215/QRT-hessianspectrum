"""Run a single experiment mode for a specific batch size."""

from __future__ import annotations

import argparse

import config
import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("clean", "robust", "combined"))
    parser.add_argument("batch_size", type=int)
    return parser.parse_args()


def main_entry() -> None:
    args = parse_args()
    config.QRT_BATCH_SIZES = (args.batch_size,)
    if args.mode == "clean":
        config.USE_COMBINED_LOSS = False
        main.main(run_mode="single")
    elif args.mode == "robust":
        config.USE_COMBINED_LOSS = True
        config.LOSS_ALPHA = 0.0
        main.main(run_mode="single")
    else:
        main.main(run_mode="combined")


if __name__ == "__main__":
    main_entry()

