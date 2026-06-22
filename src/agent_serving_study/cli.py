from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .config import load_config
from .orchestrator import build_server_command, run_experiment
from .summary import summarize_all, summarize_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent serving performance experiment runner")
    parser.add_argument("--config", default="configs/experiments.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List configured experiments")
    command_parser = subparsers.add_parser("command", help="Print an SGLang launch command")
    command_parser.add_argument("experiment")

    run_parser = subparsers.add_parser("run", help="Run one complete experiment")
    run_parser.add_argument("experiment")
    run_parser.add_argument("--artifact-dir", type=Path)
    run_parser.add_argument("--keep-server", action="store_true")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize one artifact directory")
    summarize_parser.add_argument("artifact_dir", type=Path)

    all_parser = subparsers.add_parser("summarize-all", help="Build a CSV across all runs")
    all_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    all_parser.add_argument("--output", type=Path, default=Path("artifacts/summary.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "list":
        for name, experiment in config.experiments.items():
            print(f"{name:30} {experiment.get('description', '')}")
    elif args.command == "command":
        print(shlex.join(build_server_command(config, args.experiment)))
    elif args.command == "run":
        summary = run_experiment(config, args.experiment, args.artifact_dir, args.keep_server)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize_artifact(args.artifact_dir), ensure_ascii=False, indent=2))
    elif args.command == "summarize-all":
        rows = summarize_all(args.artifacts_dir, args.output)
        print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()

