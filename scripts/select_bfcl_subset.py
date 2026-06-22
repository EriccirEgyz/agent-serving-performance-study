#!/usr/bin/env python3
"""Select a deterministic BFCL v4 multi-turn subset from the installed package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path


DEFAULT_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def locate_data_dir() -> Path:
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("bfcl-eval is not installed. Install it before selecting a subset.")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    data_dir = package_dir / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"Could not find BFCL data directory at {data_dir}")
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--output", type=Path, default=Path("configs/bfcl_run_ids.json"))
    args = parser.parse_args()

    if args.per_category < 1:
        parser.error("--per-category must be positive")
    data_dir = locate_data_dir()
    rng = random.Random(args.seed)
    selected: dict[str, list[str]] = {}

    for category in args.categories:
        candidates = sorted(data_dir.glob(f"BFCL_v*_{category}.json"))
        if not candidates:
            raise FileNotFoundError(f"No BFCL dataset found for category {category!r} in {data_dir}")
        dataset = candidates[-1]
        ids = [str(record["id"]) for record in load_records(dataset) if "id" in record]
        if len(ids) < args.per_category:
            raise ValueError(f"{dataset.name} only contains {len(ids)} IDs")
        selected[category] = sorted(rng.sample(ids, args.per_category))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {sum(map(len, selected.values()))} tasks into {args.output}")


if __name__ == "__main__":
    main()
