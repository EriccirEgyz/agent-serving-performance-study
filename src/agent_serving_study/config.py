from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    root: Path
    study: dict[str, Any]
    benchmark: dict[str, Any]
    experiments: dict[str, dict[str, Any]]

    def experiment(self, name: str) -> dict[str, Any]:
        try:
            return self.experiments[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.experiments))
            raise ValueError(f"Unknown experiment {name!r}. Available: {available}") from exc


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root (pyproject.toml + configs).")


def load_config(path: str | Path) -> LoadedConfig:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("study", "benchmark", "experiments"):
        if key not in data:
            raise ValueError(f"Missing top-level config key: {key}")

    root = find_project_root(config_path.parent)
    study = data["study"]
    benchmark = data["benchmark"]
    experiments = data["experiments"]

    required_study = {"model_path", "served_model_name", "host", "port"}
    required_benchmark = {"model_name", "run_ids_file", "concurrency"}
    missing = required_study - study.keys()
    if missing:
        raise ValueError(f"Missing study keys: {sorted(missing)}")
    missing = required_benchmark - benchmark.keys()
    if missing:
        raise ValueError(f"Missing benchmark keys: {sorted(missing)}")
    if not experiments:
        raise ValueError("At least one experiment is required.")

    for name, experiment in experiments.items():
        for key in ("cuda_visible_devices", "tensor_parallel_size", "mem_fraction_static"):
            if key not in experiment:
                raise ValueError(f"Experiment {name!r} is missing {key!r}")
        fraction = float(experiment["mem_fraction_static"])
        if not 0 < fraction < 1:
            raise ValueError(f"Experiment {name!r} has invalid mem_fraction_static={fraction}")

    return LoadedConfig(config_path, root, study, benchmark, experiments)


def resolve_from_root(config: LoadedConfig, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config.root / path).resolve()

