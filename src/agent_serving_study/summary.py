from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .prometheus import aggregate, counter_delta, labeled_counter_delta, parse_prometheus


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _task_failed(record: dict[str, Any]) -> bool:
    result = record.get("result")
    if isinstance(result, str) and "Error during inference" in result:
        return True
    return bool(record.get("_asp_metrics", {}).get("failed", False))


def _gpu_memory_summary(path: Path) -> tuple[float | None, dict[str, float]]:
    if not path.exists():
        return None, {}
    peaks: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                index = row["index"].strip()
                memory = float(row["memory.used"].strip())
            except (KeyError, TypeError, ValueError):
                continue
            peaks[index] = max(memory, peaks.get(index, 0.0))
    return (max(peaks.values()) if peaks else None), peaks


def summarize_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    metadata_path = artifact_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    records: list[dict[str, Any]] = []
    for result_file in sorted((artifact_dir / "bfcl" / "result").rglob("*_result.json")):
        records.extend(read_json_records(result_file))

    durations = [
        float(record["_asp_metrics"]["duration_seconds"])
        for record in records
        if isinstance(record.get("_asp_metrics"), dict)
        and "duration_seconds" in record["_asp_metrics"]
    ]
    failures = sum(_task_failed(record) for record in records)

    category_scores: dict[str, float] = {}
    score_correct = 0
    score_total = 0
    for score_file in sorted((artifact_dir / "bfcl" / "score").rglob("*_score.json")):
        score_records = read_json_records(score_file)
        if not score_records:
            continue
        header = score_records[0]
        if not {"accuracy", "correct_count", "total_count"} <= header.keys():
            continue
        stem = score_file.stem
        category = stem.split("_", 2)[-1].removesuffix("_score")
        category_scores[category] = float(header["accuracy"])
        score_correct += int(header["correct_count"])
        score_total += int(header["total_count"])

    before_path = artifact_dir / "metrics_before.prom"
    after_path = artifact_dir / "metrics_after.prom"
    before = before_path.read_text(encoding="utf-8") if before_path.exists() else ""
    after = after_path.read_text(encoding="utf-8") if after_path.exists() else ""
    input_tokens = counter_delta(before, after, "sglang:prompt_tokens_total")
    output_tokens = counter_delta(before, after, "sglang:generation_tokens_total")
    cached_by_source = labeled_counter_delta(
        before, after, "sglang:cached_tokens_total", "cache_source"
    )
    cached_tokens = sum(cached_by_source.values())
    gauges = aggregate(parse_prometheus(after))
    peak_gpu_memory, peak_gpu_memory_by_index = _gpu_memory_summary(
        artifact_dir / "gpu_samples.csv"
    )
    wall_seconds = float(metadata.get("benchmark_duration_seconds") or 0.0)

    summary = {
        "experiment": metadata.get("experiment", artifact_dir.name),
        "artifact_dir": str(artifact_dir),
        "task_count": len(records),
        "successful_tasks": len(records) - failures,
        "success_rate": ((len(records) - failures) / len(records)) if records else None,
        "benchmark_accuracy": score_correct / score_total if score_total else None,
        "benchmark_accuracy_by_category": category_scores,
        "mean_task_latency_seconds": (sum(durations) / len(durations)) if durations else None,
        "p90_task_latency_seconds": percentile(durations, 0.90),
        "benchmark_duration_seconds": wall_seconds or None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_throughput": input_tokens / wall_seconds if wall_seconds else None,
        "output_token_throughput": output_tokens / wall_seconds if wall_seconds else None,
        "total_token_throughput": (input_tokens + output_tokens) / wall_seconds if wall_seconds else None,
        "cached_tokens": cached_tokens,
        "cached_tokens_by_source": cached_by_source,
        "prefix_cache_hit_rate": cached_tokens / input_tokens if input_tokens else None,
        "kv_cache_capacity_tokens": gauges.get("sglang:max_total_num_tokens"),
        "hicache_host_capacity_tokens": gauges.get("sglang:hicache_host_total_tokens"),
        "peak_gpu_memory_mib": peak_gpu_memory,
        "peak_gpu_memory_mib_by_index": peak_gpu_memory_by_index,
        "benchmark_return_code": metadata.get("benchmark_return_code"),
        "evaluation_return_code": metadata.get("evaluation_return_code"),
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize_all(artifacts_dir: Path, output_csv: Path) -> list[dict[str, Any]]:
    rows = []
    for metadata in sorted(artifacts_dir.glob("*/run_metadata.json")):
        rows.append(summarize_artifact(metadata.parent))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        scalar_keys = [
            key
            for key, value in rows[0].items()
            if not isinstance(value, (dict, list))
        ]
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)
    return rows
