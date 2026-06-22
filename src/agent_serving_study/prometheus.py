from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse_prometheus(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        labels = tuple(
            sorted(
                (key, bytes(value, "utf-8").decode("unicode_escape"))
                for key, value in LABEL_RE.findall(match.group("labels") or "")
            )
        )
        samples.append(Sample(match.group("name"), labels, float(match.group("value"))))
    return samples


def aggregate(samples: list[Sample]) -> dict[str, float]:
    result: defaultdict[str, float] = defaultdict(float)
    for sample in samples:
        result[sample.name] += sample.value
    return dict(result)


def counter_delta(before_text: str, after_text: str, metric_name: str) -> float:
    before = aggregate(parse_prometheus(before_text)).get(metric_name, 0.0)
    after = aggregate(parse_prometheus(after_text)).get(metric_name, 0.0)
    return max(0.0, after - before)


def labeled_counter_delta(
    before_text: str, after_text: str, metric_name: str, label_name: str
) -> dict[str, float]:
    def grouped(text: str) -> dict[str, float]:
        values: defaultdict[str, float] = defaultdict(float)
        for sample in parse_prometheus(text):
            if sample.name != metric_name:
                continue
            labels = dict(sample.labels)
            values[labels.get(label_name, "unknown")] += sample.value
        return dict(values)

    before = grouped(before_text)
    after = grouped(after_text)
    return {key: max(0.0, value - before.get(key, 0.0)) for key, value in after.items()}

