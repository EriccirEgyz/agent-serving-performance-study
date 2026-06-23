from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_serving_study.config import load_config
from agent_serving_study.orchestrator import build_server_command
from agent_serving_study.prometheus import counter_delta, labeled_counter_delta
from agent_serving_study.summary import percentile, summarize_artifact


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_config(cls.root / "configs" / "experiments.json")

    def test_experiment_matrix_and_command(self):
        self.assertEqual(len(self.config.experiments), 5)
        command = build_server_command(self.config, "single_gpu_medium_hicache")
        self.assertIn("--enable-hierarchical-cache", command)
        self.assertIn("--enable-cache-report", command)
        self.assertIn("--hicache-size", command)
        self.assertNotIn("--tool-call-parser", command)
        self.assertNotIn("--hicache-io-backend", command)
        self.assertNotIn("--hicache-mem-layout", command)

    def test_prometheus_deltas(self):
        before = 'sglang:prompt_tokens_total 10\nsglang:cached_tokens_total{cache_source="device"} 2\n'
        after = (
            'sglang:prompt_tokens_total 34\n'
            'sglang:cached_tokens_total{cache_source="device"} 8\n'
            'sglang:cached_tokens_total{cache_source="host"} 3\n'
        )
        self.assertEqual(counter_delta(before, after, "sglang:prompt_tokens_total"), 24)
        self.assertEqual(
            labeled_counter_delta(before, after, "sglang:cached_tokens_total", "cache_source"),
            {"device": 6, "host": 3},
        )

    def test_percentile(self):
        self.assertEqual(percentile([1], 0.9), 1)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.9), 3.7)

    def test_synthetic_artifact_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp)
            result_dir = artifact / "bfcl" / "result" / "model"
            result_dir.mkdir(parents=True)
            records = [
                {"id": "a", "result": [], "_asp_metrics": {"duration_seconds": 1.0}},
                {"id": "b", "result": [], "_asp_metrics": {"duration_seconds": 3.0}},
            ]
            (result_dir / "BFCL_v4_multi_turn_base_result.json").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8"
            )
            (artifact / "run_metadata.json").write_text(
                json.dumps({"experiment": "synthetic", "benchmark_duration_seconds": 4}),
                encoding="utf-8",
            )
            (artifact / "metrics_before.prom").write_text(
                "sglang:prompt_tokens_total 0\nsglang:generation_tokens_total 0\n",
                encoding="utf-8",
            )
            (artifact / "metrics_after.prom").write_text(
                "sglang:prompt_tokens_total 100\n"
                "sglang:generation_tokens_total 20\n"
                "sglang:max_total_num_tokens 4096\n",
                encoding="utf-8",
            )
            (artifact / "gpu_samples.csv").write_text(
                "timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw\n"
                "2026/06/22 20:00:00,0,A100,1024,81920,50,200\n"
                "2026/06/22 20:00:01,0,A100,2048,81920,60,220\n",
                encoding="utf-8",
            )
            score_dir = artifact / "bfcl" / "score" / "model"
            score_dir.mkdir(parents=True)
            (score_dir / "BFCL_v4_multi_turn_base_score.json").write_text(
                json.dumps([{"accuracy": 0.5, "correct_count": 1, "total_count": 2}]),
                encoding="utf-8",
            )
            summary = summarize_artifact(artifact)
            self.assertEqual(summary["task_count"], 2)
            self.assertEqual(summary["mean_task_latency_seconds"], 2)
            self.assertEqual(summary["total_token_throughput"], 30)
            self.assertEqual(summary["benchmark_accuracy"], 0.5)
            self.assertEqual(summary["kv_cache_capacity_tokens"], 4096)
            self.assertEqual(summary["peak_gpu_memory_mib"], 2048)


if __name__ == "__main__":
    unittest.main()
