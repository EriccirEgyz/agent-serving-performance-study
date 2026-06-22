#!/usr/bin/env python3
"""Run the official BFCL generator while adding per-task wall-clock timing.

The wrapper monkey-patches one public module function in memory. It does not modify
the installed BFCL package or alter benchmark prompts/evaluation behavior.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from bfcl_eval import _llm_response_generation as generation


original_inference = generation.multi_threaded_inference


def timed_inference(handler, test_case, include_input_log, exclude_state_log):
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    result = original_inference(handler, test_case, include_input_log, exclude_state_log)
    duration = time.perf_counter() - start
    failed = isinstance(result.get("result"), str) and "Error during inference" in result["result"]
    result["_asp_metrics"] = {
        "duration_seconds": duration,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "failed": failed,
    }
    return result


generation.multi_threaded_inference = timed_inference


if __name__ == "__main__":
    generation.main(generation.get_args())

