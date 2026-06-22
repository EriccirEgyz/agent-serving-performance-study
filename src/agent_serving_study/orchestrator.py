from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LoadedConfig, resolve_from_root
from .summary import summarize_artifact


def build_server_command(config: LoadedConfig, experiment_name: str) -> list[str]:
    study = config.study
    experiment = config.experiment(experiment_name)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(study["model_path"]),
        "--served-model-name",
        str(study["served_model_name"]),
        "--host",
        str(study["host"]),
        "--port",
        str(study["port"]),
        "--tp-size",
        str(experiment["tensor_parallel_size"]),
        "--mem-fraction-static",
        str(experiment["mem_fraction_static"]),
        "--enable-metrics",
        "--enable-cache-report",
    ]
    if study.get("tool_call_parser"):
        command += ["--tool-call-parser", str(study["tool_call_parser"])]
    if study.get("reasoning_parser"):
        command += ["--reasoning-parser", str(study["reasoning_parser"])]
    if experiment.get("hicache"):
        command += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            str(experiment.get("hicache_size_gb", 16)),
            "--hicache-write-policy",
            str(experiment.get("hicache_write_policy", "write_through")),
        ]
        optional_hicache_args = {
            "hicache_io_backend": "--hicache-io-backend",
            "hicache_mem_layout": "--hicache-mem-layout",
        }
        for config_key, flag in optional_hicache_args.items():
            if config_key in experiment:
                command += [flag, str(experiment[config_key])]
    command.extend(str(arg) for arg in study.get("extra_server_args", []))
    command.extend(str(arg) for arg in experiment.get("extra_server_args", []))
    return command


def _request_text(url: str, api_key: str | None = None, timeout: float = 10) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def wait_for_server(base_url: str, timeout_seconds: float, api_key: str | None) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _request_text(f"{base_url}/v1/models", api_key=api_key, timeout=5)
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"SGLang did not become ready within {timeout_seconds}s: {last_error}")


def _gpu_monitor(output: Path, stop: threading.Event) -> None:
    fields = "timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw"
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(fields + "\n")
        while not stop.wait(1.0):
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                if completed.returncode == 0:
                    handle.write(completed.stdout)
                    handle.flush()
            except (FileNotFoundError, subprocess.SubprocessError):
                return


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=env["BFCL_PROJECT_ROOT"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode, time.perf_counter() - start


def run_experiment(
    config: LoadedConfig,
    experiment_name: str,
    artifact_dir: Path | None = None,
    keep_server: bool = False,
) -> dict[str, Any]:
    experiment = config.experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact = (artifact_dir or config.root / "artifacts" / f"{timestamp}-{experiment_name}").resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    bfcl_root = artifact / "bfcl"
    bfcl_root.mkdir()

    run_ids = resolve_from_root(config, config.benchmark["run_ids_file"])
    if not run_ids.is_file():
        raise FileNotFoundError(
            f"Missing {run_ids}. Run scripts/select_bfcl_subset.py after installing bfcl-eval."
        )
    shutil.copy2(run_ids, bfcl_root / "test_case_ids_to_generate.json")

    host = str(config.study["host"])
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    port = int(config.study["port"])
    base_url = f"http://{client_host}:{port}"
    api_key = os.environ.get("SGLANG_API_KEY") or None
    server_command = build_server_command(config, experiment_name)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(experiment["cuda_visible_devices"]),
            "BFCL_PROJECT_ROOT": str(bfcl_root),
            "LOCAL_SERVER_ENDPOINT": client_host,
            "LOCAL_SERVER_PORT": str(port),
            "REMOTE_OPENAI_BASE_URL": f"{base_url}/v1",
            "REMOTE_OPENAI_API_KEY": api_key or "EMPTY",
            "REMOTE_OPENAI_TOKENIZER_PATH": str(config.study["model_path"]),
        }
    )

    metadata: dict[str, Any] = {
        "experiment": experiment_name,
        "description": experiment.get("description", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.path),
        "study": config.study,
        "benchmark": config.benchmark,
        "experiment_config": experiment,
        "server_command": server_command,
        "environment": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
    }
    metadata_path = artifact / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    server_log = (artifact / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_command,
        cwd=config.root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=_gpu_monitor, args=(artifact / "gpu_samples.csv", monitor_stop), daemon=True
    )

    try:
        wait_for_server(
            base_url,
            float(config.study.get("startup_timeout_seconds", 900)),
            api_key,
        )
        monitor.start()
        (artifact / "metrics_before.prom").write_text(
            _request_text(f"{base_url}/metrics", api_key=api_key), encoding="utf-8"
        )

        benchmark_command = [
            sys.executable,
            str(config.root / "scripts" / "run_bfcl_instrumented.py"),
            "--model",
            str(config.benchmark["model_name"]),
            "--run-ids",
            "--skip-server-setup",
            "--num-threads",
            str(config.benchmark["concurrency"]),
            "--temperature",
            str(config.benchmark.get("temperature", 0.001)),
            "--allow-overwrite",
        ]
        benchmark_code, benchmark_duration = _run_logged(
            benchmark_command, artifact / "bfcl_generate.log", env
        )
        metadata["benchmark_command"] = benchmark_command
        metadata["benchmark_return_code"] = benchmark_code
        metadata["benchmark_duration_seconds"] = benchmark_duration

        (artifact / "metrics_after.prom").write_text(
            _request_text(f"{base_url}/metrics", api_key=api_key), encoding="utf-8"
        )

        if config.benchmark.get("evaluate", True) and benchmark_code == 0:
            run_id_categories = list(
                json.loads((bfcl_root / "test_case_ids_to_generate.json").read_text(encoding="utf-8"))
            )
            evaluation_command = [
                "bfcl",
                "evaluate",
                "--model",
                str(config.benchmark["model_name"]),
                "--test-category",
                ",".join(run_id_categories),
                "--partial-eval",
            ]
            evaluation_code, evaluation_duration = _run_logged(
                evaluation_command, artifact / "bfcl_evaluate.log", env
            )
            metadata["evaluation_command"] = evaluation_command
            metadata["evaluation_return_code"] = evaluation_code
            metadata["evaluation_duration_seconds"] = evaluation_duration

        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summarize_artifact(artifact)
    finally:
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=5)
        if not keep_server:
            _stop_process(server)
        server_log.close()
