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
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

from .config import LoadedConfig, resolve_from_root
from .summary import summarize_artifact


@dataclass(frozen=True)
class ServerSpec:
    replica_index: int
    cuda_visible_devices: str
    port: int
    command: list[str]


def build_server_command(
    config: LoadedConfig, experiment_name: str, port: int | None = None
) -> list[str]:
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
        str(study["port"] if port is None else port),
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


def build_server_specs(config: LoadedConfig, experiment_name: str) -> list[ServerSpec]:
    experiment = config.experiment(experiment_name)
    replica_count = int(experiment.get("replica_count", 1))
    frontend_port = int(config.study["port"])
    if replica_count == 1:
        return [
            ServerSpec(
                replica_index=0,
                cuda_visible_devices=str(experiment["cuda_visible_devices"]),
                port=frontend_port,
                command=build_server_command(config, experiment_name),
            )
        ]

    gpu_ids = [value.strip() for value in str(experiment["cuda_visible_devices"]).split(",")]
    configured_ports = experiment.get("replica_ports")
    ports = (
        [int(value) for value in configured_ports]
        if configured_ports is not None
        else [frontend_port + index + 1 for index in range(replica_count)]
    )
    return [
        ServerSpec(
            replica_index=index,
            cuda_visible_devices=gpu_ids[index],
            port=ports[index],
            command=build_server_command(config, experiment_name, port=ports[index]),
        )
        for index in range(replica_count)
    ]


def _request_text(url: str, api_key: str | None = None, timeout: float = 10) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _ReplicaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.server.forward(self)  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        self.server.forward(self)  # type: ignore[attr-defined]

    def do_HEAD(self) -> None:
        self.server.forward(self)  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        self.server.write_log(format % args)  # type: ignore[attr-defined]


class _ReplicaProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        backends: list[str],
        log_handle: TextIO,
        api_key: str | None,
    ) -> None:
        super().__init__(address, _ReplicaProxyHandler)
        self.backends = backends
        self.log_handle = log_handle
        self.api_key = api_key
        self._next_backend = 0
        self._lock = threading.Lock()

    def write_log(self, message: str) -> None:
        with self._lock:
            self.log_handle.write(message + "\n")
            self.log_handle.flush()

    def _select_backend(self) -> str:
        with self._lock:
            backend = self.backends[self._next_backend]
            self._next_backend = (self._next_backend + 1) % len(self.backends)
        return backend

    def _send_response(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        headers: Any,
        body: bytes,
    ) -> None:
        handler.send_response(status)
        for key, value in headers.items():
            if key.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}:
                handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(body)

    def _aggregate_metrics(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            texts = [
                _request_text(f"{backend}/metrics", api_key=self.api_key)
                for backend in self.backends
            ]
        except (OSError, urllib.error.URLError) as exc:
            handler.send_error(502, f"Failed to collect replica metrics: {exc}")
            return
        body = ("\n".join(texts) + "\n").encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(body)

    def forward(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path.split("?", 1)[0] == "/metrics":
            self._aggregate_metrics(handler)
            return

        backend = self._select_backend()
        content_length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(content_length) if content_length else None
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        request = urllib.request.Request(
            f"{backend}{handler.path}",
            data=body,
            headers=headers,
            method=handler.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                response_body = response.read()
                self._send_response(handler, response.status, response.headers, response_body)
            self.write_log(f"{handler.command} {handler.path} -> {backend}")
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            self._send_response(handler, exc.code, exc.headers, response_body)
            self.write_log(f"{handler.command} {handler.path} -> {backend} ({exc.code})")
        except (OSError, urllib.error.URLError) as exc:
            handler.send_error(502, f"Replica request failed for {backend}: {exc}")
            self.write_log(f"{handler.command} {handler.path} -> {backend} failed: {exc}")


class ReplicaProxy:
    def __init__(
        self,
        host: str,
        port: int,
        backends: list[str],
        log_handle: TextIO,
        api_key: str | None,
    ) -> None:
        self.server = _ReplicaProxyServer((host, port), backends, log_handle, api_key)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def wait_for_server(
    base_url: str,
    timeout_seconds: float,
    api_key: str | None,
    process: subprocess.Popen[Any] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited during startup with code {process.returncode}. "
                "Inspect server.log in the artifact directory."
            )
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
    server_specs = build_server_specs(config, experiment_name)
    if keep_server and len(server_specs) > 1:
        raise ValueError("--keep-server is not supported for multi-replica experiments")
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
        "environment": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
    }
    if len(server_specs) == 1:
        metadata["server_command"] = server_specs[0].command
    else:
        metadata["server_commands"] = [
            {
                "replica_index": spec.replica_index,
                "cuda_visible_devices": spec.cuda_visible_devices,
                "port": spec.port,
                "command": spec.command,
            }
            for spec in server_specs
        ]
        metadata["load_balancer"] = {
            "strategy": "round_robin_per_request",
            "host": host,
            "port": port,
            "backend_urls": [f"http://{client_host}:{spec.port}" for spec in server_specs],
            "metrics_aggregation": "sum_all_replicas",
        }
    metadata_path = artifact / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    servers: list[subprocess.Popen[Any]] = []
    server_logs: list[TextIO] = []
    proxy: ReplicaProxy | None = None
    proxy_log: TextIO | None = None
    for spec in server_specs:
        log_name = (
            "server.log"
            if len(server_specs) == 1
            else f"server_replica_{spec.replica_index}.log"
        )
        log_handle = (artifact / log_name).open("w", encoding="utf-8")
        server_env = env.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
        try:
            server = subprocess.Popen(
                spec.command,
                cwd=config.root,
                env=server_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
        except BaseException:
            log_handle.close()
            for running_server in servers:
                _stop_process(running_server)
            for running_log in server_logs:
                running_log.close()
            raise
        servers.append(server)
        server_logs.append(log_handle)
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=_gpu_monitor, args=(artifact / "gpu_samples.csv", monitor_stop), daemon=True
    )

    try:
        timeout_seconds = float(config.study.get("startup_timeout_seconds", 900))
        backend_urls = [f"http://{client_host}:{spec.port}" for spec in server_specs]
        for backend_url, server in zip(backend_urls, servers):
            wait_for_server(backend_url, timeout_seconds, api_key, process=server)
        if len(server_specs) > 1:
            proxy_log = (artifact / "load_balancer.log").open("w", encoding="utf-8")
            proxy = ReplicaProxy(host, port, backend_urls, proxy_log, api_key)
            proxy.start()
            wait_for_server(base_url, 30, api_key)
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
        if proxy is not None:
            proxy.stop()
        if not keep_server:
            for server in servers:
                _stop_process(server)
        for log_handle in server_logs:
            log_handle.close()
        if proxy_log is not None:
            proxy_log.close()
