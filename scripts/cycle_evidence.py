"""Local first-cycle provenance and bounded command runner (requires psutil)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "first_cycle"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    try:
        p = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return {
            "command": args,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": args, "error": str(e)}


def identity():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": command(["git", "rev-parse", "HEAD"]),
        "status": command(["git", "status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "memory": psutil.virtual_memory()._asdict(),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for folder in ("flybrain", "tests", "scripts", "crates")
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".rs", ".toml")
        },
    }


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def inventory():
    data = identity()
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    paths = tracked + ["docs/PROPOSTA_PESQUISA.md"]
    paths += [
        str(p.relative_to(ROOT))
        for p in (ROOT / "data").rglob("*")
        if p.is_file() and p.suffix in (".npz", ".parquet", ".csv")
    ]
    paths += [
        str(p.relative_to(ROOT))
        for p in (ROOT / "outputs").rglob("*")
        if p.is_file()
        and "first_cycle" not in p.parts
        and p.suffix in (".json", ".csv", ".txt")
    ]
    data["files"] = {
        p: {"sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size}
        for p in paths
        if (ROOT / p).is_file()
    }
    cargo_bin = Path.home() / ".cargo" / "bin"
    data["toolchain"] = [
        command([str(cargo_bin / x), "--version"]) for x in ("rustc.exe", "cargo.exe")
    ]
    data["hardware"] = command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json",
        ]
    )
    data["gpu"] = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    data.update(
        requested_mode="preserve_v0",
        effective_mode="preserve_v0",
        completed=True,
        stop_reason="finished",
        evidence_level="file_hashes_and_inventory",
        seeds=None,
        dataset="flywire_630",
        mask=None,
        historical_spike_digest="d28feb6b82d1108056718dd246480cf4061c2c83bc14241779213941541f3760",
        historical_digest_status="recorded_from_ESTADO_ATUAL_not_rerun",
    )
    save(OUT / "initial_manifest.json", data)
    archive = OUT / "v0-30ad237.zip"
    if archive.exists():
        raise FileExistsError(archive)
    subprocess.run(
        ["git", "archive", "--format=zip", "-o", str(archive), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    save(OUT / "reference_archive.json", {"file": str(archive), "sha256": sha(archive)})
    print("Preserved initial manifest and source archive", flush=True)


def run(args):
    name = args.name
    log = OUT / (name + ".log")
    report = OUT / (name + ".json")
    if log.exists() or report.exists():
        raise FileExistsError(name)
    data = identity()
    data.update(
        command=args.command,
        requested_mode=args.mode,
        effective_mode=args.mode,
        seeds="specified by command/test fixtures",
        dataset="specified by command/test fixtures",
        mask=None,
        evidence_level="command_execution",
        completed=False,
        stop_reason="running",
        limits={"seconds": args.seconds, "memory_mb": args.memory_mb, "log_mb": 10},
    )
    available = psutil.virtual_memory().available
    if args.memory_mb * 1e6 > available * 0.6:
        data.update(stop_reason="requested_memory_exceeds_60_percent_available")
        save(report, data)
        print(
            json.dumps(
                {
                    "completed": False,
                    "stop_reason": data["stop_reason"],
                    "available_bytes": available,
                    "requested_memory_mb": args.memory_mb,
                }
            )
        )
        return 2
    start = time.perf_counter()
    peak = 0
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with log.open("xb") as f:
        p = subprocess.Popen(
            args.command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env
        )
        proc = psutil.Process(p.pid)
        reason = "finished"
        while p.poll() is None:
            try:
                family = [proc] + proc.children(recursive=True)
                rss = sum(x.memory_info().rss for x in family if x.is_running())
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - start > args.seconds:
                reason = "wall_limit"
            elif peak > args.memory_mb * 1e6:
                reason = "memory_limit"
            elif log.stat().st_size > 10e6:
                reason = "log_limit"
            if reason != "finished":
                for child in reversed(family):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(0.1)
        p.wait()
    data.update(
        returncode=p.returncode,
        completed=reason == "finished" and p.returncode == 0,
        stop_reason=reason
        if reason != "finished" or p.returncode == 0
        else "command_failed",
        elapsed_seconds=time.perf_counter() - start,
        sampled_peak_tree_rss_bytes=peak,
        log_sha256=sha(log),
        log_bytes=log.stat().st_size,
    )
    save(report, data)
    print(
        json.dumps(
            {
                k: data[k]
                for k in (
                    "completed",
                    "stop_reason",
                    "returncode",
                    "elapsed_seconds",
                    "sampled_peak_tree_rss_bytes",
                )
            }
        )
    )
    print(log.read_text(encoding="utf-8", errors="replace")[-12000:])
    return 0 if data["completed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "run"])
    parser.add_argument("--name")
    parser.add_argument("--mode", default="validation")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--memory-mb", type=float, default=2500)
    args, remainder = parser.parse_known_args()
    args.command = remainder[1:] if remainder[:1] == ["--"] else remainder
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "inventory":
        inventory()
    else:
        if not args.name or not args.command:
            parser.error("run needs --name and a command after --")
        sys.exit(run(args))
