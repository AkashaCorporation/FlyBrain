"""Backend abstraction and hardware detection.

Two things are kept strictly separate here, because conflating them is how a
project ends up lying about its performance:

* **What hardware exists** (``HardwareInfo``): the OS, CPU, RAM, and any NVIDIA
  GPU the machine reports. Detected by querying the system, never assumed.
* **What the running process can actually use** (``Backend``): the array module
  the simulation executes in. A machine can have a GPU that the active backend
  cannot touch, and FlyBrain says so instead of printing "CUDA".

Verified experimentally on the development machine (GTX 1650, driver 591.86,
CUDA 13.1, Windows 11): JAX reports ``default_backend() == 'cpu'`` and
``devices() == [CpuDevice(id=0)]``, because CUDA-enabled jaxlib wheels are
published for Linux only. ``docs/ARCHITECTURE.md`` records this.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..errors import FlyBrainError

# --------------------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------------------


def _total_ram_gb() -> Optional[float]:
    """Total physical RAM in GB, or None if it cannot be determined."""
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / 1e9, 2)
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(re.search(r"(\d+)", line).group(1))  # type: ignore[union-attr]
                        return round(kb * 1024 / 1e9, 2)
        except Exception:
            pass
    return None


def _cpu_name() -> Optional[str]:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            for line in out.stdout.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    return line
        except Exception:
            pass
    try:
        return platform.processor() or None
    except Exception:
        return None


def available_ram_gb() -> Optional[float]:
    """Currently *available* (not total) RAM in GB, or None if undeterminable.

    This is the number the feasibility gate must use. Total RAM is the wrong
    quantity: a machine with 16 GB total and 2 GB free will thrash or die on a
    simulation sized against 16 GB.
    """
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullAvailPhys / 1e9, 2)
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        kb = int(re.search(r"(\d+)", line).group(1))  # type: ignore[union-attr]
                        return round(kb * 1024 / 1e9, 2)
        except Exception:
            pass
    return None


def query_nvidia_gpus() -> List[Dict[str, Any]]:
    """Ask ``nvidia-smi`` about installed NVIDIA GPUs.

    Returns an empty list if the tool is absent or fails. Presence of a GPU here
    does **not** mean the active backend can use it; see :class:`HardwareInfo`.
    """
    fields = "name,memory.total,memory.free,driver_version,compute_cap"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    gpus: List[Dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "name": parts[0],
                    "vram_total_mb": int(float(parts[1])),
                    "vram_free_mb": int(float(parts[2])),
                    "driver_version": parts[3],
                    "compute_capability": parts[4],
                }
            )
        except ValueError:
            continue
    return gpus


@dataclass
class HardwareInfo:
    """A factual snapshot of the machine. Every field may be None: unknown is a
    legitimate answer and is never replaced by a guess."""

    os_name: str
    os_version: str
    machine: str
    cpu_name: Optional[str]
    cpu_cores_physical: Optional[int]
    cpu_cores_logical: Optional[int]
    ram_total_gb: Optional[float]
    python_version: str
    python_executable: str
    python_implementation: str
    nvidia_gpus: List[Dict[str, Any]] = field(default_factory=list)
    jax_installed: bool = False
    jax_version: Optional[str] = None
    jax_backend: Optional[str] = None
    jax_devices: List[str] = field(default_factory=list)
    jax_gpu_available: bool = False
    jax_import_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- derived, stated plainly ---------------------------------------------------
    @property
    def gpu_names(self) -> List[str]:
        return [g["name"] for g in self.nvidia_gpus]

    @property
    def vram_total_mb(self) -> Optional[int]:
        if not self.nvidia_gpus:
            return None
        return int(sum(g["vram_total_mb"] for g in self.nvidia_gpus))

    @property
    def vram_free_mb(self) -> Optional[int]:
        if not self.nvidia_gpus:
            return None
        return int(sum(g["vram_free_mb"] for g in self.nvidia_gpus))

    def summary_lines(self) -> List[str]:
        lines = [
            f"OS: {self.os_name} {self.os_version} ({self.machine})",
            f"CPU: {self.cpu_name or 'unknown'} "
            f"({self.cpu_cores_physical or '?'} cores / {self.cpu_cores_logical or '?'} threads)",
            f"RAM: {self.ram_total_gb if self.ram_total_gb is not None else 'unknown'} GB",
            f"Python: {self.python_version.split()[0]} ({self.python_implementation})",
        ]
        if self.nvidia_gpus:
            for g in self.nvidia_gpus:
                lines.append(
                    f"NVIDIA GPU: {g['name']} - {g['vram_total_mb']} MB VRAM "
                    f"({g['vram_free_mb']} MB free), driver {g['driver_version']}, "
                    f"compute {g['compute_capability']}"
                )
        else:
            lines.append("NVIDIA GPU: none detected via nvidia-smi")
        if self.jax_installed:
            lines.append(
                f"JAX: {self.jax_version} backend={self.jax_backend} "
                f"devices={self.jax_devices} gpu_available={self.jax_gpu_available}"
            )
        else:
            lines.append(f"JAX: not installed ({self.jax_import_error})")
        return lines


def detect_hardware(probe_jax: bool = True) -> HardwareInfo:
    """Detect the machine. Never raises; unknown values stay None."""
    cores_phys = cores_log = None
    try:
        cores_log = os.cpu_count()
        cores_phys = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    except Exception:
        pass
    if cores_phys is None:
        cores_phys = cores_log if sys.platform != "win32" else None
    if sys.platform == "win32" and cores_phys is None:
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "numberofcores"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            for line in out.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    cores_phys = int(line)
        except Exception:
            pass

    hw = HardwareInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        cpu_name=_cpu_name(),
        cpu_cores_physical=cores_phys,
        cpu_cores_logical=cores_log,
        ram_total_gb=_total_ram_gb(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_implementation=platform.python_implementation(),
        nvidia_gpus=query_nvidia_gpus(),
    )

    if probe_jax:
        try:
            import jax  # type: ignore

            hw.jax_installed = True
            hw.jax_version = getattr(jax, "__version__", "unknown")
            hw.jax_backend = jax.default_backend()
            devices = jax.devices()
            hw.jax_devices = [f"{d.platform}:{d.device_kind}#{i}" for i, d in enumerate(devices)]
            hw.jax_gpu_available = any(d.platform in ("gpu", "cuda") for d in devices)
        except ImportError as exc:
            hw.jax_import_error = f"ImportError: {exc}"
        except Exception as exc:  # JAX can raise non-ImportError problems at probe time
            hw.jax_import_error = f"{type(exc).__name__}: {exc}"

    return hw


# --------------------------------------------------------------------------------------
# Array backends
# --------------------------------------------------------------------------------------


class Backend:
    """The minimal array-module contract the simulation kernel needs.

    Everything else in the kernel is written in terms of ``self.xp``, which both
    NumPy and ``jax.numpy`` satisfy. Only the three operations below genuinely
    differ between the two, so only those are abstracted. This is deliberately
    small: a large abstraction layer would be its own source of bugs.
    """

    name: str = "?"
    is_gpu: bool = False

    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        self.xp = np

    # -- dtype helpers ------------------------------------------------------------
    @property
    def float_dtype(self):
        return np.dtype(self.dtype)

    def asarray(self, x, dtype=None):
        """Convert to a backend array.

        ``dtype=None`` preserves the input dtype. This matters: a default that
        silently coerced everything to the float dtype would turn integer index
        arrays into floats, which fails at the first fancy-index lookup. Index
        arrays are int32 and weight arrays are float32, and both must stay that way.
        """
        raise NotImplementedError

    def zeros(self, n, dtype=None):
        """Zero-filled array; ``dtype=None`` means "the configured float dtype"."""
        raise NotImplementedError

    # -- the three primitive differences -----------------------------------------
    def scatter_add(self, size: int, index, values, dtype=None):
        """Return an array of length ``size`` where ``out[index[k]] += values[k]``."""
        raise NotImplementedError

    def add_at(self, dest, index, values):
        """Return ``dest`` with ``values`` added at ``index`` (even if index repeats)."""
        raise NotImplementedError

    def set_row(self, ring, i: int, row):
        """Return ``ring`` with row ``i`` set to ``row``."""
        raise NotImplementedError

    # -- compilation --------------------------------------------------------------
    def compile(self, fn):
        """Return a possibly-compiled version of ``fn``. Identity for NumPy."""
        return fn

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "dtype": self.dtype, "is_gpu": self.is_gpu}


class NumpyBackend(Backend):
    """Reference backend. Always available; the semantics every other backend must match."""

    name = "numpy"
    is_gpu = False

    def asarray(self, x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def zeros(self, n, dtype=None):
        return np.zeros(n, dtype=dtype or self.dtype)

    def scatter_add(self, size, index, values, dtype=None):
        dt = dtype or self.dtype
        if index.size == 0:
            return np.zeros(size, dtype=dt)
        # bincount is NumPy's fastest scatter-add; it accumulates in float64, so the
        # result is narrowed back to the requested precision explicitly rather than
        # silently promoting the whole state to float64.
        acc = np.bincount(index, weights=values, minlength=size)
        return acc.astype(dt, copy=False)

    def add_at(self, dest, index, values):
        if np.size(index) == 0:
            return dest
        np.add.at(dest, index, values)
        return dest

    def set_row(self, ring, i, row):
        ring[i] = row
        return ring


class JaxBackend(Backend):
    """``jax.numpy`` backend.

    On a CPU-only JAX install this is a real backend, not a fallback pretence: it
    executes the same kernel through XLA. ``is_gpu`` reflects what JAX actually
    reports, so a CPU-only JAX is never labelled as acceleration.
    """

    name = "jax"

    def __init__(self, dtype: str = "float32", probe: bool = True):
        import jax  # noqa: F401  (import errors surface to the caller)

        super().__init__(dtype)
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        # float64 must stay off: the model is specified in float32 and enabling
        # x64 would silently change every number, and double memory traffic.
        jax.config.update("jax_enable_x64", False)
        self.xp = jnp
        self.is_gpu = False
        if probe:
            try:
                self.is_gpu = any(d.platform in ("gpu", "cuda") for d in jax.devices())
            except Exception:
                self.is_gpu = False

    def asarray(self, x, dtype=None):
        return self.jnp.asarray(x, dtype=dtype)

    def zeros(self, n, dtype=None):
        return self.jnp.zeros(n, dtype=dtype or self.dtype)

    def scatter_add(self, size, index, values, dtype=None):
        dt = dtype or self.dtype
        out = self.jnp.zeros(size, dtype=dt)
        if index.size == 0:
            return out
        return out.at[index].add(values)

    def add_at(self, dest, index, values):
        if np.size(index) == 0:
            return dest
        return dest.at[index].add(values)

    def set_row(self, ring, i, row):
        return ring.at[i].set(row)

    def compile(self, fn):
        return self.jax.jit(fn)


def available_backends() -> List[str]:
    names = ["numpy"]
    if importlib.util.find_spec("jax") is not None:
        names.append("jax")
    return names


def get_backend(name: str = "auto", dtype: str = "float32") -> Backend:
    """Resolve a backend by name.

    ``auto`` prefers **NumPy**, which is measured faster than JAX on CPU for this
    kernel: the hot operation is a wide gather plus a scatter-add over 14.7 M edges,
    and NumPy's ``bincount`` beats XLA's scatter on that shape. Measured on the
    development machine over a 4 000-neuron / 126 267-edge subset, 1000 steps:

    ========  ==============
    backend   ms per step
    ========  ==============
    numpy     1.78
    jax       3.15
    ========  ==============

    Both backends produce **bit-identical** spike counts for the same seed (asserted
    in ``tests/test_runtime_backend.py``), so this is purely a performance choice.
    JAX remains available via ``--backend jax``, and on a machine where a real GPU
    is reachable by JAX it would win outright - but that case is decided by
    measurement, not by assuming a GPU is fast.
    """
    name = (name or "auto").lower()
    if name == "numpy":
        return NumpyBackend(dtype)
    if name == "jax":
        try:
            return JaxBackend(dtype)
        except ImportError as exc:
            raise FlyBrainError(
                f"backend 'jax' was requested but JAX is not importable: {exc}. "
                f"Install it with `pip install jax`, or use --backend numpy."
            ) from exc
    if name == "auto":
        return NumpyBackend(dtype)
    raise FlyBrainError(f"unknown backend {name!r}; available: {available_backends()} + ['auto']")
