"""Shared pytest fixtures.

Nothing here mocks FlyBrain's own code. Where a test needs a controlled network it
builds one from an explicit edge list with hand-computable behaviour, which is the
opposite of mocking: the expected answer is derived independently of the
implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import dataset_available  # noqa: E402

from flybrain.data.synthetic import chain_connectome  # noqa: E402
from flybrain.model import LIFNetwork, LIFParams  # noqa: E402
from flybrain.runtime.backend import NumpyBackend  # noqa: E402

needs_dataset = pytest.mark.skipif(
    not dataset_available(),
    reason="staged FlyWire dataset not present (run `flybrain datasets --stage`)",
)


@pytest.fixture(scope="session")
def numpy_backend():
    return NumpyBackend("float32")


@pytest.fixture
def tiny_chain():
    return chain_connectome(4, synapses=60)


@pytest.fixture
def params():
    return LIFParams()


@pytest.fixture
def net(tiny_chain, params, numpy_backend):
    return LIFNetwork(tiny_chain, params, dt_ms=0.1, backend=numpy_backend, seed=0)


@pytest.fixture(scope="session")
def real_connectome():
    """The staged v630 dataset, loaded once per session."""
    from flybrain import config as fb_config
    from flybrain.data import load_connectome

    return load_connectome(
        "flywire_630",
        fb_config.RAW_DIR,
        fb_config.PROCESSED_DIR,
        fb_config.METADATA_DIR,
    )
