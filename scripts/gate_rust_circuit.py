"""Gate for C2: can the Rust kernel carry the selected circuit, and does it agree
with the NumPy reference on this graph?

The pre-registration requires the circuit to clear the same differential test the
128-neuron benchmark cleared. This script is that gate. It measures, on the
actual selected circuit and not on a toy graph:

  1. whether the Rust bridge budget accepts the circuit at all;
  2. wall-clock per step, which decides whether C2 is feasible at any N;
  3. agreement with the NumPy reference under identical prerecorded input.

A failure here is a result to report, not a threshold to lower.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.data import Connectome  # noqa: E402
from flybrain.model.network import LIFNetwork  # noqa: E402
from flybrain.model.rust import RecordingOptions, RustBrain  # noqa: E402

CIRCUIT = ROOT / "outputs/neural_c2/circuit_sugar_grn_mn9_h2.npz"
CARD = ROOT / "outputs/neural_c2/circuit_card.json"
OUT = ROOT / "outputs/neural_c2"

# delta_v is large enough to actually spike a cell but not to blow the state up.
DELTA_V_MV = 15.0
DELTA_G_MV = 0.0


def build_connectome() -> Connectome:
    z = np.load(CIRCUIT)
    return Connectome(
        dataset_id="v783_circuit_sugar_grn_mn9_h2",
        flywire_ids=z["flywire_ids"].astype(np.int64),
        pre=z["pre"].astype(np.int32),
        post=z["post"].astype(np.int32),
        signed_count=z["signed_count"].astype(np.int64),
    )


def make_input(conn: Connectome, n_events: int, span: int, seed: int) -> list:
    """Prerecorded stimulus, in the shape RustBrain.advance expects.

    The Rust core rejects unsorted blocks: lib.rs requires every event step to be
    strictly below the block length, every neuron index in range, both deltas
    finite, and the sequence sorted by step (non-decreasing). Satisfying all four
    is a caller obligation, not something the core repairs.
    """
    rng = np.random.default_rng(seed)
    pre_sorted = np.sort(np.unique(conn.pre))
    pick = rng.choice(pre_sorted, size=n_events, replace=True)
    steps = rng.integers(0, span, size=n_events)
    order = np.argsort(steps, kind="stable")
    events = [(int(steps[i]), int(pick[i]), DELTA_V_MV, DELTA_G_MV) for i in order]
    assert all(events[i][0] <= events[i + 1][0] for i in range(len(events) - 1)), "unsorted"
    assert all(0 <= e[0] < span and 0 <= e[1] < conn.n_neurons for e in events), "out of range"
    return events


def main() -> int:
    card = json.loads(CARD.read_text(encoding="utf-8"))
    conn = build_connectome()
    n, e = conn.n_neurons, len(conn.pre)
    print(f"circuito : {n:,} neurons  {e:,} edges  "
          f"({card['synapses_total']:,} sinapses)")
    print(f"budget estimado: {e*256 + n*(160+8*18):,} bytes")
    print()

    # ---- 1. does Rust accept it? -------------------------------------------
    try:
        rust = RustBrain(conn, dt_ms=0.1, reference_order=True)
    except Exception as exc:  # noqa: BLE001
        print(f"PORTAO: Rust RECUSOU o circuito: {type(exc).__name__}: {exc}")
        (OUT / "gate_rust.json").write_text(
            json.dumps({"accepted": False, "error": str(exc)}, indent=2), encoding="utf-8")
        return 2
    print(f"PORTAO 1: Rust aceitou. estimated_native_bytes = {rust.estimated_native_bytes:,}")
    print()

    events = make_input(conn, n_events=400, span=2000, seed=1)
    steps = 2000

    # ---- 2. wall clock ------------------------------------------------------
    rust.reset()
    t0 = time.perf_counter()
    res = rust.advance(steps, inputs=events, recording_options=RecordingOptions(max_spikes=0))
    rust_secs = time.perf_counter() - t0
    ms = 1000.0 * rust_secs / steps
    print(f"PORTAO 2: {steps} passos em {rust_secs:.2f}s = {ms:.4f} ms/passo")
    print(f"          total_spikes={res.get('total_spikes')}  "
          f"native_bytes={rust.estimated_native_bytes:,}")
    print()

    # ---- 3. agreement with NumPy, using the validated protocol ---------------
    # The protocol is copied from tests/test_rust_differential.py, which passes at
    # 16 neurons: identical starting state, one step at a time, the SAME prerecorded
    # voltage/conductance table fed to both, and full state compared every step.
    # Inventing a protocol here would test nothing; this one is already trusted.
    from flybrain.runtime.backend import NumpyBackend  # noqa: PLC0415

    class RecordedVoltage:
        def __init__(self, table, dt):
            self.table, self.dt = table, dt
            self.indices = np.arange(table.shape[1], dtype=np.int32)

        def counts(self, t):
            return self.table[round(t / self.dt)]

    steps = 200
    rng = np.random.default_rng(904)
    p_nonneg = 0.02
    voltage = (rng.random((steps, n)) < p_nonneg).astype(np.float32) * np.float32(68.75)
    conductance = rng.integers(-2, 3, (steps, n)).astype(np.float32) * np.float32(0.275)

    ref = LIFNetwork(conn, backend=NumpyBackend(), dt_ms=0.1)
    rust2 = RustBrain(conn, dt_ms=0.1, reference_order=True)

    ref.v[:] = rng.uniform(-53, -44, n)
    ref.g[:] = rng.uniform(-10, 20, n)
    ref.tau_ref[::4] = 0
    ref.t_last[1::5] = 0
    rust2.set_state(ref.v, ref.g, ref.t_last, ref.tau_ref)
    ref._samplers = [RecordedVoltage(voltage, 0.1)]
    ref.stim_weight = np.array(1, np.float32)

    first_bad = None
    n_spikes_ref = 0
    n_spikes_rust = 0
    for tick in range(steps):
        ref.step()
        ref.g += conductance[tick]
        ev = [
            (0, i, float(voltage[tick, i]), float(conductance[tick, i]))
            for i in range(n)
            if voltage[tick, i] != 0.0 or conductance[tick, i] != 0.0
        ]
        rust2.advance(1, ev)
        st = rust2.state()
        n_spikes_ref += int(np.count_nonzero(ref.last_spike))
        n_spikes_rust += int(np.count_nonzero(st["last_spike"]))
        if first_bad is None:
            for field in ("v", "g", "t_last", "tau_ref"):
                want = np.asarray(getattr(ref, field), dtype=np.float32)
                a = st[field].view(np.uint32)
                b = want.view(np.uint32)
                bad = np.flatnonzero(a != b)
                if len(bad):
                    first_bad = (tick, field, int(bad[0]),
                                 float(st[field][bad[0]]), float(want[bad[0]]))
                    break
            else:
                if not np.array_equal(st["last_spike"], ref.last_spike):
                    first_bad = (tick, "last_spike", -1, 0, 0)

    print(f"PORTAO 3: Agreement on {steps} passos, protocolo de tests/test_rust_differential.py")
    print(f"          spikes por passo (numpy) ~ {n_spikes_ref}, (rust) ~ {n_spikes_rust}")
    if first_bad is None:
        print("          ESTADO IDENTICO bit a bit em todos os passos e todos os campos")
        identical = True
    else:
        tick, field, idx, rv, nv = first_bad
        print(f"          primeira divergencia: passo {tick} campo {field} indice {idx}")
        print(f"            rust = {rv!r}   numpy = {nv!r}")
        identical = False
    print(f"          eventos de entrada nao nulos: {int((voltage != 0).sum())} de voltagem, "
          f"{int((conductance != 0).sum())} de condutancia")

    out = {
        "accepted": True,
        "circuit": {"neurons": n, "edges": e, "hops": card["hops"]},
        "rust": {
            "estimated_native_bytes": rust.estimated_native_bytes,
            "steps": steps, "seconds": rust_secs, "ms_per_step": ms,
            "total_spikes": res.get("total_spikes"),
        },
        "differential": {
            "steps": steps,
            "protocol": "mirrors tests/test_rust_differential.py",
            "nonzero_voltage_events": int((voltage != 0).sum()),
            "nonzero_conductance_events": int((conductance != 0).sum()),
            "bit_identical": identical,
            "first_divergence": first_bad,
        },
    }
    (OUT / "gate_rust.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngravado em {OUT / 'gate_rust.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
