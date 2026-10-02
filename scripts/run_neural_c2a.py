"""C2a - executes docs/PREREGISTRATION_C2.md rev. 4.

Measures how much of a cue survives to the output population of the real
circuit, against a degree-preserving rewire and against a random-target control.

Read section 2 of the preregistration before reading this code. The experiment is
NOT a competition between conditions: A is the reference by definition, and what
is measured is how far B and C depart from A. The mandatory sanity check is that
a decoder trained on the controls and applied to A must FAIL, otherwise the
controls are not different enough to be controls.

The calibration touches condition A only. B and C are simulated after W is fixed.
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
from flybrain.experiments.rewire import compare_marginals, degree_preserving_rewire  # noqa: E402
from flybrain.model.rust import RecordingOptions, RustBrain  # noqa: E402

CIRCUIT = ROOT / "outputs/neural_c2/circuit_sugar_grn_mn9_h2.npz"
CARD = ROOT / "outputs/neural_c2/circuit_card.json"
ANN = Path(
    r"E:\HipoCampo\stack\third_party\flywire_annotations"
    r"\supplemental_files\Supplemental_file1_neuron_annotations.tsv"
)
OUT = ROOT / "outputs/neural_c2/c2a"

# --- fixed by the preregistration -------------------------------------------
N_SEEDS = 16
K_PATTERNS = 8
R_TRIALS = 4
MIN_ACTIVE = 3
# Stimulus protocol is the one the v0 whole-brain run uses, not an invented one:
# sugar_grn driven as Poisson spikes at STIM_RATE_HZ. A first attempt injected a
# 3-step voltage pulse and the circuit was measurably dead - 26 spikes per pattern
# across 10 201 neurons, and two different patterns produced bit-identical
# count vectors (nearest-pair distance 0.000000). Same lesson as the Rust gate:
# reuse the validated protocol, do not invent one.
STIM_RATE_HZ = 150.0
# Per-spike voltage jump, the value the v0 differential test drives with.
DELTA_V_MV = 68.75
REWIRE_SEED = 20261002
BOOTSTRAP_SEED = 20261002
BOOTSTRAP_B = 10_000
W_GRID = (60, 80, 100, 120, 140, 160, 180, 200, 250, 320, 400, 640,
          1000, 2000, 4000, 8000, 16000)
TARGET_ACC, TARGET_TOL = 0.75, 0.05
# The v0 default is 256 MB, and this circuit's conservative Python-side bridge
# estimate is 250.6 MB, which leaves under 5 MB for input conversion - not enough
# for ~24 000 events at 150 Hz over 8000 steps. The MEASURED native footprint is
# 64 164 368 bytes, so the estimate is roughly 4x pessimistic. Raising the budget
# is a resource decision with headroom to spare, not a scientific one; the
# selection of the circuit itself still respected the original 256 MB ceiling.
RUST_MEMORY_BYTES = 1_000_000_000


def utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_circuit() -> Connectome:
    z = np.load(CIRCUIT)
    return Connectome(
        dataset_id="v783_circuit_sugar_grn_mn9_h2",
        flywire_ids=z["flywire_ids"].astype(np.int64),
        pre=z["pre"].astype(np.int32),
        post=z["post"].astype(np.int32),
        signed_count=z["signed_count"].astype(np.int64),
    )


def load_annotations() -> dict[int, dict]:
    import csv
    out: dict[int, dict] = {}
    with open(ANN, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[int(row["root_id"])] = {
                "super_class": row.get("super_class", ""),
                "cell_type": row.get("cell_type", ""),
                "status": row.get("status", ""),
            }
    return out


def output_population(conn: Connectome, ann: dict[int, dict]) -> list[int]:
    """Indices inside the circuit of the descending and motor neurons.

    Same validity filter as the selection: cell_type present and status empty.
    """
    idx = []
    for i, fid in enumerate(conn.flywire_ids):
        r = ann.get(int(fid))
        if r is None or not r["cell_type"].strip() or r["status"].strip():
            continue
        if r["super_class"] in ("descending", "motor"):
            idx.append(i)
    return idx


def sensory_indices(conn: Connectome) -> list[int]:
    from flybrain.model.populations import PopulationRegistry
    reg = PopulationRegistry.builtin()
    lookup = {int(v): i for i, v in enumerate(conn.flywire_ids)}
    return sorted(lookup[int(x)] for x in reg.get("sugar_grn") if int(x) in lookup)


def make_patterns(rng: np.random.Generator, k: int, width: int) -> np.ndarray:
    """k binary patterns over `width` neurons, each with at least MIN_ACTIVE on."""
    pats = []
    while len(pats) < k:
        p = rng.integers(0, 2, width)
        if p.sum() >= MIN_ACTIVE and not any(np.array_equal(p, q) for q in pats):
            pats.append(p)
    return np.array(pats, dtype=np.int8)


def simulate(brain: RustBrain, pattern: np.ndarray, neurons: list[int],
             rng: np.random.Generator, w_steps: int, population: list[int]) -> np.ndarray:
    """One trial: drive the ON sensory neurons as Poisson spikes, run W steps,
    return the per-neuron firing count of every output neuron.

    The drive is the v0 protocol (sugar_grn at STIM_RATE_HZ) expressed as the
    prerecorded events the Rust core consumes. The core requires events sorted by
    step, in range, finite, with step below the block length.

    `population_counts` is ONE COUNTER PER POPULATION, not per neuron:
    flybrain-core builds it as vec![0; populations.len()], and the existing test
    pins it with populations=[[0], [1, 2]] -> [1, 0]. So to get a per-neuron
    feature vector the readout must be one SINGLETON population per output
    neuron. Passing the whole set as one population yields a single scalar, which
    is what a first version of this script did - and which silently reduced a
    655-dimensional representational measurement to one number per trial.
    """
    brain.reset()
    events: list[tuple[int, int, float, float]] = []
    p = STIM_RATE_HZ * 1e-3  # per step
    for n_i, on in zip(neurons, pattern):
        if not on:
            continue
        times = np.nonzero(rng.random(w_steps) < p)[0]
        for t in times.tolist():
            events.append((int(t), int(n_i), DELTA_V_MV, 0.0))
    events.sort(key=lambda e: e[0])
    res = brain.advance(
        w_steps, inputs=events,
        recording_options=RecordingOptions(max_spikes=0),
        populations=[[i] for i in population],
    )
    counts = np.asarray(res["population_counts"], dtype=np.float64)
    if counts.size != len(population):
        raise RuntimeError(
            f"readout returned {counts.size} counts for {len(population)} output "
            f"neurons; the per-neuron contract is broken, not the data"
        )
    return counts


def nearest_centroid_loo(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out nearest-centroid accuracy. x = features, y = class labels."""
    n = len(y)
    if n < 2 or len(set(y.tolist())) < 2:
        return float("nan")
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cents, labels = [], []
        for c in np.unique(y[mask]):
            cents.append(x[mask & (y == c)].mean(axis=0))
            labels.append(c)
        d = np.linalg.norm(np.array(cents) - x[i], axis=1)
        correct += int(labels[int(np.argmin(d))] == y[i])
    return correct / n


def bootstrap_ci(d: np.ndarray, b: int, seed: int, alpha: float = 0.05) -> dict:
    """Percentile bootstrap on the paired differences. Assumes no distribution."""
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = rng.integers(0, n, size=(b, n))
    means = d[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lower": float(lo), "upper": float(hi), "b": int(b), "seed": int(seed)}


def build_conditions(conn: Connectome) -> dict:
    """A real, B degree-preserving rewire, C random targets. Weights travel with
    the source, so each node keeps its own outgoing weight multiset exactly."""
    n = conn.n_neurons
    pre = np.asarray(conn.pre, dtype=np.int64)
    post = np.asarray(conn.post, dtype=np.int64)
    w = np.asarray(conn.signed_count, dtype=np.int64)
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    sign = np.where(w_s > 0, 1, -1).astype(np.int8)

    b_pre, b_post, _b_sign, b_rep = degree_preserving_rewire(
        pre_s, post_s, sign, n, seed=REWIRE_SEED)
    # argsort inside the rewire is a no-op on already-sorted input, so w_s still
    # lines up with the returned arrays; assert it rather than trust it.
    assert np.array_equal(b_pre, pre_s), "rewire changed the presynaptic order"
    B = Connectome("c2_B_degree_preserving", conn.flywire_ids,
                   b_pre.astype(np.int32), b_post.astype(np.int32), w_s)

    # C: degrees deliberately destroyed. It must be a free random draw, NOT a
    # permutation: np.random.Generator.permutation(post) preserves the multiset
    # of targets, so it preserves every in-degree exactly and is therefore
    # indistinguishable from B. Measured, not assumed: see the assert below.
    rng = np.random.default_rng(REWIRE_SEED + 1)
    c_post = rng.integers(0, n, size=len(pre_s))
    # repair self-loops by swapping with another edge, which is enough because
    # the draw is dense and independent
    bad = np.nonzero(c_post == pre_s)[0]
    for i in bad.tolist():
        j = (i + 1) % len(pre_s)
        guard = 0
        while c_post[j] == pre_s[i] and guard < len(pre_s):
            j = (j + 1) % len(pre_s)
            guard += 1
        c_post[i], c_post[j] = c_post[j], c_post[i]
    # the whole point of C is that it does NOT preserve degrees; assert that
    assert not np.array_equal(np.bincount(post_s, minlength=n),
                              np.bincount(c_post, minlength=n)), \
        "condition C preserved in-degree, so it is not a destroyed-degree control"
    C = Connectome("c2_C_random_targets", conn.flywire_ids,
                   pre_s.astype(np.int32), c_post.astype(np.int32), w_s)
    A = Connectome("c2_A_real", conn.flywire_ids,
                   pre_s.astype(np.int32), post_s.astype(np.int32), w_s)

    audit = {
        "A_vs_B": compare_marginals(pre_s, post_s, w_s, b_pre, b_post, w_s, n),
        "A_vs_C": compare_marginals(pre_s, post_s, w_s, pre_s, c_post, w_s, n),
        "B_rewire_report": {
            "swap_accepted": b_rep.swap_accepted,
            "rejected_duplicate": b_rep.swaps_rejected_duplicate,
            "in_degree_preserved": b_rep.in_degree_preserved,
            "out_degree_preserved": b_rep.out_degree_preserved,
            "sign_marginals_preserved": b_rep.sign_marginals_preserved,
        },
    }
    return {"A": A, "B": B, "C": C}, audit


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    conn = load_circuit()
    card = json.loads(CARD.read_text(encoding="utf-8"))
    ann = load_annotations()
    pop = output_population(conn, ann)
    sens = sensory_indices(conn)
    print(f"circuito : {conn.n_neurons:,} neurons  {len(conn.pre):,} edges "
          f"(hops={card['hops']}, {card['synapses_total']:,} sinapses)")
    print(f"entrada  : sugar_grn -> {len(sens)} neurons no circuito")
    print(f"saida    : {len(pop)} neurons (descending + motor)")
    if len(pop) < 3:
        print("POPULACAO DE SAIDA PEQUENA DEMAIS; abortando sem resultado")
        return 2
    print()

    conditions, audit = build_conditions(conn)
    print("=== auditoria dos controles (antes de qualquer decodificacao) ===")
    for k in ("A_vs_B", "A_vs_C"):
        a = audit[k]
        print(f"  {k}: graus identicos={a['out_degree_identical'] and a['in_degree_identical']}  "
              f"kl_grau_saida={a['out_degree_histogram_kl']:.4f}  "
              f"kl_peso={a['weight_histogram_kl']:.4f}")
    print(f"  B: trocas aceitas={audit['B_rewire_report']['swap_accepted']}  "
          f"sign_marginais_ok={audit['B_rewire_report']['sign_marginals_preserved']}")
    (OUT / "control_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print()

    brain = RustBrain(conditions["A"], dt_ms=0.1, reference_order=True, max_memory_bytes=RUST_MEMORY_BYTES)

    # ---- calibration: condition A only ---------------------------------------
    print("=== calibracao (so a condicao A) ===")
    ss = np.random.SeedSequence(20261002).spawn(N_SEEDS)
    seed0 = int(ss[0].generate_state(1)[0])
    rng0 = np.random.default_rng(seed0)
    pats0 = make_patterns(rng0, K_PATTERNS, len(sens))
    acc_by_w = {}
    for w in W_GRID:
        xs, ys = [], []
        for pi, p in enumerate(pats0):
            for r in range(R_TRIALS):
                xs.append(simulate(brain, p, sens, np.random.default_rng(
                    seed0 + 1000 * w + 10 * pi + r), w, pop))
                ys.append(pi)
        acc = nearest_centroid_loo(np.array(xs), np.array(ys))
        total = float(np.mean([x.sum() for x in xs]))
        acc_by_w[w] = acc
        ok = abs(acc - TARGET_ACC) <= TARGET_TOL
        print(f"  W={w:>6}  acuracia LOO = {acc:.3f}  spikes/saida={total:>9.1f}  "
              f"{'<- na faixa' if ok else ''}")
    chosen_w = next((w for w in W_GRID
                     if abs(acc_by_w[w] - TARGET_ACC) <= TARGET_TOL), None)
    if chosen_w is None:
        print()
        print("CALIBRACAO IMPOSSIVEL: nenhuma W preregistrada leva A a 0,75 +/- 0,05.")
        print("C2a e reportado como 'calibracao impossivel' e B e C NAO sao simuladas,")
        print("porque sem regime intermediario elas nao significam nada.")
        (OUT / "VERDICT.json").write_text(json.dumps(
            {"verdict": "CALIBRACAO_IMPOSSIVEL", "accuracy_by_W": acc_by_w},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return 3
    print(f"  W ESCOLHIDO = {chosen_w}  (menor W na faixa; B e C nao entraram nesta escolha)")
    (OUT / "calibration.json").write_text(json.dumps(
        {"accuracy_by_W": {str(k): v for k, v in acc_by_w.items()},
         "chosen_W": chosen_w, "target": TARGET_ACC, "tolerance": TARGET_TOL,
         "calibrated_on": "condition A only", "seed_set": seed0},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print()

    # ---- main: all seeds, all conditions, fixed W ---------------------------
    print(f"=== medicao principal (W={chosen_w}, {N_SEEDS} conjuntos de sementes) ===")
    rows = []
    t0 = time.perf_counter()
    for si, s in enumerate(ss):
        seed = int(s.generate_state(1)[0])
        rng = np.random.default_rng(seed)
        pats = make_patterns(rng, K_PATTERNS, len(sens))
        feats = {}
        for cond, c in conditions.items():
            b = brain if cond == "A" else RustBrain(c, dt_ms=0.1, reference_order=True, max_memory_bytes=RUST_MEMORY_BYTES)
            xs, ys = [], []
            for pi, p in enumerate(pats):
                for r in range(R_TRIALS):
                    xs.append(simulate(b, p, sens, np.random.default_rng(
                        seed + 1000 * chosen_w + 10 * pi + r), chosen_w, pop))
                    ys.append(pi)
            feats[cond] = (np.array(xs), np.array(ys))
        accs = {c: nearest_centroid_loo(*feats[c]) for c in ("A", "B", "C")}
        # mandatory sanity check: decoder trained on the controls, applied to A
        ctrl_x = np.vstack([feats["B"][0], feats["C"][0]])
        ctrl_y = np.concatenate([feats["B"][1], feats["C"][1]]) + 100
        a_x, a_y = feats["A"]
        dec = nearest_centroid_loo(np.vstack([ctrl_x, a_x]),
                                   np.concatenate([ctrl_y, a_y + 200]))
        accs["A_via_control_decoder"] = dec
        rows.append({"seed_set": si, "seed": seed, **accs})
        print(f"  [{si+1:>2}/{N_SEEDS}] A={accs['A']:.3f} B={accs['B']:.3f} "
              f"C={accs['C']:.3f}  A_via_controle={accs['A_via_control_decoder']:.3f}")
    print(f"  tempo: {time.perf_counter()-t0:.1f}s")
    print()

    a_arr = np.array([r["A"] for r in rows])
    b_arr = np.array([r["B"] for r in rows])
    c_arr = np.array([r["C"] for r in rows])
    ctrl_arr = np.array([r["A_via_control_decoder"] for r in rows])
    d_ab = a_arr - b_arr
    d_ac = a_arr - c_arr
    ci_ab = bootstrap_ci(d_ab, BOOTSTRAP_B, BOOTSTRAP_SEED)
    ci_ac = bootstrap_ci(d_ac, BOOTSTRAP_B, BOOTSTRAP_SEED + 1)

    if len(rows) < N_SEEDS:
        verdict, why = "INCONCLUSIVO", f"N {len(rows)} < {N_SEEDS}"
    elif ctrl_arr.mean() >= 0.5:
        verdict, why = "INCONCLUSIVO", (
            f"sanidade falhou: decoder treinado nos controles acerta A em "
            f"{ctrl_arr.mean():.3f}; A e os controles sao a mesma representacao")
    elif ci_ab["lower"] > 0:
        verdict, why = "POSITIVO", f"IC95 de A-B = [{ci_ab['lower']:.4f}, {ci_ab['upper']:.4f}]"
    elif ci_ab["lower"] <= 0 <= ci_ab["upper"] and ci_ac["lower"] <= 0 <= ci_ac["upper"]:
        verdict, why = "NEGATIVO", "A-B e A-C contem 0"
    else:
        verdict, why = "INCONCLUSIVO", f"IC95 de A-B = [{ci_ab['lower']:.4f}, {ci_ab['upper']:.4f}]"

    analysis = {
        "preregistration": "docs/PREREGISTRATION_C2.md rev. 4",
        "executed_utc": utc(),
        "chosen_W": chosen_w,
        "n_seed_sets": len(rows),
        "patterns": K_PATTERNS,
        "trials_per_pattern": R_TRIALS,
        "measurements_per_seed_set": K_PATTERNS * R_TRIALS,
        "output_population_size": len(pop),
        "sensory_neurons": len(sens),
        "descriptive": {
            "A": {"mean": float(a_arr.mean()), "std": float(a_arr.std(ddof=1)),
                  "min": float(a_arr.min()), "max": float(a_arr.max())},
            "B": {"mean": float(b_arr.mean()), "std": float(b_arr.std(ddof=1)),
                  "min": float(b_arr.min()), "max": float(b_arr.max())},
            "C": {"mean": float(c_arr.mean()), "std": float(c_arr.std(ddof=1)),
                  "min": float(c_arr.min()), "max": float(c_arr.max())},
        },
        "primary_A_minus_B": {"mean": float(d_ab.mean()), "ci": ci_ab},
        "secondary_A_minus_C": {"mean": float(d_ac.mean()), "ci": ci_ac},
        "sanity_control_decoder_on_A": {
            "mean": float(ctrl_arr.mean()),
            "interpretation": "must be near chance; if it is high, B and C are not "
                              "different enough from A to be controls",
        },
        "verdict": verdict,
        "verdict_reason": why,
        "wall_clock_seconds": time.perf_counter() - t0,
    }
    (OUT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "per_seed_set.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== RESULTADO C2a ===")
    for c in ("A", "B", "C"):
        d = analysis["descriptive"][c]
        print(f"  {c}: media {d['mean']:.4f}  dp {d['std']:.4f}  "
              f"[{d['min']:.3f}, {d['max']:.3f}]")
    print(f"  A - B: {d_ab.mean():+.4f}  IC95 [{ci_ab['lower']:.4f}, {ci_ab['upper']:.4f}]")
    print(f"  A - C: {d_ac.mean():+.4f}  IC95 [{ci_ac['lower']:.4f}, {ci_ac['upper']:.4f}]")
    print(f"  sanidade (decoder dos controles em A): {ctrl_arr.mean():.4f}  "
          f"(precisa ser baixo)")
    print(f"  VEREDITO: {verdict}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
