"""C2b - executes docs/PREREGISTRATION_C2B.md rev. 1.

The circuit IS the sender. The private cue drives the real connectome, the
655-neuron output response is decoded into the message by templates built on
condition A, and a receiver identical to the one in C1 learns to use it.

What this does and does not claim is in section 3 of the pre-registration. Read
that before the code. The short version: the sender has no agency and is frozen,
the circuit is not plastic, and only the receiver learns.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flybrain.learning.tabular import make_individuals, train  # noqa: E402
from flybrain.model.rust import RustBrain  # noqa: E402
from flybrain.social.evaluation import control_gate, evaluate  # noqa: E402
from run_neural_c2a import (  # noqa: E402
    DELTA_V_MV,
    RUST_MEMORY_BYTES,
    STIM_RATE_HZ,
    build_conditions,
    load_annotations,
    load_circuit,
    output_population,
    sensory_indices,
)

C2A = ROOT / "outputs/neural_c2/c2a"
OUT = ROOT / "outputs/neural_c2/c2b"

# --- fixed by the pre-registration ------------------------------------------
N_SEEDS = 16
SEED_ROOT = 20261003          # deliberately not C2a's 20261002
R_TRIALS = 16
EPISODES = 12_000
MAX_SECONDS = 60
EVAL_REPEATS = 64
BOOTSTRAP_SEED = 20261002
BOOTSTRAP_B = 10_000
TOL = 1e-9


def utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cue_pattern(cue: int, sensory: list[int], half: np.ndarray) -> np.ndarray:
    """The cue is the binary world state; it drives a fixed, shared sensory set.

    Cue 1 activates the first half of the sensory neurons and cue 0 the other
    half, so both cues drive exactly the same NUMBER of neurons and the two
    stimuli are matched in size. The complement of a boolean index must be
    `~half`: `1 - half` yields 0/1 integers, which as an index array selects
    neurons 0 and 1 and silently makes both cues identical.
    """
    p = np.zeros(len(sensory), dtype=np.int8)
    p[half if cue == 1 else ~half] = 1
    assert p.sum() == int(half.sum()), (
        f"cue {cue} drives {int(p.sum())} neurons, expected {int(half.sum())}")
    return p


def responses_for(brain: RustBrain, sensory: list[int], pop: list[int],
                  w_steps: int, seed: int) -> dict[int, np.ndarray]:
    """R independent 655-neuron responses per cue."""
    half = np.zeros(len(sensory), dtype=bool)
    half[: len(sensory) // 2] = True
    out = {}
    for cue in (0, 1):
        pat = cue_pattern(cue, sensory, half)
        trials = []
        for r in range(R_TRIALS):
            rng = np.random.default_rng(seed + 100 * cue + r)
            brain.reset()
            events = []
            pr = STIM_RATE_HZ * 1e-3
            for ni, on in zip(sensory, pat):
                if on:
                    for t in np.nonzero(rng.random(w_steps) < pr)[0].tolist():
                        events.append((int(t), int(ni), DELTA_V_MV, 0.0))
            events.sort(key=lambda e: e[0])
            res = brain.advance(
                w_steps, inputs=events,
                populations=[[i] for i in pop],
            )
            counts = np.asarray(res["population_counts"], dtype=np.float64)
            if counts.size != len(pop):
                raise RuntimeError(
                    f"readout returned {counts.size} counts for {len(pop)} output "
                    f"neurons; the per-neuron contract is broken"
                )
            trials.append(counts)
        out[cue] = np.array(trials)
    return out


def nearest_template(response: np.ndarray, templates: dict[int, np.ndarray]) -> int:
    """Symbol is the cue whose template is closest. No free parameters."""
    keys = sorted(templates)
    d = [float(np.linalg.norm(response - templates[k])) for k in keys]
    return int(keys[int(np.argmin(d))])


def bootstrap_ci(d: np.ndarray, b: int, seed: int, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = rng.integers(0, n, size=(b, n))
    means = d[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lower": float(lo), "upper": float(hi), "b": int(b), "seed": int(seed)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)

    cal = json.loads((C2A / "calibration.json").read_text(encoding="utf-8"))
    w_steps = int(cal["chosen_W"])
    if w_steps != 160:
        raise SystemExit(f"unexpected inherited W={w_steps}; C2b registered for 160")

    conn = load_circuit()
    ann = load_annotations()
    pop = output_population(conn, ann)
    sens = sensory_indices(conn)
    print(f"circuito : {conn.n_neurons:,} neurons  {len(conn.pre):,} edges")
    print(f"entrada  : {len(sens)} sugar_grn | saida {len(pop)} | W = {w_steps} (herdado)")
    print(f"ensaios por pista : {R_TRIALS}")
    print()

    gate = control_gate(repeats=16)
    (OUT / "control_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    if not gate["passed"]:
        (OUT / "VERDICT.json").write_text(json.dumps(
            {"verdict": "NAO_EXECUTADO", "reason": "control gate failed"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print("CONTROLE FALHOU - nenhum treino executado")
        return 2
    print("controles de ambiente OK (o portao que C1 usa)\n")

    conditions, audit = build_conditions(conn)
    (OUT / "control_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    seeds = [int(s.generate_state(1)[0])
             for s in np.random.SeedSequence(SEED_ROOT).spawn(N_SEEDS)]
    (OUT / "preregistered_seeds.json").write_text(json.dumps(
        {"n": N_SEEDS, "seed_root": SEED_ROOT,
         "derivation": "np.random.SeedSequence(20261003).spawn(16) -> generate_state(1)[0]",
         "seeds": seeds, "written_before_training": True},
        indent=2), encoding="utf-8")
    print(f"sementes gravadas antes do treino ({N_SEEDS})\n")

    brains = {k: RustBrain(c, dt_ms=0.1, reference_order=True,
                           max_memory_bytes=RUST_MEMORY_BYTES)
              for k, c in conditions.items()}

    rows = []
    t0 = time.perf_counter()
    for si, seed in enumerate(seeds):
        resp = {cond: responses_for(brains[cond], sens, pop, w_steps, seed)
                for cond in ("A", "B", "C")}
        templates_A = {c: resp["A"][c].mean(axis=0) for c in (0, 1)}
        # mandatory reverse sanity: templates from the controls, applied to A
        tpl_ctrl = {c: np.mean([resp["B"][c].mean(axis=0), resp["C"][c].mean(axis=0)])
                    for c in (0, 1)}

        # channel fidelity under the A templates: how often does the emitted
        # symbol equal the cue, before any receiver exists
        fidelity = {}
        for cond in ("A", "B", "C"):
            hits = 0
            n = 0
            for c in (0, 1):
                for trial in resp[cond][c]:
                    hits += int(nearest_template(trial, templates_A) == c)
                    n += 1
            fidelity[cond] = hits / n
        hits = 0
        n = 0
        for c in (0, 1):
            for trial in resp["A"][c]:
                hits += int(nearest_template(trial, tpl_ctrl) == c)
                n += 1
        sanity_channel = hits / n

        entry = {"seed_set": si, "seed": seed, "channel_fidelity": fidelity,
                 "sanity_channel_fidelity": sanity_channel}

        for cond in ("A", "B", "C"):
            tpl = templates_A
            pick_rng = np.random.default_rng(seed + 7)

            def message_fn(cue, _resp=resp[cond], _tpl=tpl, _rng=pick_rng):
                trial = _resp[cue][_rng.integers(R_TRIALS)]
                return nearest_template(trial, _tpl)

            env, agents, _streams = make_individuals(seed)
            report = train(env, agents, episodes=EPISODES, max_seconds=MAX_SECONDS,
                           message_fn=message_fn)
            # the SAME message source must drive the evaluation. Grading a
            # receiver trained on the circuit against a replay that re-emits with
            # the frozen sender's policy scores a channel training never used:
            # that mistake gave intact 0.459 while the training curve reached
            # 0.990. See tests/test_channel_coherence.py.
            ev = evaluate(agents, seed=seed + 20000, repeats=EVAL_REPEATS,
                          message_fn=message_fn)
            exp = ev["expected"]
            entry[cond] = {
                "intact": float(exp["intact"]),
                "blocked": float(exp["blocked"]),
                "resampled": float(exp["resampled"]),
                "gain_vs_resampled": float(exp["gain_vs_resampled"]),
                "gain_vs_blocked": float(exp["gain_vs_blocked"]),
                "classification": ev["classification"],
                "episodes_completed": int(report["episodes_completed"]),
            }
        rows.append(entry)
        print(f"[{si+1:>2}/{N_SEEDS}] fidelidade A={fidelity['A']:.3f} "
              f"B={fidelity['B']:.3f} C={fidelity['C']:.3f} | "
              f"ganho_resamp A={entry['A']['gain_vs_resampled']:.3f} "
              f"B={entry['B']['gain_vs_resampled']:.3f} "
              f"C={entry['C']['gain_vs_resampled']:.3f}", flush=True)
    elapsed = time.perf_counter() - t0

    def col(cond, field):
        return np.array([r[cond][field] for r in rows], dtype=float)

    g_ab = col("A", "gain_vs_resampled") - col("B", "gain_vs_resampled")
    g_ac = col("A", "gain_vs_resampled") - col("C", "gain_vs_resampled")
    i_ab = col("A", "intact") - col("B", "intact")
    ci_gab = bootstrap_ci(g_ab, BOOTSTRAP_B, BOOTSTRAP_SEED)
    ci_gac = bootstrap_ci(g_ac, BOOTSTRAP_B, BOOTSTRAP_SEED + 1)
    ci_iab = bootstrap_ci(i_ab, BOOTSTRAP_B, BOOTSTRAP_SEED + 2)
    sanity = np.array([r["sanity_channel_fidelity"] for r in rows], dtype=float)

    if len(rows) < N_SEEDS:
        verdict, why = "INCONCLUSIVO", f"N {len(rows)} < {N_SEEDS}"
    elif sanity.mean() >= 0.90:
        verdict, why = "INCONCLUSIVO", (
            f"sanidade: molde dos controles le A com {sanity.mean():.3f}; "
            f"A e os controles sao a mesma representacao")
    elif ci_gab["lower"] > 0:
        verdict, why = "POSITIVO", (
            f"ganho A-B = {g_ab.mean():+.4f}, IC95 [{ci_gab['lower']:.4f}, "
            f"{ci_gab['upper']:.4f}]")
    elif ci_gab["lower"] <= 0 <= ci_gab["upper"] and ci_gac["lower"] <= 0 <= ci_gac["upper"]:
        verdict, why = "NEGATIVO", "A-B e A-C contem 0"
    else:
        verdict, why = "INCONCLUSIVO", (
            f"IC95 do ganho A-B = [{ci_gab['lower']:.4f}, {ci_gab['upper']:.4f}]")

    analysis = {
        "preregistration": "docs/PREREGISTRATION_C2B.md rev. 1",
        "executed_utc": utc(),
        "inherited_W": w_steps,
        "n_seed_sets": len(rows),
        "trials_per_cue": R_TRIALS,
        "output_population_size": len(pop),
        "sensory_neurons": len(sens),
        "episodes_per_training": EPISODES,
        "channel_fidelity": {
            c: {"mean": float(np.mean([r["channel_fidelity"][c] for r in rows])),
                "std": float(np.std([r["channel_fidelity"][c] for r in rows], ddof=1))}
            for c in ("A", "B", "C")
        },
        "sanity_channel_fidelity_control_templates": {
            "mean": float(sanity.mean()), "std": float(sanity.std(ddof=1))},
        "primary_gain_vs_resampled_A_minus_B": {
            "mean": float(g_ab.mean()), "ci": ci_gab},
        "secondary_gain_vs_resampled_A_minus_C": {
            "mean": float(g_ac.mean()), "ci": ci_gac},
        "secondary_intact_A_minus_B": {"mean": float(i_ab.mean()), "ci": ci_iab},
        "descriptive": {
            c: {f: {"mean": float(col(c, f).mean()),
                    "std": float(col(c, f).std(ddof=1))}
                for f in ("intact", "blocked", "resampled",
                          "gain_vs_resampled", "gain_vs_blocked")}
            for c in ("A", "B", "C")
        },
        "all_trained_to_budget": all(
            r[c]["episodes_completed"] == EPISODES for r in rows for c in ("A", "B", "C")),
        "verdict": verdict,
        "verdict_reason": why,
        "wall_clock_seconds": elapsed,
    }
    (OUT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "per_seed_set.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULTADO C2b ===")
    for c in ("A", "B", "C"):
        f = analysis["channel_fidelity"][c]
        d = analysis["descriptive"][c]
        print(f"  {c}: fidelidade do canal {f['mean']:.3f} | "
              f"intacto {d['intact']['mean']:.4f} | "
              f"bloqueado {d['blocked']['mean']:.4f} | "
              f"reamostrado {d['resampled']['mean']:.4f} | "
              f"ganho {d['gain_vs_resampled']['mean']:+.4f}")
    print(f"  sanidade (moldes dos controles em A): {sanity.mean():.4f}")
    print(f"  ganho A-B: {g_ab.mean():+.4f} IC95 [{ci_gab['lower']:.4f}, {ci_gab['upper']:.4f}]")
    print(f"  ganho A-C: {g_ac.mean():+.4f} IC95 [{ci_gac['lower']:.4f}, {ci_gac['upper']:.4f}]")
    print(f"  intacto A-B: {i_ab.mean():+.4f} IC95 [{ci_iab['lower']:.4f}, {ci_iab['upper']:.4f}]")
    print(f"  VEREDITO: {verdict}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())