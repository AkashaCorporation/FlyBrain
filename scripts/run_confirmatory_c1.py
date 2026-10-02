"""Confirmatory study C1 - executes docs/PREREGISTRATION_C1.md v1.0.

The pre-registration was sealed before this script ran (see PREREG_SEAL.txt).
Nothing here may be tuned after the data exists: the seed list, the budget, the
estimand, the CI method and the decision rule are all fixed by that document.

Refuses to write into an existing directory, so a run can never overwrite the
evidence of an earlier one.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cycle_evidence import identity, save  # noqa: E402
from flybrain.learning.tabular import make_individuals, train  # noqa: E402
from flybrain.social.evaluation import control_gate, evaluate  # noqa: E402

# ---- constants fixed by the pre-registration, section 9 ----------------------
N_SEEDS = 64
SEED_ROOT = 20261002
BOOTSTRAP_SEED = 20261002
BOOTSTRAP_B = 10_000
EPISODES = 12_000
MAX_SECONDS = 30
EVAL_EPISODES = 512
EVAL_REPEATS = 64
THRESHOLD_POSITIVE = 0.40   # CI lower bound must exceed this
THRESHOLD_NEGATIVE = 0.30   # CI upper bound at or below this -> negative
S3_CURVE_THRESHOLD = 0.75
S3_EPISODE_FRACTION = 0.60

OUT = ROOT / "outputs/confirmatory_c1"
SEEDS_PATH = OUT / "preregistered_seeds.json"


def first_curve_crossing(curve: list[dict], threshold: float) -> int | None:
    """S3: first recorded curve point whose mean training return reaches threshold."""
    for point in curve:
        if point["mean_training_return"] >= threshold:
            return int(point["episodes"])
    return None


def bootstrap_ci(values: np.ndarray, statistic, *, b: int, seed: int,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap. Assumes nothing about the sampling distribution."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = rng.integers(0, n, size=(b, n))
    stats = np.array([statistic(values[idx]) for idx in draws])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lower": float(lo),
        "upper": float(hi),
        "b": int(b),
        "alpha": float(alpha),
        "seed": int(seed),
        "method": "percentile_bootstrap",
    }


def paired_sign_flip_test(d: np.ndarray, *, b: int, seed: int) -> dict:
    """Two-sided paired randomization test by sign flips. No normality assumed.

    Under the null that the paired difference has symmetric distribution around
    zero, the sign of each pair is exchangeable. This is the assumption-light
    counterpart to Wilcoxon and is reported instead when scipy is absent.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    obs = abs(float(np.mean(d)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
    null_means = np.abs((signs * d).mean(axis=1))
    # add-one smoothing: never report an exact zero p-value
    p = (np.sum(null_means >= obs) + 1) / (b + 1)
    return {
        "statistic": "abs_mean_paired_difference",
        "observed": obs,
        "p_two_sided": float(p),
        "b": int(b),
        "seed": int(seed),
        "method": "paired_sign_flip_randomization",
    }


def wilcoxon_if_available(d: np.ndarray) -> dict | None:
    try:
        from scipy.stats import wilcoxon  # noqa: PLC0415
    except ImportError:
        return None
    if np.allclose(d, 0.0):
        return {"note": "all differences zero", "p_two_sided": 1.0}
    r = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {
        "statistic": float(r.statistic),
        "p_two_sided": float(r.pvalue),
        "method": "scipy_wilcoxon_signed_rank",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)

    # The sealed pre-registration travels with the evidence. Copied in, never
    # written by this script, so the run cannot edit what it is judged against.
    seal_src = ROOT / "docs/PREREGISTRATION_C1_SEAL.txt"
    prereg = ROOT / "docs/PREREGISTRATION_C1.md"
    if not seal_src.exists():
        raise SystemExit("pre-registration seal missing; refusing to run unpreregistered")
    seal_text = seal_src.read_text(encoding="utf-8")
    (OUT / "PREREG_SEAL.txt").write_text(seal_text, encoding="utf-8")

    # The seal lists the current revision last; that must be what is on disk now.
    sealed_hashes = re.findall(r"\b[0-9a-f]{64}\b", seal_text)
    live = hashlib.sha256(prereg.read_bytes()).hexdigest()
    if not sealed_hashes or live != sealed_hashes[-1]:
        raise SystemExit(
            f"pre-registro divergente do selo.\n  vivo   : {live}\n"
            f"  selado : {sealed_hashes[-1] if sealed_hashes else '(nenhum)'}"
        )
    print(f"pre-registro conferido contra o selo: {live[:16]}...")

    # ---- 1. the seed list is written BEFORE any training exists ------------
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(SEED_ROOT).spawn(N_SEEDS)]
    save(SEEDS_PATH, {
        "n": N_SEEDS,
        "seed_root": SEED_ROOT,
        "derivation": "np.random.SeedSequence(20261002).spawn(64) -> generate_state(1)[0]",
        "seeds": seeds,
        "sha256_of_list": hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest(),
        "written_before_training": True,
    })
    print(f"seeds gravados em {SEEDS_PATH} (antes de qualquer treino)")

    # ---- 2. controls gate everything ---------------------------------------
    t0 = time.perf_counter()
    gate = control_gate(repeats=16)
    save(OUT / "control_gate.json", gate)
    if not gate["passed"]:
        save(OUT / "VERDICT.json", {
            "verdict": "NAO_EXECUTADO",
            "reason": "control gate failed; no training performed",
        })
        print("CONTROLE FALHOU - nenhum treino executado")
        return 2
    print(f"controles OK em {time.perf_counter()-t0:.1f}s")

    # ---- 3. per-seed run ---------------------------------------------------
    rows = []
    for i, seed in enumerate(seeds, 1):
        report = identity()
        env, agents, streams = make_individuals(seed)
        report.update(
            seed=seed,
            seeds={"root": seed, "world_init_action_streams": streams},
            requested_mode="C1_independent_tabular_confirmatory",
            effective_mode="C1_independent_tabular_confirmatory",
            dataset="synthetic_hidden_choice",
            mask=None,
            evidence_level="confirmatory_preregistered_N64",
            preregistration="docs/PREREGISTRATION_C1.md v1.0",
            limits={
                "training_episodes": EPISODES,
                "training_seconds": MAX_SECONDS,
                "evaluation_episodes": EVAL_EPISODES,
            },
            retention_tested=False,
            neural_substrate_used=False,
            control_gate_passed=True,
        )
        report["before"] = evaluate(agents, seed=seed + 10000, repeats=EVAL_REPEATS)
        report["training"] = train(env, agents, episodes=EPISODES, max_seconds=MAX_SECONDS)

        # frozen weights: no further learning happens between after and holdout
        report["after"] = evaluate(agents, seed=seed + 20000, repeats=EVAL_REPEATS)
        report["holdout"] = evaluate(agents, seed=seed + 30000, repeats=EVAL_REPEATS)

        wpath = OUT / f"seed_{seed}_weights.npz"
        np.savez(
            wpath,
            **{
                f"{name}_{field}": getattr(p, field)
                for name, p in agents.items()
                for field in ("sender_logits", "receiver_logits",
                              "sender_baseline", "receiver_baseline")
            },
        )
        report["weights_sha256"] = hashlib.sha256(wpath.read_bytes()).hexdigest()
        report["s3_first_curve_crossing_episodes"] = first_curve_crossing(
            report["training"]["curve"], S3_CURVE_THRESHOLD
        )
        save(OUT / f"seed_{seed}.json", report)

        pre = float(report["before"]["expected"]["intact"])
        post = float(report["after"]["expected"]["intact"])
        rows.append({
            "seed": seed,
            "intact_pre": pre,
            "intact_post": post,
            "d": post - pre,
            "intact_blocked_post": float(report["after"]["expected"]["blocked"]),
            "intact_resampled_post": float(report["after"]["expected"]["resampled"]),
            "intact_holdout": float(report["holdout"]["expected"]["intact"]),
            "gain_vs_resampled": float(report["after"]["expected"]["gain_vs_resampled"]),
            "tv_uniform_pairs": float(report["after"]["expected"]["tv_uniform_pairs"]),
            "classification": report["after"]["classification"],
            "completed": bool(report["training"]["completed"]),
            "stop_reason": report["training"]["stop_reason"],
            "s3_crossing": report["s3_first_curve_crossing_episodes"],
        })
        print(f"[{i:>2}/{N_SEEDS}] seed {seed:>11}  pre={pre:.4f}  post={post:.4f}  "
              f"d={post - pre:+.4f}  {report['after']['classification']}  "
              f"({report['training']['elapsed_seconds']:.1f}s)", flush=True)

    # ---- 4. the pre-registered analysis -----------------------------------
    d = np.array([r["d"] for r in rows], dtype=float)
    n_analyzed = len(d)
    mean_ci = bootstrap_ci(d, np.mean, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    median_ci = bootstrap_ci(d, np.median, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED + 1)

    # leave-one-out: how much does any single seed move the mean?
    loo = [float(np.mean(np.delete(d, i))) for i in range(n_analyzed)]

    q1, q3 = np.percentile(d, [25, 75])
    analysis = {
        "n_preregistered": N_SEEDS,
        "n_analyzed": n_analyzed,
        "no_seed_discarded": True,
        "primary_estimand": "per-seed paired change in expected intact return (post - pre)",
        "descriptive": {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "iqr": [float(q1), float(q3)],
            "std": float(np.std(d, ddof=1)) if n_analyzed > 1 else None,
            "min": float(np.min(d)),
            "max": float(np.max(d)),
        },
        "ci_mean": mean_ci,
        "ci_median": median_ci,
        "leave_one_out_mean_min": float(np.min(loo)),
        "leave_one_out_mean_max": float(np.max(loo)),
        "support_tests": {
            "sign_flip": paired_sign_flip_test(d, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED + 2),
            "wilcoxon": wilcoxon_if_available(d),
        },
        "secondary_S1_fraction_meeting_operational_criteria": float(np.mean([
            r["classification"] == "functional_channel_use" for r in rows
        ])),
        "secondary_S1_target": 0.95,
        "secondary_S2_holdout_ratio_median": float(np.median([
            r["intact_holdout"] / r["intact_post"] if r["intact_post"] > 0 else float("nan")
            for r in rows
        ])),
        "secondary_S3_median_crossing_episodes": float(np.median([
            r["s3_crossing"] for r in rows if r["s3_crossing"] is not None
        ])) if any(r["s3_crossing"] is not None for r in rows) else None,
        "secondary_S3_target_episodes": EPISODES * S3_EPISODE_FRACTION,
        "secondary_S3_crossed": int(sum(
            r["s3_crossing"] is not None and r["s3_crossing"] <= EPISODES * S3_EPISODE_FRACTION
            for r in rows
        )),
        "all_seeds_completed_training": all(r["completed"] for r in rows),
        "wall_clock_seconds": time.perf_counter() - t0,
    }

    # decision rule, verbatim from pre-registration section 2
    if n_analyzed < N_SEEDS:
        verdict, why = "INCONCLUSIVO", f"N analisado {n_analyzed} < {N_SEEDS} declarado"
    elif mean_ci["lower"] > THRESHOLD_POSITIVE:
        verdict, why = "POSITIVO", f"limite inferior {mean_ci['lower']:.4f} > {THRESHOLD_POSITIVE}"
    elif mean_ci["upper"] <= THRESHOLD_NEGATIVE:
        verdict, why = "NEGATIVO", f"limite superior {mean_ci['upper']:.4f} <= {THRESHOLD_NEGATIVE}"
    else:
        verdict, why = "INCONCLUSIVO", (
            f"IC [{mean_ci['lower']:.4f}, {mean_ci['upper']:.4f}] atravessa a faixa "
            f"({THRESHOLD_NEGATIVE}, {THRESHOLD_POSITIVE}]"
        )
    analysis["verdict"] = verdict
    analysis["verdict_reason"] = why
    analysis["decision_rule"] = {
        "positive": f"CI95 lower > {THRESHOLD_POSITIVE}",
        "negative": f"CI95 upper <= {THRESHOLD_NEGATIVE}",
        "otherwise": "INCONCLUSIVO",
    }

    save(OUT / "analysis.json", analysis)
    save(OUT / "per_seed.csv.json", rows)

    print()
    print("=== ANALISE PRE-REGISTRADA ===")
    print(f"  N analisado          : {n_analyzed}/{N_SEEDS}")
    print(f"  media de d           : {analysis['descriptive']['mean']:.4f}")
    print(f"  mediana de d         : {analysis['descriptive']['median']:.4f}")
    print(f"  IQR                  : [{q1:.4f}, {q3:.4f}]")
    print(f"  min / max            : {analysis['descriptive']['min']:.4f} / "
          f"{analysis['descriptive']['max']:.4f}")
    print(f"  IC95 da media        : [{mean_ci['lower']:.4f}, {mean_ci['upper']:.4f}]")
    print(f"  leave-one-out (min/max): [{min(loo):.4f}, {max(loo):.4f}]")
    print(f"  S1 frac criterio     : {analysis['secondary_S1_fraction_meeting_operational_criteria']:.3f} "
          f"(alvo >= 0.95)")
    print(f"  S2 razao holdout     : {analysis['secondary_S2_holdout_ratio_median']:.4f} (mediana)")
    s3 = analysis["secondary_S3_median_crossing_episodes"]
    print(f"  S3 cruzamento mediana: {'nenhuma semente' if s3 is None else f'{s3:.0f} episodios'} "
          f"(alvo <= {analysis['secondary_S3_target_episodes']:.0f}, "
          f"{analysis['secondary_S3_crossed']}/{n_analyzed} sementes)")
    print(f"  todas_COMPLETaram    : {analysis['all_seeds_completed_training']}")
    print(f"  VEREDITO             : {verdict}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
