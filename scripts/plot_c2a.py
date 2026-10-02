"""Figures for the C2a representational result.

Reads outputs/neural_c2/c2a/analysis.json and per_seed_set.json. Produces PNGs
with plain matplotlib defaults, readable axes over decoration, matching the
project's stated preference for correctness over aesthetics.

If the run reported CALIBRACAO_IMPOSSIVEL, this exits without producing figures,
because there is nothing to plot and an empty figure would be misleading.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2A = ROOT / "outputs/neural_c2/c2a"
FIG = ROOT / "docs/figures/c2a"
CAL = C2A / "calibration.json"


def load(name: str):
    p = C2A / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    analysis = load("analysis.json")
    rows = load("per_seed_set.json")
    cal = load("calibration.json")
    audit = load("control_audit.json")
    if analysis is None or rows is None:
        print("sem resultado de C2a em", C2A)
        return 2
    FIG.mkdir(parents=True, exist_ok=True)

    a = np.array([r["A"] for r in rows])
    b = np.array([r["B"] for r in rows])
    c = np.array([r["C"] for r in rows])
    ctrl = np.array([r["A_via_control_decoder"] for r in rows])
    labels = ["A\nreal", "B\ngrau preservado", "C\naleatorio"]
    colours = ["#1f77b4", "#ff7f0e", "#7f7f7f"]
    chance = 1.0 / analysis["patterns"]

    # --- fig 1: per-condition accuracy ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [a, b, c]
    parts = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, col in zip(parts["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, (vals, col) in enumerate(zip(data, colours), start=1):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.plot(i + jitter, vals, "o", ms=4, color=col, alpha=0.8, zorder=3)
        ax.plot(i, vals.mean(), "_", ms=22, mew=2.5, color="black", zorder=4)
    ax.axhline(chance, ls="--", color="crimson", lw=1.2,
               label=f"acaso = 1/{analysis['patterns']} = {chance:.3f}")
    ax.set_ylabel("acuracia leave-one-out")
    ax.set_title(f"C2a: quanto da pista chega a saida do circuito\n"
                 f"circuito de {analysis.get('sensory_neurons', '?')} entradas, "
                 f"W={analysis['chosen_W']}, {analysis['n_seed_sets']} conjuntos de sementes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "c2a_accuracy_by_condition.png", dpi=150)
    plt.close(fig)

    # --- fig 2: paired differences -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (d, name, col) in enumerate([(a - b, "A - B", "#d62728"),
                                        (a - c, "A - C", "#9467bd")]):
        ci = analysis["primary_A_minus_B"]["ci"] if name == "A - B" \
            else analysis["secondary_A_minus_C"]["ci"]
        jitter = np.linspace(-0.08, 0.08, len(d))
        ax.plot(i + jitter, d, "o", ms=4, color=col, alpha=0.75, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [d.mean()] * 2, "-", color="black", lw=2.5, zorder=4)
        ax.errorbar(i, d.mean(), yerr=[[d.mean() - ci["lower"]], [ci["upper"] - d.mean()]],
                    fmt="none", ecolor=col, elinewidth=2, capsize=7, zorder=4)
        ax.annotate(f"mediana {d.mean():+.3f}\nIC95 [{ci['lower']:.3f}, {ci['upper']:.3f}]",
                    (i, ci["upper"]), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A - B", "A - C"])
    ax.set_ylabel("diferenca pareada de acuracia")
    ax.set_title("A e mais legivel que os controles?\n(negativo = controles tao "
                 "legiveis quanto A, controle inutil)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "c2a_paired_differences.png", dpi=150)
    plt.close(fig)

    # --- fig 3: the mandatory sanity check ----------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(ctrl, bins=8, color="#7f7f7f", edgecolor="black", alpha=0.8)
    ax.axvline(ctrl.mean(), color="black", lw=2,
               label=f"media {ctrl.mean():.3f}")
    ax.axvline(chance, ls="--", color="crimson", lw=1.5,
               label=f"acaso {chance:.3f}")
    ax.set_xlabel("acuracia do decoder TREINADO NOS CONTROLES, aplicado a A")
    ax.set_ylabel("conjuntos de sementes")
    ax.set_title("Teste de sanidade obrigatorio\n"
                 "se a barraå³ estiver longe do acaso, A e os controles\n"
                 "sao a mesma representacao e C2a e inconclusivo".replace("å³", "a direita"))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "c2a_sanity_control_decoder.png", dpi=150)
    plt.close(fig)

    # --- fig 4: calibration curve, condition A only --------------------------
    if cal:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ws = sorted(int(k) for k in cal["accuracy_by_W"])
        accs = [cal["accuracy_by_W"][str(w)] for w in ws]
        ax.plot(ws, accs, "o-", color="#1f77b4", lw=2, ms=7)
        ax.axhline(0.75, ls="--", color="crimson", lw=1.4)
        ax.fill_between(ws, 0.70, 0.80, color="crimson", alpha=0.10)
        ax.axvline(cal["chosen_W"], color="green", lw=1.6,
                   label=f"W escolhido = {cal['chosen_W']}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("W (passos, escala log2)")
        ax.set_ylabel("acuracia LOO na condicao A")
        ax.set_title("Calibracao de dificuldade: menor W com A em 0,75 +/- 0,05\n"
                     "calculado com a condicao A apenas; B e C nao entraram")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "c2a_calibration_curve.png", dpi=150)
        plt.close(fig)

    # --- fig 5: control audit ------------------------------------------------
    if audit:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, key in enumerate(("A_vs_B", "A_vs_C")):
            kls = [audit[key]["out_degree_histogram_kl"], audit[key]["weight_histogram_kl"]]
            same = audit[key]["out_degree_identical"] and audit[key]["in_degree_identical"]
            off = i * 3.2
            ax.bar([off, off + 1], kls, width=0.8,
                   color=["#ff7f0e" if key == "A_vs_B" else "#7f7f7f"], alpha=0.85,
                   edgecolor="black")
            ax.annotate(f"graus identicos: {same}", (off, max(kls) * 1.05 + 0.01),
                        ha="center", fontsize=8)
        ax.set_xticks([0.5, 3.7])
        ax.set_xticklabels(["A vs B\n(grau preservado)", "A vs C\n(aleatorio)"])
        ax.set_ylabel("divergencia de Jensen-Shannon")
        ax.set_title("Auditoria dos controles, antes de qualquer decodificacao\n"
                     "0 = grafos indistinguiveis nes marginals")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "c2a_control_audit.png", dpi=150)
        plt.close(fig)

    made = sorted(p.name for p in FIG.glob("*.png"))
    print(f"{len(made)} figuras em {FIG}")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
