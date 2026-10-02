"""Figures for C2b, the circuit-as-message study.

Reads outputs/neural_c2/c2b/analysis.json and per_seed_set.json. Four panels that
answer, in order: is the channel real, does the receiver use it, is the clean
part the blocked/resampled conditions, and what does the C2a gradient add.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C2B = ROOT / "outputs/neural_c2/c2b"
C2A = ROOT / "outputs/neural_c2/c2a"
FIG = ROOT / "docs/figures/c2b"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    a = load(C2B / "analysis.json")
    rows = load(C2B / "per_seed_set.json")
    a2 = load(C2A / "analysis.json")
    if a is None or rows is None:
        print("sem resultado de C2b em", C2B)
        return 2
    FIG.mkdir(parents=True, exist_ok=True)

    conds = ("A", "B", "C")
    labels = ["A\nreal", "B\ngrau preservado", "C\naleatorio"]
    colours = ["#1f77b4", "#ff7f0e", "#7f7f7f"]

    def col(c, f):
        return np.array([r[c][f] for r in rows], dtype=float)

    fid = {c: np.array([r["channel_fidelity"][c] for r in rows]) for c in conds}
    intact = {c: col(c, "intact") for c in conds}
    blocked = {c: col(c, "blocked") for c in conds}
    resamp = {c: col(c, "resampled") for c in conds}
    gain = {c: col(c, "gain_vs_resampled") for c in conds}

    # --- fig 1: the two panels that carry the claim ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    parts = ax.boxplot([fid[c] for c in conds], tick_labels=labels,
                       patch_artist=True, widths=0.55)
    for patch, col_ in zip(parts["boxes"], colours):
        patch.set_facecolor(col_)
        patch.set_alpha(0.55)
    for i, c in enumerate(conds, 1):
        j = np.linspace(-0.12, 0.12, len(fid[c]))
        ax.plot(i + j, fid[c], "o", ms=4, color=colours[i - 1], alpha=0.8, zorder=3)
    ax.axhline(0.5, ls="--", color="crimson", lw=1.4, label="acaso binario = 0,50")
    ax.set_ylabel("fidelidade do canal")
    ax.set_title("1. o circuito entrega a pista?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    width = 0.26
    xs = np.arange(3)
    for k, (name, vals, col_) in enumerate(
            [("intacto", intact, "#1f77b4"), ("bloqueado", blocked, "#2ca02c"),
             ("reamostrado", resamp, "#d62728")]):
        ax.bar(xs + (k - 1) * width, [vals[c].mean() for c in conds], width,
               label=name, color=col_, edgecolor="black", alpha=0.85)
    ax.axhline(0.5, ls="--", color="crimson", lw=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("retorno esperado")
    ax.set_title("2. e o receptor usa o canal?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("C2b: a fiação real carrega a mensagem da comunicação", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "c2b_channel_and_use.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- fig 2: paired difference with CI ------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    d = {"A - B": gain["A"] - gain["B"], "A - C": gain["A"] - gain["C"]}
    cis = {"A - B": a["primary_gain_vs_resampled_A_minus_B"]["ci"],
           "A - C": a["secondary_gain_vs_resampled_A_minus_C"]["ci"]}
    colmap = {"A - B": "#d62728", "A - C": "#9467bd"}
    for i, name in enumerate(d):
        ci = cis[name]
        j = np.linspace(-0.08, 0.08, len(d[name]))
        ax.plot(i + j, d[name], "o", ms=4, color=colmap[name], alpha=0.75, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [d[name].mean()] * 2, "-", color="black",
                lw=2.5, zorder=4)
        ax.errorbar(i, d[name].mean(),
                    yerr=[[d[name].mean() - ci["lower"]], [ci["upper"] - d[name].mean()]],
                    fmt="none", ecolor=colmap[name], elinewidth=2, capsize=8, zorder=4)
        ax.annotate(f"mediana {d[name].mean():+.4f}\nIC95 [{ci['lower']:.4f}, {ci['upper']:.4f}]",
                    (i, ci["upper"]), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(d))
    ax.set_ylabel("diferenca pareada no ganho sobre reamostragem")
    ax.set_title("3. ganho do canal, A menos cada controle\n"
                 "zero significaria que o receptor nao dependia da topologia")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "c2b_paired_gain.png", dpi=150)
    plt.close(fig)

    # --- fig 3: C2a gradient vs C2b step --------------------------------------
    if a2 is not None:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        x = np.arange(3)
        w = 0.38
        g2a = [a2["descriptive"][c]["mean"] for c in conds]
        g2b = [a["descriptive"][c]["gain_vs_resampled"]["mean"] for c in conds]
        ax.bar(x - w / 2, g2a, w, label="C2a  (8 padroes, decodificacao)",
               color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.bar(x + w / 2, g2b, w, label="C2b  (2 pistas, ganho de canal)",
               color="#ff7f0e", edgecolor="black", alpha=0.85)
        for i, (u, v) in enumerate(zip(g2a, g2b)):
            ax.text(i - w / 2, u + 0.02, f"{u:.3f}", ha="center", fontsize=9)
            ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
            # A zero-height bar is invisible and reads as MISSING data rather than
            # as a measured zero. Mark it, because "the random graph carries
            # nothing" is one of the claims of this figure.
            if v < 0.01:
                ax.plot([i + w / 2 - w / 2, i + w / 2 + w / 2], [0.006, 0.006],
                        "-", color="black", lw=2.5, solid_capstyle="butt", zorder=5)
        ax.axhline(0.125, ls="--", color="crimson", lw=1.3)
        ax.text(2.45, 0.14, "acaso 1/8", color="crimson", fontsize=8, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("desempenho")
        ax.set_ylim(0, 1.08)
        ax.set_title("4. as duas medidas nao se substituem\n"
                     "C2a gradua quanto a anatomia acrescenta; C2b mostra que ela e necessaria")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "c2a_vs_c2b.png", dpi=150)
        plt.close(fig)

    made = sorted(p.name for p in FIG.glob("*.png"))
    print(f"{len(made)} figuras em {FIG}")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())