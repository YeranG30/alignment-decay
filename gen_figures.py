#!/usr/bin/env python3
"""
gen_figures.py — generate all 5 paper figures from local per-model data.

Fig 1  fig1_landscape   — attack universality: heatmap + Arm A vs B bars
Fig 2  fig2_srd         — SRD asymmetry: C3 omission decay vs C4 commission stable
Fig 3  fig3_causal      — causal isolation: 3-arm design (Gemini + Llama)
Fig 4  fig4_survival    — safe-turn-depth survival curves, all 12 models
Fig 5  fig5_constraints — per-constraint compliance heatmap (Mistral/Nemotron/Qwen)

Usage:
    python3 experiments/tasef/gen_figures.py
    python3 experiments/tasef/gen_figures.py --data-dir data --out-dir paper_data/figures
    python3 experiments/tasef/gen_figures.py --figs 1 2          # only figs 1 and 2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "paper_data" / "figures"

TURNS = [5, 10, 13, 16, 20, 25]
ARM_A = "no_dilution_control"
ARM_B = "tool_schema_dilution"
ARM_C = "token_matched_padding"
SKIP  = {"ablation_reframe", "test_nim", "deepseek"}

MODEL_DISPLAY = {
    "gemini":   "Gemini 2.5 Flash",
    "gpt41":    "GPT-4.1",
    "gpt4o":    "GPT-4o",
    "haiku":    "Claude Haiku",
    "kimi":     "Kimi K2.5",
    "llama":    "Llama 3.3 70B",
    "mini":     "GPT-4o-mini",
    "mistral":  "Mistral Large 3",
    "nemotron": "Nemotron Super 120B",
    "qwen":     "Qwen 3.5 397B",
    "sonnet":   "Claude Sonnet 4",
    "gemma":    "Gemma 4 31B",
}

# Constraints ordered: omission first (C3 at top — it's the key finding), then commission
OMISSION   = ["C3_NO_BULLETS", "C5_NO_HEADERS", "C6_NO_FIRST_PERSON",
              "C7_NO_CODE_BLOCKS", "C9_NO_HOWEVER"]
COMMISSION = ["C1_AUDIT_OK", "C4_STATUS_HEADER", "C8_INCIDENT_ID"]
CONSTRAINTS = OMISSION + COMMISSION

CONSTRAINT_LABELS = {
    "C3_NO_BULLETS":      "C3: no bullets  [omission]",
    "C5_NO_HEADERS":      "C5: no headers  [omission]",
    "C6_NO_FIRST_PERSON": "C6: no 1st-pers [omission]",
    "C7_NO_CODE_BLOCKS":  "C7: no code blk [omission]",
    "C9_NO_HOWEVER":      "C9: no 'however'[omission]",
    "C1_AUDIT_OK":        "C1: no audit-OK [commission]",
    "C4_STATUS_HEADER":   "C4: status hdr  [commission]",
    "C8_INCIDENT_ID":     "C8: incident ID [commission]",
}

SRD_MODELS  = ["mistral", "nemotron", "qwen"]
SRD_COLORS  = {"mistral": "#1f77b4", "nemotron": "#d62728", "qwen": "#2ca02c"}
SRD_MARKERS = {"mistral": "o",       "nemotron": "s",       "qwen": "^"}
SRD_LS      = {"mistral": "-",       "nemotron": "--",      "qwen": ":"}

IMMUNE_MODEL  = "gemma"
IMMUNE_COLOR  = "#888888"
IMMUNE_MARKER = "D"
IMMUNE_LS     = "-"

CAUSAL_MODELS = {"gemini": "Gemini 2.5 Flash", "llama": "Llama 3.3 70B"}

PROVIDER_COLORS = {
    "gemini":   "#1a9850", "gemma":    "#91cf60",
    "gpt4o":    "#d73027", "gpt41":    "#f46d43", "mini":     "#fdae61",
    "haiku":    "#4575b4", "sonnet":   "#74add1",
    "mistral":  "#762a83", "nemotron": "#9970ab",
    "qwen":     "#e08214", "kimi":     "#8c510a",
    "llama":    "#35978f",
}
PROVIDER_LS = {
    "gemini": "-",  "gemma": "--",
    "gpt4o": "-",   "gpt41": "--",  "mini": ":",
    "haiku": "-",   "sonnet": "--",
    "mistral": "-", "nemotron": "--",
    "qwen": "-",    "kimi": "-.",   "llama": ":",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(data_dir: Path) -> dict:
    """Returns data[model_key][arm] = list of {exploit, turn, cr}."""
    data: dict = {}
    for mdir in sorted(data_dir.iterdir()):
        if not mdir.is_dir() or mdir.name in SKIP:
            continue
        if mdir.name not in MODEL_DISPLAY:
            continue
        jsonl = mdir / "results.jsonl"
        if not jsonl.exists():
            continue

        by_arm: dict[str, list] = defaultdict(list)
        for line in open(jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            vt = r.get("vector_type", "")
            if vt not in (ARM_A, ARM_B, ARM_C):
                continue
            by_arm[vt].append({
                "exploit": bool(r.get("exploit_detected", False)),
                "turn":    r.get("injection_turn", 0),
                "cr":      r.get("constraint_results") or {},
            })

        if by_arm:
            data[mdir.name] = dict(by_arm)
            counts = {arm: len(recs) for arm, recs in data[mdir.name].items()}
            print(f"  {mdir.name:<12}  {counts}")

    return data


def exploit_by_turn(records: list) -> dict[int, float]:
    by_t: dict[int, list] = defaultdict(list)
    for r in records:
        by_t[r["turn"]].append(float(r["exploit"]))
    return {t: sum(v) / len(v) for t, v in by_t.items() if v}


def compliance_by_turn(records: list, constraint: str) -> dict[int, float]:
    by_t: dict[int, list] = defaultdict(list)
    for r in records:
        if constraint in r["cr"]:
            by_t[r["turn"]].append(float(r["cr"][constraint]))
    return {t: sum(v) / len(v) for t, v in by_t.items() if v}


def mean_rate(records: list, key: str = "exploit") -> float | None:
    vals = [float(r[key]) for r in records]
    return sum(vals) / len(vals) if vals else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  → {p}")
    plt.close(fig)


# ── Figure 1: Attack landscape ────────────────────────────────────────────────

def fig1_landscape(data: dict, out_dir: Path):
    print("\n[Fig 1] Attack landscape")

    # Sort models by mean Arm B exploit rate descending
    scored = {m: (mean_rate(data[m].get(ARM_B, [])) or 0) for m in data if ARM_B in data[m]}
    models = sorted(scored, key=lambda m: -scored[m])

    n_m = len(models)
    n_t = len(TURNS)

    mat_b = np.full((n_m, n_t), np.nan)
    mean_a_vals, mean_b_vals = [], []

    for i, m in enumerate(models):
        rb = exploit_by_turn(data[m].get(ARM_B, []))
        ra = exploit_by_turn(data[m].get(ARM_A, []))
        for j, t in enumerate(TURNS):
            if t in rb:
                mat_b[i, j] = rb[t]
        mean_a_vals.append(mean_rate(data[m].get(ARM_A, [])) or 0)
        mean_b_vals.append(mean_rate(data[m].get(ARM_B, [])) or 0)

    ylabels = [MODEL_DISPLAY[m] for m in models]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.38, left=0.04, right=0.97, top=0.88, bottom=0.12)

    # ── Left: heatmap ──────────────────────────────────────────────────────────
    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad("whitesmoke")
    im = ax_l.imshow(mat_b, aspect="auto", cmap=cmap_heat, vmin=0, vmax=1,
                     interpolation="nearest")
    ax_l.set_xticks(range(n_t))
    ax_l.set_xticklabels([f"T={t}" for t in TURNS], fontsize=8.5)
    ax_l.set_yticks(range(n_m))
    ax_l.set_yticklabels(ylabels, fontsize=8.5)
    ax_l.set_xlabel("Injection turn depth", fontsize=9)
    ax_l.set_title("Exploit rate under CEI attack  (Arm B)", fontsize=11, fontweight="bold")

    for i in range(n_m):
        for j in range(n_t):
            v = mat_b[i, j]
            if not np.isnan(v):
                fc = "white" if v > 0.55 else "black"
                ax_l.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                          fontsize=7.5, color=fc, fontweight="bold")

    cb = fig.colorbar(im, ax=ax_l, fraction=0.025, pad=0.02)
    cb.set_label("Exploit rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Right: paired horizontal bar chart ────────────────────────────────────
    y = np.arange(n_m)
    h = 0.36

    ax_r.barh(y + h / 2, mean_a_vals, h,
              color="#2171b5", alpha=0.85, label="Arm A  (no dilution)")
    bars_b = ax_r.barh(y - h / 2, mean_b_vals, h,
                       color="#cb181d", alpha=0.85, label="Arm B  (schema dilution)")

    for bar, v in zip(bars_b, mean_b_vals):
        if v > 0.04:
            ax_r.text(min(v + 0.015, 1.02), bar.get_y() + bar.get_height() / 2,
                      f"{v*100:.0f}%", va="center", fontsize=7)

    ax_r.set_yticks(y)
    ax_r.set_yticklabels(ylabels, fontsize=8.5)
    ax_r.set_xlim(0, 1.15)
    ax_r.set_xlabel("Mean exploit rate", fontsize=9)
    ax_r.set_title("Schema dilution amplifies exploit rate  (A → B)", fontsize=11, fontweight="bold")
    ax_r.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_r.legend(fontsize=8.5, loc="lower right", frameon=False)
    clean_ax(ax_r)

    fig.suptitle("CEI attack exploits all tested LLMs; schema dilution amplifies the effect across all model families",
                 fontsize=11, y=0.97)
    save_fig(fig, out_dir, "fig1_landscape")


# ── Figure 2: SRD asymmetry ───────────────────────────────────────────────────

def fig2_srd(data: dict, out_dir: Path):
    print("\n[Fig 2] SRD asymmetry")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.82, bottom=0.14)

    def _plot_model(ax, m, color, marker, ls, label, recs, constraint):
        rates = compliance_by_turn(recs, constraint)
        xs = [t for t in TURNS if t in rates]
        ys = [rates[t] for t in xs]
        if xs:
            ax.plot(xs, ys, color=color, lw=2.2, ls=ls,
                    marker=marker, markersize=5.5,
                    label=label, zorder=3)

    # SRD models
    for m in SRD_MODELS:
        if m not in data:
            print(f"  WARNING: {m} not in data — skipping")
            continue
        recs = data[m].get(ARM_A, []) + data[m].get(ARM_B, [])
        for ax, constraint in [(ax_l, "C3_NO_BULLETS"), (ax_r, "C4_STATUS_HEADER")]:
            _plot_model(ax, m, SRD_COLORS[m], SRD_MARKERS[m], SRD_LS[m],
                        MODEL_DISPLAY[m], recs, constraint)

    # Gemma immune control
    if IMMUNE_MODEL in data:
        recs = data[IMMUNE_MODEL].get(ARM_A, []) + data[IMMUNE_MODEL].get(ARM_B, [])
        for ax, constraint in [(ax_l, "C3_NO_BULLETS"), (ax_r, "C4_STATUS_HEADER")]:
            _plot_model(ax, IMMUNE_MODEL, IMMUNE_COLOR, IMMUNE_MARKER, IMMUNE_LS,
                        f"{MODEL_DISPLAY[IMMUNE_MODEL]}  [immune]", recs, constraint)

    # Panel titles — short, no overlap with suptitle
    ax_l.set_title("Omission constraint  (C3: no bullets)\n"
                   "SRD models decay · Gemma stays flat",
                   fontsize=10, fontweight="bold", pad=6)
    ax_r.set_title("Commission constraint  (C4: status header)\n"
                   "All models stable  [asymmetry confirmed]",
                   fontsize=10, fontweight="bold", pad=6)

    for ax, ylabel, leg_loc in [
        (ax_l, "Compliance rate  (C3, omission)",   "lower left"),
        (ax_r, "Compliance rate  (C4, commission)",  "lower right"),
    ]:
        ax.set_xlim(3, 27)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xticks(TURNS)
        ax.set_xlabel("Injection turn depth", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.legend(fontsize=8.5, frameon=True, framealpha=0.92,
                  edgecolor="none", loc=leg_loc)
        ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        clean_ax(ax)

    fig.suptitle(
        "Gemma as immune control: SRD is a real asymmetry, not a measurement artifact",
        fontsize=11, fontweight="bold", y=0.98,
    )
    save_fig(fig, out_dir, "fig2_srd")


# ── Figure 3: Causal isolation ────────────────────────────────────────────────

def fig3_causal(data: dict, out_dir: Path):
    print("\n[Fig 3] Causal isolation")

    ARM_COLORS  = {ARM_A: "#238b45", ARM_B: "#6a51a3", ARM_C: "#d94801"}
    ARM_LABELS  = {
        ARM_A: "Arm A  (no dilution)",
        ARM_B: "Arm B  (schema dilution)",
        ARM_C: "Arm C  (token-matched padding)",
    }
    ARM_MARKERS = {ARM_A: "o", ARM_B: "s", ARM_C: "^"}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, top=0.88, bottom=0.14)

    # ── Left: Gemini A vs B by turn ───────────────────────────────────────────
    gk = "gemini"
    if gk in data:
        for arm in (ARM_A, ARM_B):
            rates = exploit_by_turn(data[gk].get(arm, []))
            xs = [t for t in TURNS if t in rates]
            ys = [rates[t] for t in xs]
            if xs:
                ax_l.plot(xs, ys, color=ARM_COLORS[arm], lw=2.2,
                          marker=ARM_MARKERS[arm], markersize=6,
                          label=ARM_LABELS[arm], zorder=3)

        # Shade A→B gap
        ra = exploit_by_turn(data[gk].get(ARM_A, []))
        rb = exploit_by_turn(data[gk].get(ARM_B, []))
        valid = [t for t in TURNS if t in ra and t in rb]
        if valid:
            ax_l.fill_between(valid, [ra[t] for t in valid], [rb[t] for t in valid],
                              alpha=0.12, color="#6a51a3")

    ax_l.set_title("Gemini 2.5 Flash — Arm A vs B", fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Injection turn depth", fontsize=9)
    ax_l.set_ylabel("Exploit rate", fontsize=9)
    ax_l.set_xlim(3, 27)
    ax_l.set_ylim(-0.03, 0.90)
    ax_l.set_xticks(TURNS)
    ax_l.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax_l.legend(fontsize=8.5, loc="upper left", frameon=False)
    clean_ax(ax_l)

    # ── Right: A/B/C mean bars — Gemini + Llama ───────────────────────────────
    causal = [(m, CAUSAL_MODELS[m]) for m in CAUSAL_MODELS if m in data]
    arms   = [ARM_A, ARM_B, ARM_C]
    n_m, n_a = len(causal), len(arms)
    bw = 0.22
    x_centers = np.arange(n_m) * 0.9
    offsets   = np.linspace(-(n_a - 1) / 2, (n_a - 1) / 2, n_a) * bw

    for i, arm in enumerate(arms):
        means = [mean_rate(data[m].get(arm, [])) or 0 for m, _ in causal]
        bars  = ax_r.bar(x_centers + offsets[i], means,
                         width=bw * 0.9, color=ARM_COLORS[arm],
                         alpha=0.88, label=ARM_LABELS[arm], zorder=3)
        for bar, val in zip(bars, means):
            if val > 0.02:
                ax_r.text(bar.get_x() + bar.get_width() / 2,
                          val + 0.013, f"{val:.2f}",
                          ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_r.set_title("Arm A / B / C mean exploit rate", fontsize=11, fontweight="bold")
    ax_r.set_ylabel("Mean exploit rate", fontsize=9)
    ax_r.set_xticks(x_centers)
    ax_r.set_xticklabels([name for _, name in causal], fontsize=9)
    ax_r.set_ylim(0, 0.75)
    ax_r.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax_r.yaxis.grid(True, lw=0.5, alpha=0.5, zorder=0)
    ax_r.set_axisbelow(True)
    clean_ax(ax_r)

    fig.suptitle(
        "Causal isolation: schema dilution (B) > token padding (C) > no dilution (A) — "
        "semantic content drives the effect",
        fontsize=10.5, y=0.97,
    )
    save_fig(fig, out_dir, "fig3_causal")


# ── Figure 4: Survival curves ─────────────────────────────────────────────────

def fig4_survival(data: dict, out_dir: Path):
    print("\n[Fig 4] Survival curves")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.09, right=0.72, top=0.88, bottom=0.13)

    # Compute mean Arm B exploit rate for end-label vertical sorting
    end_vals: list[tuple[float, str]] = []
    for m in data:
        if ARM_B not in data[m]:
            continue
        rates = exploit_by_turn(data[m][ARM_B])
        last = rates.get(TURNS[-1]) or rates.get(TURNS[-2])
        survival_end = 1 - (last or 0)
        end_vals.append((survival_end, m))
    end_vals.sort(key=lambda x: -x[0])  # top-to-bottom by final survival

    for survival_end, m in end_vals:
        rates = exploit_by_turn(data[m][ARM_B])
        xs = [t for t in TURNS if t in rates]
        ys = [1 - rates[t] for t in xs]
        if not xs:
            continue

        line, = ax.plot(xs, ys,
                        color=PROVIDER_COLORS.get(m, "gray"),
                        ls=PROVIDER_LS.get(m, "-"),
                        lw=2, marker="o", markersize=4,
                        zorder=3)

        # Right-side direct label instead of legend
        y_end = ys[-1]
        ax.text(TURNS[-1] + 0.4, y_end, MODEL_DISPLAY[m],
                va="center", ha="left", fontsize=7.5,
                color=PROVIDER_COLORS.get(m, "gray"))

    ax.axhline(0.5, color="gray", lw=1, ls=":", alpha=0.7)
    ax.text(4, 0.515, "Safe Turn Depth threshold  (50%)",
            ha="left", va="bottom", fontsize=7.5, color="gray")

    ax.set_xlim(3, 27)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(TURNS)
    ax.set_xlabel("Injection turn depth", fontsize=10)
    ax.set_ylabel("P(not exploited)  under Arm B", fontsize=10)
    ax.set_title("Safe Turn Depth — survival under CEI attack (Arm B)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    clean_ax(ax)

    save_fig(fig, out_dir, "fig4_survival")


# ── Figure 5: Per-constraint compliance heatmap ───────────────────────────────

def fig5_constraints(data: dict, out_dir: Path):
    print("\n[Fig 5] Per-constraint heatmap")

    available = [m for m in SRD_MODELS if m in data]
    if not available:
        print("  No SRD models found — skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.38, left=0.14, right=0.92, top=0.88, bottom=0.16)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("whitesmoke")
    im_last = None

    for ax, m in zip(axes, available):
        recs     = data[m].get(ARM_A, []) + data[m].get(ARM_B, [])
        cr_recs  = [r for r in recs if r["cr"]]

        mat = np.full((len(CONSTRAINTS), len(TURNS)), np.nan)
        for i, c in enumerate(CONSTRAINTS):
            cd = compliance_by_turn(cr_recs, c)
            for j, t in enumerate(TURNS):
                if t in cd:
                    mat[i, j] = cd[t]

        im_last = ax.imshow(mat, aspect="auto", cmap=cmap,
                            vmin=0, vmax=1, interpolation="nearest")

        ax.set_xticks(range(len(TURNS)))
        ax.set_xticklabels([f"T={t}" for t in TURNS], fontsize=8)
        ax.set_yticks(range(len(CONSTRAINTS)))
        ax.set_yticklabels([CONSTRAINT_LABELS[c] for c in CONSTRAINTS], fontsize=8)
        ax.set_xlabel("Turn depth", fontsize=9)
        ax.set_title(MODEL_DISPLAY[m], fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(len(CONSTRAINTS)):
            for j in range(len(TURNS)):
                v = mat[i, j]
                if not np.isnan(v):
                    fc = "white" if (v < 0.3 or v > 0.75) else "black"
                    ax.text(j, i, f"{v*100:.0f}%",
                            ha="center", va="center", fontsize=7, color=fc)

        # Separator line between omission and commission blocks
        sep = len(OMISSION) - 0.5
        ax.axhline(sep, color="black", lw=1.8, ls="--", alpha=0.55)

    if im_last is not None:
        cb = fig.colorbar(im_last, ax=axes, fraction=0.012, pad=0.02)
        cb.set_label("Compliance rate", fontsize=9)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "C3 (omission) degrades with depth; commission constraints C4/C8 remain compliant  "
        "(dashed line separates omission from commission)",
        fontsize=10.5, y=0.97,
    )
    save_fig(fig, out_dir, "fig5_constraints")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Per-model data directory  (default: <script_dir>/data)")
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR,
                        help="Output directory for figures  (default: paper_data/figures)")
    parser.add_argument("--figs", nargs="*", type=int, default=[1, 2, 3, 4, 5],
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}\n")
    data = load_all(args.data_dir)
    print(f"\nLoaded {len(data)} models: {sorted(data.keys())}\n")

    dispatch = {
        1: fig1_landscape,
        2: fig2_srd,
        3: fig3_causal,
        4: fig4_survival,
        5: fig5_constraints,
    }

    for n in sorted(set(args.figs)):
        if n in dispatch:
            dispatch[n](data, args.out_dir)
        else:
            print(f"Unknown figure number: {n}")

    print(f"\nDone. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
