"""Reproducible publication graphics; no experimental measurements are synthesized."""

import json
import tempfile
import numpy as np
import csv
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#142C3D"
MUTED = "#516570"
RULE = "#CCD7DC"
ACCENT = "#087F8C"
LIGHT = "#EFF7F8"
WARM = "#B86B20"


def style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "publication-20260927",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def canvas(height=4.6):
    fig = plt.figure(figsize=(7.2, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=10, weight="normal", color=INK, ha="left", va="center"):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.45,
    )


def panel(ax, x, y, letter, title):
    label(ax, x, y, letter, 12, "bold", ACCENT)
    label(ax, x + 0.038, y, title, 11, "bold")


def box(ax, x, y, w, h, title, body="", accent=None, face=None, size=9.5):
    accent = ACCENT if accent is None else accent
    face = LIGHT if face is None else face
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=0.7,
            edgecolor=RULE,
            facecolor=face,
        )
    )
    ax.plot(
        [x + 0.015, x + 0.015],
        [y + 0.02, y + h - 0.02],
        color=accent,
        lw=2.1,
        solid_capstyle="round",
    )
    label(ax, x + 0.034, y + h - 0.037, title, 10, "bold", accent, va="top")
    if body:
        label(ax, x + 0.034, y + h - 0.099, body, size, va="top")


def arrow(ax, a, b, color=MUTED, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1,
            color=color,
            linestyle=style,
            shrinkA=2,
            shrinkB=2,
        )
    )


def save(fig, folder, stem, formats=("png", "svg", "pdf")):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        meta = (
            {"Date": None}
            if ext == "svg"
            else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        )
        fig.savefig(folder / f"{stem}.{ext}", dpi=450, metadata=meta)
    plt.close(fig)


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6, color=RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = ROOT / "paper/figures"
ACCENT = "#137E68"
LIGHT = "#EFF8F4"


def workflow():
    fig, ax = canvas(4.8)
    panel(ax, 0.035, 0.95, "a", "Two measurements from image sequences")
    box(
        ax,
        0.035,
        0.685,
        0.27,
        0.195,
        "Inputs",
        "Reference + image series\nRegions and configuration",
        size=9,
    )
    box(
        ax,
        0.385,
        0.695,
        0.58,
        0.185,
        "Virtual extensometer",
        "Track a region pair → gauge-length change\nEngineering strain and true strain",
        size=10,
    )
    box(
        ax,
        0.385,
        0.385,
        0.58,
        0.235,
        "Fixed-reference 2D DIC",
        "Local subsets → displacement → strain\nDisplacement and strain validity kept separate\nInfinitesimal / Green–Lagrange tensor strain",
        size=9.5,
    )
    arrow(ax, (0.305, 0.79), (0.385, 0.79), ACCENT)
    ax.plot([0.325, 0.325], [0.79, 0.505], color=MUTED, lw=1)
    arrow(ax, (0.325, 0.505), (0.385, 0.505), ACCENT)
    label(ax, 0.035, 0.535, "In-plane images", 10, "bold", ACCENT)
    label(
        ax, 0.035, 0.475, "Texture and geometry\nlimit interpretation", 9, color=MUTED
    )
    ax.plot([0.035, 0.965], [0.325, 0.325], color=RULE, lw=0.8)
    panel(ax, 0.035, 0.265, "b", "Outputs retain the analysis context")
    label(ax, 0.035, 0.19, "Tables and maps", 11, "bold", ACCENT)
    label(ax, 0.40, 0.19, "Diagnostics and validity", 11, "bold", ACCENT)
    label(
        ax,
        0.035,
        0.13,
        "Configuration · input identities · source fingerprints · file manifest",
        9.6,
    )
    label(
        ax,
        0.035,
        0.055,
        "Failed measurements remain explicit; synthetic verification does not establish camera uncertainty.",
        8.5,
        color=MUTED,
    )
    save(fig, PAPER, "workflow")


def benchmark_figure(destination):
    from benchmarks.run_benchmark import run_benchmark
    from benchmarks.synthetic_cases import make_case, locked_case_document

    report = run_benchmark(output_dir=destination)
    assert report["overall_pass"], "Locked benchmark failed"
    records = list(
        csv.DictReader((destination / "benchmark_report.csv").open(encoding="utf-8"))
    )
    rows = [
        r
        for r in records
        if r["case_id"] == "small_translation" and r["run_id"] == "clean"
    ]
    case = next(
        c
        for c in locked_case_document()["cases"]
        if c["case_id"] == "small_translation"
    )
    fixture = make_case(case)
    fig = plt.figure(figsize=(7.2, 4.65))
    a = fig.add_axes([0.08, 0.25, 0.35, 0.57])
    b = fig.add_axes([0.59, 0.25, 0.35, 0.57])
    a.imshow(
        fixture["reference"],
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest",
    )
    xy = fixture["coordinates"]
    a.scatter(xy[:, 0], xy[:, 1], facecolors="none", edgecolors="#E69F00", s=9, lw=0.65)
    a.set_title("a   Locked synthetic reference", loc="left", fontweight="bold", pad=13)
    a.set(xlabel="x (pixels)", ylabel="y (pixels)")
    indices = np.array([int(r["point_index"]) for r in rows])
    for key, marker, color, title in [
        ("error_u_px", "o", ACCENT, "u error"),
        ("error_v_px", "s", WARM, "v error"),
    ]:
        vals = np.array([float(r[key]) for r in rows])
        b.plot(indices, vals, marker=marker, ms=3, lw=0, color=color, label=title)
    b.axhline(0, color=MUTED, lw=0.7, ls="--")
    b.set_title("b   Displacement residuals", loc="left", fontweight="bold", pad=13)
    b.set(xlabel="Point index", ylabel="Estimated − prescribed (pixels)")
    b.legend(
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    b.set_ylim(-0.007, 0.041)
    clean_axes(b)
    fig.text(
        0.08,
        0.91,
        "Prescribed translation: u = 2.3 px, v = −1.2 px · 81 evaluation points",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.075,
        "Clean small-translation case from the locked benchmark; no experimental images.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, PAPER, "synthetic_verification")
    # Retain exact plotted observations and the complete benchmark report for audit.
    (PAPER / "synthetic_verification.data.json").write_text(
        json.dumps(
            {
                "case": case,
                "input_sha256": fixture["input_sha256"],
                "points": [
                    {
                        k: r[k]
                        for k in (
                            "point_index",
                            "x",
                            "y",
                            "u_raw",
                            "v_raw",
                            "error_u_px",
                            "error_v_px",
                            "valid",
                        )
                    }
                    for r in rows
                ],
                "scope": "Clean small-translation benchmark only; full report retained separately",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (PAPER / "benchmark_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main():
    style()
    workflow()
    with tempfile.TemporaryDirectory(prefix="straintrace_figures_") as temp:
        benchmark_figure(Path(temp))


if __name__ == "__main__":
    main()
