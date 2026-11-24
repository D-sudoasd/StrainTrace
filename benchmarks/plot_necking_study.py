"""Plot recorded necking evidence without interpolating missing measurements."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks._common import sha256_file  # noqa: E402


def _float(value):
    return float(value) if value not in (None, "") else float("nan")


def _flag(value):
    return str(value).strip().lower() in {"1", "true"}


def plot(report_path, output):
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report["pointwise_csv"]
    if Path(artifact["filename"]).name != artifact["filename"]:
        raise ValueError("Expected adjacent pointwise CSV")
    points_path = report_path.parent / artifact["filename"]
    if sha256_file(points_path) != artifact["sha256"]:
        raise ValueError("Pointwise evidence does not match its recorded SHA-256")
    with points_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != artifact["row_count"]:
        raise ValueError("Pointwise row count changed")
    if output.exists():
        raise FileExistsError(f"Use a new figure path: {output}")

    def case(name):
        selected = [row for row in rows if row.get("case") == name]
        if not selected:
            raise ValueError(f"Missing required case in pointwise CSV: {name}")
        return selected

    def line(name):
        selected = [row for row in case(name) if np.isclose(_float(row["x_px"]), 80.0)]
        selected.sort(key=lambda row: _float(row["y_px"]))
        return selected

    required_b = ["broad_localization"] + [f"broad_texture_mix_{x:g}" for x in (.25, .5, .75, 1.)]
    missing_b = [name for name in required_b if name not in report.get("cases", {})]
    if missing_b:
        raise ValueError(f"Missing required cases in report JSON: {missing_b}")

    colors = {"blue": "#2464a0", "orange": "#cf661f", "green": "#287b65",
              "red": "#b53542", "gray": "#bfc6ce", "purple": "#8d72a6"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        try:
            ax = axes[0]
            base = line("narrow_localization")
            base_y = np.array([_float(r["y_px"]) for r in base], dtype=float)
            ax.plot(base_y, np.array([_float(r["true_Eyy"]) for r in base], dtype=float), "o-", color="#26313b",
                    label="Analytic strain at POIs", markersize=4)
            ax.plot(base_y, np.array([_float(r["exact_displacement_full_Eyy"]) for r in base], dtype=float), "--",
                    color=colors["green"], linewidth=2, label="Exact displacement, window 5")
            for name, label, color in (("narrow_window3", "DIC, window 3", colors["orange"]),
                                       ("narrow_localization", "DIC, window 5", colors["blue"]),
                                       ("narrow_window7", "DIC, window 7", colors["purple"])):
                data = line(name)
                ys = np.array([_float(r["y_px"]) for r in data], dtype=float)
                values = np.array([_float(r["Eyy"]) if _flag(r["strain_valid"]) else np.nan for r in data], dtype=float)
                ax.plot(ys, values, "o-", label=label, color=color, markersize=3)
            ax.set(xlabel="Reference y coordinate (px)", ylabel="Green–Lagrange Eyy (1)",
                   title="A  High coverage can hide peak loss", ylim=(0, .35))
            ax.legend(fontsize=8, frameon=False, loc="upper left")
            ax.text(.02, .02, "x = 80 px; subset 25 px; step 8 px\nNo added displacement smoothing",
                    transform=ax.transAxes, fontsize=8, va="bottom")

            ax = axes[1]
            fractions = np.array([0, .25, .5, .75, 1.])
            entire = [report["cases"][name]["regions"]["entire_poi_domain"]["strain_coverage"] for name in required_b]
            neck = [report["cases"][name]["regions"]["neck_band"]["strain_coverage"] for name in required_b]
            entire_n = report["cases"][required_b[0]]["regions"]["entire_poi_domain"]["eligible_count"]
            neck_n = report["cases"][required_b[0]]["regions"]["neck_band"]["eligible_count"]
            ax.plot(fractions, 100*np.array(entire, dtype=float), "o-", color=colors["blue"], label=f"Whole domain ({entire_n} POIs)")
            ax.plot(fractions, 100*np.array(neck, dtype=float), "s-", color=colors["orange"], label=f"Neck band ({neck_n} POIs)")
            ax.set(xlabel="Local unrelated-texture replacement fraction", ylabel="Valid strain coverage (%)",
                   title="B  Local coverage falls first", ylim=(0, 106), xticks=fractions)
            ax.legend(fontsize=8, frameon=False, loc="lower left")
            ax.text(.98, .95, "Fixed underlying deformation\nControlled texture-loss stress test",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top")

            ax = axes[2]
            data = case("broad_texture_mix_0.5")
            xs = np.array([_float(r["x_px"]) for r in data], dtype=float)
            ys = np.array([_float(r["y_px"]) for r in data], dtype=float)
            accepted = np.array([_flag(r["correlation_valid"]) for r in data], dtype=bool)
            u_vals = np.array([_float(r["u_px"]) for r in data], dtype=float)
            v_vals = np.array([_float(r["v_px"]) for r in data], dtype=float)
            tu_vals = np.array([_float(r["true_u_px"]) for r in data], dtype=float)
            tv_vals = np.array([_float(r["true_v_px"]) for r in data], dtype=float)
            errors = np.hypot(u_vals - tu_vals, v_vals - tv_vals)
            finite_err = np.isfinite(errors)
            masks = [(~accepted, "Rejected (kept missing)", colors["gray"], "x"),
                     (accepted & finite_err & (errors <= .5), "Accepted, error ≤ 0.5 px", colors["blue"], "o"),
                     (accepted & finite_err & (errors > .5), "Accepted, error > 0.5 px", colors["red"], "s")]
            folded = accepted & ~finite_err
            if folded.any():
                masks.append((folded, "Accepted, no published displacement", colors["purple"], "^"))
            for mask, label, color, marker in masks:
                ax.scatter(xs[mask], ys[mask], s=23, marker=marker,
                           color=color, label=f"{label} ({int(mask.sum())})")
            ax.set(xlabel="Reference x coordinate (px)", ylabel="Reference y coordinate (px)",
                   title="C  Acceptance still needs scrutiny", aspect="equal")
            ax.invert_yaxis()
            ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.17))
            ax.text(.02, .98, "50% local texture replacement\n0.5 px is a diagnostic cutoff only",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
            for ax in axes[:2]:
                ax.grid(alpha=.18)
            fig.suptitle("Known-truth tests: strain-window bias and local tracking failures", fontsize=14)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=190, metadata={"Description": f"Source report SHA-256: {sha256_file(report_path)}"})
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
