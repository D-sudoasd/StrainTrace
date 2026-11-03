"""Plot recorded necking evidence without interpolating missing measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_path, output):
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report["pointwise_csv"]
    if Path(artifact["filename"]).name != artifact["filename"]:
        raise ValueError("Expected adjacent pointwise CSV")
    points_path = report_path.parent / artifact["filename"]
    if hashlib.sha256(points_path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Pointwise evidence does not match its recorded SHA-256")
    points = np.genfromtxt(points_path, delimiter=",", names=True, dtype=None,
                          encoding="utf-8", filling_values=np.nan)
    if len(points) != artifact["row_count"]:
        raise ValueError("Pointwise row count changed")
    if output.exists():
        raise FileExistsError(f"Use a new figure path: {output}")

    def case(name):
        return points[points["case"] == name]

    def line(name):
        selected = case(name)
        selected = selected[np.isclose(selected["x_px"], 80.0)]
        return np.sort(selected, order="y_px")

    colors = {"blue": "#2464a0", "orange": "#cf661f", "green": "#287b65",
              "red": "#b53542", "gray": "#bfc6ce"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        ax = axes[0]
        base = line("narrow_localization")
        ax.plot(base["y_px"], base["true_Eyy"], "o-", color="#26313b",
                label="Analytic strain at POIs", markersize=4)
        ax.plot(base["y_px"], base["exact_displacement_full_Eyy"], "--",
                color=colors["green"], linewidth=2, label="Exact displacement, window 5")
        for name, label, color in (("narrow_window3", "DIC, window 3", colors["orange"]),
                                   ("narrow_localization", "DIC, window 5", colors["blue"]),
                                   ("narrow_window7", "DIC, window 7", "#8d72a6")):
            data = line(name)
            values = np.where(data["strain_valid"].astype(bool), data["Eyy"], np.nan)
            ax.plot(data["y_px"], values, "o-", label=label, color=color, markersize=3)
        ax.set(xlabel="Reference y coordinate (px)", ylabel="Green–Lagrange Eyy (1)",
               title="A  High coverage can hide peak loss", ylim=(0, .35))
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.text(.02, .02, "x = 80 px; subset 25 px; step 8 px\nNo added displacement smoothing",
                transform=ax.transAxes, fontsize=8, va="bottom")

        ax = axes[1]
        names = ["broad_localization"] + [f"broad_texture_mix_{x:g}" for x in (.25, .5, .75, 1.)]
        fractions = np.array([0, .25, .5, .75, 1.])
        entire = [report["cases"][name]["regions"]["entire_poi_domain"]["strain_coverage"] for name in names]
        neck = [report["cases"][name]["regions"]["neck_band"]["strain_coverage"] for name in names]
        ax.plot(fractions, 100*np.array(entire), "o-", color=colors["blue"], label="Whole domain (144 POIs)")
        ax.plot(fractions, 100*np.array(neck), "s-", color=colors["orange"], label="Neck band (60 POIs)")
        ax.set(xlabel="Local unrelated-texture replacement fraction", ylabel="Valid strain coverage (%)",
               title="B  Local coverage falls first", ylim=(0, 106), xticks=fractions)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.text(.98, .95, "Fixed underlying deformation\nControlled texture-loss stress test",
                transform=ax.transAxes, fontsize=8, ha="right", va="top")

        ax = axes[2]
        data = case("broad_texture_mix_0.5")
        accepted = data["correlation_valid"].astype(bool)
        errors = np.hypot(data["u_px"]-data["true_u_px"], data["v_px"]-data["true_v_px"])
        masks = [(~accepted, "Rejected (kept missing)", colors["gray"], "x"),
                 (accepted & (errors <= .5), "Accepted, error ≤ 0.5 px", colors["blue"], "o"),
                 (accepted & (errors > .5), "Accepted, error > 0.5 px", colors["red"], "s")]
        for mask, label, color, marker in masks:
            ax.scatter(data["x_px"][mask], data["y_px"][mask], s=23, marker=marker,
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
        fig.savefig(output, dpi=190, metadata={"Description": f"Source report SHA-256: {hashlib.sha256(report_path.read_bytes()).hexdigest()}"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
