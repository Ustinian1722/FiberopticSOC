from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from make_q2_publication_figures import (
    COLORS, PROFILES, RATES, WIDTH_IN, add_panel_label, load_dataset,
    mm, save_figure, style_ax,
)


def corr_condition(rho: float) -> float:
    rho = abs(float(rho))
    return (1 + rho) / max(1 - rho, 1e-12)


def plain_log_colorbar(cb) -> None:
    # Keep logarithmic density normalization but avoid mathtext 10^n tick labels,
    # whose superscripts can fall below the 5-pt final-PDF glyph floor.
    cb.formatter = mticker.FuncFormatter(lambda x, _pos: f"{x:g}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6, length=2)


def fix_fig2(data, source_dir: Path, out_dir: Path) -> None:
    all_df = pd.concat(
        [df.assign(profile=p, rate=r) for (p, r), df in data.items()],
        ignore_index=True,
    )
    transfer = pd.read_csv(source_dir / "fig2_representation_transfer.csv")

    fig = plt.figure(figsize=(WIDTH_IN, mm(115)))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    hb1 = ax_a.hexbin(
        all_df["Wavelength_1"], all_df["Wavelength_2"], gridsize=65,
        mincnt=1, cmap="Blues", linewidths=0, bins="log",
    )
    ax_a.set_xlabel("W1 shift (nm)")
    ax_a.set_ylabel("W2 shift (nm)")
    style_ax(ax_a)
    cb1 = fig.colorbar(hb1, ax=ax_a, fraction=0.045, pad=0.025)
    cb1.set_label("Count (log scale)", fontsize=7)
    plain_log_colorbar(cb1)

    hb2 = ax_b.hexbin(
        all_df["temperature_℃"], all_df["force_N"], gridsize=65,
        mincnt=1, cmap="Reds", linewidths=0, bins="log",
    )
    ax_b.set_xlabel("Decoupled temperature (°C)")
    ax_b.set_ylabel("Decoupled force (N)")
    style_ax(ax_b)
    cb2 = fig.colorbar(hb2, ax=ax_b, fraction=0.045, pad=0.025)
    cb2.set_label("Count (log scale)", fontsize=7)
    plain_log_colorbar(cb2)

    bars = []
    conditions = []
    positions = [0.0, 0.36, 1.0, 1.36]
    for rate in RATES:
        d = all_df[all_df["rate"] == rate]
        rho_w = float(d[["Wavelength_1", "Wavelength_2"]].corr().iloc[0, 1])
        rho_tf = float(d[["temperature_℃", "force_N"]].corr().iloc[0, 1])
        bars.extend([abs(rho_w), abs(rho_tf)])
        conditions.extend([corr_condition(rho_w), corr_condition(rho_tf)])
    values = [bars[0], bars[2], bars[1], bars[3]]
    kappas = [conditions[0], conditions[2], conditions[1], conditions[3]]
    labels = ["Raw W\n1C", "Raw W\n2C", "T/F\n1C", "T/F\n2C"]
    colors = [COLORS["blue"], COLORS["blue2"], COLORS["red"], "#D77A72"]
    rects = ax_c.bar(positions, values, width=0.30, color=colors, edgecolor="white", linewidth=0.5)
    for rect, kappa in zip(rects, kappas):
        ax_c.text(
            rect.get_x() + rect.get_width() / 2,
            min(1.015, rect.get_height() + 0.025),
            f"κ={kappa:.1f}", ha="center", va="bottom", fontsize=6.3,
        )
    ax_c.set_xticks(positions, labels)
    ax_c.set_ylabel("Absolute Pearson correlation")
    ax_c.set_ylim(0, 1.10)
    style_ax(ax_c)

    rects = ax_d.bar(
        transfer["representation"], transfer["mae_pct"],
        color=[COLORS["blue"], COLORS["red"], COLORS["gray"]], width=0.66,
    )
    ax_d.set_ylabel("Cross-rate development MAE (% SOC)")
    ax_d.set_ylim(0, float(transfer["mae_pct"].max()) * 1.22)
    for rect, value in zip(rects, transfer["mae_pct"]):
        ax_d.text(
            rect.get_x() + rect.get_width() / 2, value + 0.035,
            f"{value:.2f}", ha="center", va="bottom", fontsize=6.5,
        )
    style_ax(ax_d)

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        add_panel_label(ax, lab)
    fig.suptitle("Dual-FBG representation geometry and predictive transfer", fontsize=9.5, y=0.995)
    save_figure(fig, out_dir, "fig2_representation_analysis")


def fix_fig5(data, source_dir: Path, out_dir: Path) -> None:
    bins = pd.read_csv(source_dir / "fig5_ood_bins.csv")
    held_out = "NEDC"
    train_current = pd.concat(
        [data[(p, "1C")][["Current_A"]] for p in PROFILES if p != held_out],
        ignore_index=True,
    )["Current_A"]
    lo, hi = np.quantile(train_current.to_numpy(), [0.005, 0.995])
    test = data[(held_out, "2C")]
    t = test["Time_s"].to_numpy() / 60.0
    current = test["Current_A"].to_numpy()

    fig = plt.figure(figsize=(WIDTH_IN, mm(75)))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.25, 1.0, 1.0])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax_a.axhspan(lo, hi, color=COLORS["light"], alpha=0.55)
    ax_a.plot(t, current, color=COLORS["blue"], lw=0.75)
    mask = (current < lo) | (current > hi)
    ax_a.fill_between(
        t, current, np.where(current < lo, lo, hi), where=mask,
        color=COLORS["red"], alpha=0.25, interpolate=False,
    )
    ax_a.set_xlabel("Time (min)")
    ax_a.set_ylabel("Current (A)")
    ax_a.set_title("NEDC 2C test current", fontsize=8, pad=4)
    style_ax(ax_a)

    x = np.arange(len(bins))
    w = 0.36
    ax_b.bar(x - w / 2, bins["vi_mae"] * 100, width=w, color=COLORS["gray"], label="VI")
    ax_b.bar(x + w / 2, bins["viw_mae"] * 100, width=w, color=COLORS["blue"], label="VI+W")
    ax_b.set_xticks(x, ["ID", "0–25", "25–50", "50–75", "75–100"])
    for tick in ax_b.get_xticklabels():
        tick.set_rotation(50)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        tick.set_fontsize(6.3)
    ax_b.set_xlabel("Electrical OOD fraction (%)")
    ax_b.set_ylabel("MAE (% SOC)")
    ax_b.legend(loc="upper left")
    style_ax(ax_b)

    gains = bins["relative_optical_gain_pct"].to_numpy()
    ax_c.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_c.plot(x, gains, color=COLORS["red"], marker="o", ms=4.5, lw=1.2)
    ax_c.fill_between(x, 0, gains, where=gains >= 0, color=COLORS["blue"], alpha=0.10)
    ax_c.fill_between(x, 0, gains, where=gains < 0, color=COLORS["red"], alpha=0.10)
    ax_c.set_xticks(x, ["ID", "0–25", "25–50", "50–75", "75–100"])
    for tick in ax_c.get_xticklabels():
        tick.set_rotation(50)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        tick.set_fontsize(6.3)
    ax_c.set_xlabel("Electrical OOD fraction (%)")
    ax_c.set_ylabel("Relative optical gain (%)")
    ax_c.set_ylim(min(-24, float(gains.min()) - 4), max(54, float(gains.max()) + 4))
    style_ax(ax_c)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        add_panel_label(ax, lab)
    fig.suptitle("Optical assistance emerges with electrical distribution shift", fontsize=9.5, y=1.01)
    save_figure(fig, out_dir, "fig5_electrical_ood_optical_gain")


def fix_fig6(source_dir: Path, out_dir: Path) -> None:
    prof = pd.read_csv(source_dir / "fig6_t4_profile_summary.csv")
    boot = pd.read_csv(source_dir / "fig6_t4_bootstrap.csv")
    fig = plt.figure(figsize=(WIDTH_IN, mm(78)))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.08, 1.08, 0.94])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, direction, color, title in [
        (ax_a, "1C_to_2C", COLORS["red"], "1C → 2C"),
        (ax_b, "2C_to_1C", COLORS["blue"], "2C → 1C"),
    ]:
        d = prof[prof["direction"] == direction].set_index("held_out_profile").loc[list(PROFILES)].reset_index()
        x = np.arange(len(d))
        ax.bar(
            x, d["MAE_mean"] * 100, yerr=d["MAE_std"] * 100,
            color=color, alpha=0.84, width=0.68,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "capsize": 2.5},
        )
        ax.set_xticks(x, d["held_out_profile"])
        for tick in ax.get_xticklabels():
            tick.set_rotation(55)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
            tick.set_fontsize(6.0)
        ax.set_ylabel("MAE (% SOC)")
        ax.set_title(title, fontsize=8, pad=4)
        style_ax(ax)

    order = ["1C_to_2C", "2C_to_1C", "overall"]
    labels = ["1C → 2C", "2C → 1C", "Overall"]
    d = boot.set_index("scope").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    vals = d["MAE_mean"].to_numpy() * 100
    lo = d["seed_cluster_bootstrap95_lo"].to_numpy() * 100
    hi = d["seed_cluster_bootstrap95_hi"].to_numpy() * 100
    for yi, val, low, high, color in zip(y, vals, lo, hi, [COLORS["red"], COLORS["blue"], COLORS["dark"]]):
        ax_c.plot([low, high], [yi, yi], color=color, lw=1.4)
        ax_c.plot(val, yi, "o", color=color, ms=5)
    ax_c.set_yticks(y, labels)
    ax_c.set_title("Seed-cluster bootstrap 95% CI", fontsize=8, pad=4)
    ax_c.set_xlabel("MAE (% SOC)")
    ax_c.set_ylim(-0.6, len(y) - 0.4)
    style_ax(ax_c)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        add_panel_label(ax, lab)
    fig.suptitle("Cross-rate unseen-profile generalization across five seeds", fontsize=9.5, y=1.01)
    save_figure(fig, out_dir, "fig6_cross_rate_unseen_profile")


def fix_fig7(source_dir: Path, pred: pd.DataFrame, out_dir: Path) -> None:
    noise = pd.read_csv(source_dir / "fig7_noise_summary.csv")
    uq = pd.read_csv(source_dir / "fig7_uq_summary.csv")
    block_name = "LA92_1C.xlsx::test::2039:2407"
    d = pred[pred["source_name"] == block_name].copy().reset_index(drop=True)
    if d.empty:
        raise ValueError(f"Expected UQ example block not found: {block_name}")
    d["sample"] = np.arange(len(d))

    fig = plt.figure(figsize=(WIDTH_IN, mm(78)))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[0.95, 0.95, 1.25])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for direction, color, label in [
        ("1C_to_2C", COLORS["red"], "1C → 2C"),
        ("2C_to_1C", COLORS["blue"], "2C → 1C"),
    ]:
        q = noise[noise["direction"] == direction].sort_values("sigma_pm_each_wavelength")
        ax_a.plot(
            q["sigma_pm_each_wavelength"], q["MAE_mean"] * 100,
            marker="o", ms=4.2, lw=1.2, color=color, label=label,
        )
        ax_b.plot(
            q["sigma_pm_each_wavelength"], q["Q95_AE_mean"] * 100,
            marker="o", ms=4.2, lw=1.2, color=color, label=label,
        )
    for ax, ylabel in [(ax_a, "MAE (% SOC)"), (ax_b, "Q95 absolute error (% SOC)")]:
        ax.set_xlabel("Wavelength noise σ (pm/channel)")
        ax.set_xticks([0, 0.5, 1, 2])
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        style_ax(ax)

    x = d["sample"].to_numpy()
    ax_c.fill_between(
        x, d["lower"].to_numpy() * 100, d["upper"].to_numpy() * 100,
        color=COLORS["blue2"], alpha=0.18,
    )
    ax_c.plot(x, d["y_true"].to_numpy() * 100, color=COLORS["dark"], lw=1.1)
    ax_c.plot(x, d["y_pred"].to_numpy() * 100, color=COLORS["blue"], lw=0.9, ls="--")
    u95 = uq.loc[np.isclose(uq["nominal_coverage"], 0.95)].iloc[0]
    ax_c.set_title(
        f"95% interval: PICP {u95['PICP'] * 100:.2f}%, MPIW {u95['MPIW'] * 100:.3f}%",
        fontsize=7, pad=5,
    )
    ax_c.set_xlabel("Window index")
    ax_c.set_ylabel("SOC (%)")
    style_ax(ax_c)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    fig.suptitle("FBG noise robustness and calibrated uncertainty", fontsize=9.5, y=1.01)
    save_figure(fig, out_dir, "fig7_noise_and_uncertainty")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    data = load_dataset(a.data_root)
    pred = pd.read_csv(a.predictions)
    fix_fig2(data, a.source_data, a.out_dir)
    fix_fig5(data, a.source_data, a.out_dir)
    fix_fig6(a.source_data, a.out_dir)
    fix_fig7(a.source_data, pred, a.out_dir)


if __name__ == "__main__":
    main()
