from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from make_q2_publication_figures import (
    COLORS, RATES, WIDTH_IN, add_panel_label, load_dataset, mm, save_figure, style_ax,
)


def plain_log_colorbar(cb) -> None:
    cb.formatter = mticker.FuncFormatter(lambda x, _pos: f"{x:g}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6, length=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    data = load_dataset(a.data_root)
    all_df = pd.concat(
        [df.assign(profile=profile, rate=rate) for (profile, rate), df in data.items()],
        ignore_index=True,
    )
    transfer = pd.read_csv(a.source_data / "fig2_representation_transfer.csv")

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

    positions = [0.0, 0.36, 1.0, 1.36]
    values = []
    for representation in ("W", "TF"):
        for rate in RATES:
            d = all_df[all_df["rate"] == rate]
            if representation == "W":
                rho = float(d[["Wavelength_1", "Wavelength_2"]].corr().iloc[0, 1])
            else:
                rho = float(d[["temperature_℃", "force_N"]].corr().iloc[0, 1])
            values.append(abs(rho))
    labels = ["Raw W\n1C", "Raw W\n2C", "T/F\n1C", "T/F\n2C"]
    colors = [COLORS["blue"], COLORS["blue2"], COLORS["red"], "#D77A72"]
    rects = ax_c.bar(positions, values, width=0.30, color=colors, edgecolor="white", linewidth=0.5)
    for rect, value in zip(rects, values):
        ax_c.text(
            rect.get_x() + rect.get_width() / 2,
            min(1.02, value + 0.025),
            f"{value:.3f}", ha="center", va="bottom", fontsize=6.3,
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

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        add_panel_label(ax, label)
    fig.suptitle("Dual-FBG representation geometry and predictive transfer", fontsize=9.5, y=0.995)
    save_figure(fig, a.out_dir, "fig2_representation_analysis")


if __name__ == "__main__":
    main()
