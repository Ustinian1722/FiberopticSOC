from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project typography override: manuscript/figures use Times New Roman-style serif.
# Upstream nature-figure defaults to Arial/Helvetica/Liberation Sans; we retain its
# editable-SVG, physical-size, source-data and QA principles but keep project typography.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

WIDTH_MM = 183
WIDTH_IN = WIDTH_MM / 25.4
DPI = 600
COLORS = {
    "blue": "#0F4D92", "blue2": "#3775BA", "teal": "#42949E",
    "red": "#B64342", "gray": "#767676", "dark": "#272727",
    "light": "#CFCECE", "green": "#5B8E55",
}
PROFILES = ("HWFET", "LA92", "NEDC", "NYCC", "US06", "WLTC")
RATES = ("1C", "2C")


def mm(v: float) -> float:
    return v / 25.4


def add_panel_label(ax, label: str, x: float = 0, y: float = 1,
                    x_offset_pt: float = -4, y_offset_pt: float = 3) -> None:
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + offset, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def style_ax(ax) -> None:
    ax.tick_params(direction="out", width=0.8, length=3, pad=2)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)


def save_figure(fig, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": out_dir / f"{stem}.svg",
        "pdf": out_dir / f"{stem}.pdf",
        "tiff": out_dir / f"{stem}.tiff",
        "png": out_dir / f"{stem}.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=DPI, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def find_workbooks(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.xlsx")):
        if "_" not in path.stem:
            continue
        profile, rate = path.stem.rsplit("_", 1)
        if profile in PROFILES and rate in RATES:
            files.append(path)
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 SiC-18 workbooks under {root}, found {len(files)}")
    return files


def load_dataset(root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    needed = ["Time_s", "Voltage_V", "Current_A", "Wavelength_1", "Wavelength_2",
              "temperature_℃", "force_N", "SOC"]
    out = {}
    for path in find_workbooks(root):
        profile, rate = path.stem.rsplit("_", 1)
        df = pd.read_excel(path)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing {missing}")
        out[(profile, rate)] = df[needed].copy()
    return out


def fig1_dataset_signals(data, out_dir: Path) -> dict:
    profile = "NEDC"
    d1, d2 = data[(profile, "1C")], data[(profile, "2C")]
    fig = plt.figure(figsize=(WIDTH_IN, mm(112)))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.38)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax_a.set_axis_off()
    ax_a.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.84, transform=ax_a.transAxes,
                                 facecolor="#F7F7F7", edgecolor=COLORS["light"], linewidth=1.0))
    ax_a.text(0.5, 0.55, "Experimental setup and\ndual-FBG sensing schematic",
              transform=ax_a.transAxes, ha="center", va="center", fontsize=9, color=COLORS["gray"])
    ax_a.text(0.5, 0.30, "PLACEHOLDER", transform=ax_a.transAxes, ha="center", va="center",
              fontsize=8, fontweight="bold", color=COLORS["red"])

    t1 = d1["Time_s"].to_numpy() / 60.0
    ax_b.plot(t1, d1["Current_A"], color=COLORS["gray"], lw=0.7, label="Current")
    ax_b.set_xlabel("Time (min)"); ax_b.set_ylabel("Current (A)"); style_ax(ax_b)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(t1, d1["SOC"] * 100, color=COLORS["blue"], lw=1.1, label="SOC")
    ax_b2.set_ylabel("SOC (%)", color=COLORS["blue"])
    ax_b2.tick_params(axis="y", colors=COLORS["blue"], direction="out", width=0.8, length=3)
    ax_b2.spines["top"].set_visible(False); ax_b2.spines["right"].set_visible(True)
    ax_b.set_title(f"{profile} at 1C", fontsize=8, pad=4)

    ax_c.plot(t1, d1["Wavelength_1"], color=COLORS["blue2"], lw=0.9, label="W1")
    ax_c.plot(t1, d1["Wavelength_2"], color=COLORS["teal"], lw=0.9, label="W2")
    ax_c.set_xlabel("Time (min)"); ax_c.set_ylabel("Wavelength shift (nm)")
    ax_c.legend(loc="best", ncol=2); style_ax(ax_c)

    ax_d.plot(d1["SOC"] * 100, d1["Wavelength_2"], color=COLORS["blue"], lw=1.0, label="1C")
    ax_d.plot(d2["SOC"] * 100, d2["Wavelength_2"], color=COLORS["red"], lw=1.0, label="2C")
    ax_d.set_xlabel("SOC (%)"); ax_d.set_ylabel("W2 wavelength shift (nm)")
    ax_d.invert_xaxis(); ax_d.legend(loc="best"); style_ax(ax_d)

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        add_panel_label(ax, lab)
    fig.suptitle("Electrical–optical observations under dynamic discharge", fontsize=9.5, y=0.995)
    return save_figure(fig, out_dir, "fig1_dataset_signals")


def corr_condition(rho: float) -> float:
    rho = abs(float(rho))
    return (1 + rho) / max(1 - rho, 1e-12)


def fig2_representation(data, out_dir: Path) -> dict:
    all_df = pd.concat([df.assign(profile=p, rate=r) for (p, r), df in data.items()], ignore_index=True)
    transfer = pd.DataFrame({
        "representation": ["Raw W", "Decoupled T/F", "ETMF-T/F"],
        "mae_pct": [1.3847, 2.0328, 1.5692],
    })
    fig = plt.figure(figsize=(WIDTH_IN, mm(115)))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.42)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    hb1 = ax_a.hexbin(all_df["Wavelength_1"], all_df["Wavelength_2"], gridsize=65,
                      mincnt=1, cmap="Blues", linewidths=0, bins="log")
    ax_a.set_xlabel("W1 shift (nm)"); ax_a.set_ylabel("W2 shift (nm)"); style_ax(ax_a)
    cb1 = fig.colorbar(hb1, ax=ax_a, fraction=0.045, pad=0.025)
    cb1.set_label("log count", fontsize=7); cb1.ax.tick_params(labelsize=6, length=2)

    hb2 = ax_b.hexbin(all_df["temperature_℃"], all_df["force_N"], gridsize=65,
                      mincnt=1, cmap="Reds", linewidths=0, bins="log")
    ax_b.set_xlabel("Decoupled temperature (°C)"); ax_b.set_ylabel("Decoupled force (N)"); style_ax(ax_b)
    cb2 = fig.colorbar(hb2, ax=ax_b, fraction=0.045, pad=0.025)
    cb2.set_label("log count", fontsize=7); cb2.ax.tick_params(labelsize=6, length=2)

    x = np.arange(2); width = 0.34
    for ri, rate in enumerate(RATES):
        d = all_df[all_df["rate"] == rate]
        rho_w = d[["Wavelength_1", "Wavelength_2"]].corr().iloc[0, 1]
        rho_tf = d[["temperature_℃", "force_N"]].corr().iloc[0, 1]
        offset = (ri - 0.5) * width
        bars = ax_c.bar(x + offset, [abs(rho_w), abs(rho_tf)], width=width, label=rate,
                        color=[COLORS["blue"] if ri == 0 else COLORS["blue2"]] * 2,
                        edgecolor="white", linewidth=0.5)
        for bar, rho in zip(bars, [rho_w, rho_tf]):
            ax_c.text(bar.get_x() + bar.get_width()/2, min(0.99, bar.get_height()+0.025),
                      f"κ={corr_condition(rho):.1f}", ha="center", va="bottom", fontsize=6.3)
    ax_c.set_xticks(x, ["Raw W1/W2", "Decoupled T/F"])
    ax_c.set_ylabel("Absolute Pearson correlation"); ax_c.set_ylim(0, 1.08)
    ax_c.legend(loc="lower right", ncol=2); style_ax(ax_c)

    bars = ax_d.bar(transfer["representation"], transfer["mae_pct"],
                    color=[COLORS["blue"], COLORS["red"], COLORS["gray"]], width=0.66)
    ax_d.set_ylabel("Cross-rate development MAE (% SOC)")
    ax_d.set_ylim(0, max(transfer["mae_pct"]) * 1.22)
    for bar, v in zip(bars, transfer["mae_pct"]):
        ax_d.text(bar.get_x()+bar.get_width()/2, v+0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    style_ax(ax_d)

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        add_panel_label(ax, lab)
    fig.suptitle("Dual-FBG representation geometry and predictive transfer", fontsize=9.5, y=0.995)
    return save_figure(fig, out_dir, "fig2_representation_analysis")


def largest_block(df: pd.DataFrame, profile: str, rate: str) -> pd.DataFrame:
    sub = df[(df["profile"] == profile) & (df["rate"] == rate)].copy()
    if sub.empty:
        raise ValueError(f"No T1 predictions for {profile} {rate}")
    block = sub.groupby("source_name", sort=False).size().idxmax()
    out = sub[sub["source_name"] == block].copy().reset_index(drop=True)
    out["sample"] = np.arange(len(out))
    return out


def load_t1_95_predictions(source_dir: Path, t1_dir: Path | None,
                           generated_source_dir: Path) -> pd.DataFrame:
    committed = source_dir / "fig4_t1_predictions_uq.csv"
    if committed.exists():
        return pd.read_csv(committed)
    if t1_dir is None:
        raise FileNotFoundError("Need either committed fig4_t1_predictions_uq.csv or --t1-dir")
    intervals = pd.read_csv(t1_dir / "uq" / "test_intervals.csv")
    df = intervals[np.isclose(intervals["alpha"], 0.05)].copy()
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()
    generated_source_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(generated_source_dir / "fig4_t1_predictions_uq.csv", index=False)
    return df


def fig4_predictions(df: pd.DataFrame, out_dir: Path) -> dict:
    smooth = largest_block(df, "NEDC", "1C")
    dynamic = largest_block(df, "NYCC", "1C")
    fig = plt.figure(figsize=(WIDTH_IN, mm(112)))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.40)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, d, title in [(ax_a, smooth, "NEDC 1C"), (ax_b, dynamic, "NYCC 1C")]:
        ax.plot(d["sample"], d["y_true"]*100, color=COLORS["dark"], lw=1.3, label="Reference")
        ax.plot(d["sample"], d["y_pred"]*100, color=COLORS["blue"], lw=1.0, label="RA-FBG-TCN")
        ax.set_xlabel("Window index"); ax.set_ylabel("SOC (%)"); ax.set_title(title, fontsize=8, pad=4)
        style_ax(ax)
    ax_a.legend(loc="best", ncol=2)

    ax_c.plot(dynamic["sample"], dynamic["abs_error"]*100, color=COLORS["red"], lw=0.9)
    ax_c.axhline(dynamic["abs_error"].median()*100, color=COLORS["gray"], ls="--", lw=0.8,
                 label="Median")
    ax_c.set_xlabel("Window index"); ax_c.set_ylabel("Absolute error (% SOC)")
    ax_c.legend(loc="best"); style_ax(ax_c)

    err = df["abs_error"].to_numpy()*100
    ax_d.hist(err, bins=40, density=True, color=COLORS["blue2"], alpha=0.78,
              edgecolor="white", linewidth=0.3)
    ax_d.axvline(np.mean(err), color=COLORS["red"], lw=1.0, label=f"Mean = {np.mean(err):.2f}%")
    ax_d.axvline(np.quantile(err, 0.95), color=COLORS["gray"], lw=1.0, ls="--",
                 label=f"95th = {np.quantile(err, 0.95):.2f}%")
    ax_d.set_xlabel("Absolute error (% SOC)"); ax_d.set_ylabel("Density")
    ax_d.legend(loc="best"); style_ax(ax_d)

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        add_panel_label(ax, lab)
    fig.suptitle("Conventional mixed-condition SOC estimation", fontsize=9.5, y=0.995)
    return save_figure(fig, out_dir, "fig4_conventional_soc_prediction")


def fig5_ood(data, source_dir: Path, out_dir: Path) -> dict:
    bins = pd.read_csv(source_dir / "fig5_ood_bins.csv")
    held_out = "NEDC"
    train_current = pd.concat([data[(p, "1C")][["Current_A"]] for p in PROFILES if p != held_out],
                              ignore_index=True)["Current_A"]
    lo, hi = np.quantile(train_current.to_numpy(), [0.005, 0.995])
    test = data[(held_out, "2C")]
    t = test["Time_s"].to_numpy()/60.0; current = test["Current_A"].to_numpy()

    fig = plt.figure(figsize=(WIDTH_IN, mm(73)))
    gs = fig.add_gridspec(1, 3, wspace=0.38, width_ratios=[1.25, 1.0, 1.0])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax_a.axhspan(lo, hi, color=COLORS["light"], alpha=0.55, label="1C training support")
    ax_a.plot(t, current, color=COLORS["blue"], lw=0.75, label="NEDC 2C test")
    mask = (current < lo) | (current > hi)
    ax_a.fill_between(t, current, np.where(current < lo, lo, hi), where=mask,
                      color=COLORS["red"], alpha=0.25, interpolate=False, label="Outside support")
    ax_a.set_xlabel("Time (min)"); ax_a.set_ylabel("Current (A)")
    ax_a.legend(loc="best", fontsize=6.2); style_ax(ax_a)

    x = np.arange(len(bins)); w = 0.36
    ax_b.bar(x-w/2, bins["vi_mae"]*100, width=w, color=COLORS["gray"], label="VI")
    ax_b.bar(x+w/2, bins["viw_mae"]*100, width=w, color=COLORS["blue"], label="VI+W")
    ax_b.set_xticks(x, ["ID", "0–25", "25–50", "50–75", "75–100"], rotation=35,
                    ha="right", rotation_mode="anchor")
    ax_b.set_xlabel("Electrical OOD fraction (%)"); ax_b.set_ylabel("MAE (% SOC)")
    ax_b.legend(loc="upper left"); style_ax(ax_b)

    gains = bins["relative_optical_gain_pct"].to_numpy()
    ax_c.axhline(0, color=COLORS["gray"], lw=0.8)
    ax_c.plot(x, gains, color=COLORS["red"], marker="o", ms=4.5, lw=1.2)
    ax_c.fill_between(x, 0, gains, where=gains>=0, color=COLORS["blue"], alpha=0.10)
    ax_c.fill_between(x, 0, gains, where=gains<0, color=COLORS["red"], alpha=0.10)
    ax_c.set_xticks(x, ["ID", "0–25", "25–50", "50–75", "75–100"], rotation=35,
                    ha="right", rotation_mode="anchor")
    ax_c.set_xlabel("Electrical OOD fraction (%)"); ax_c.set_ylabel("Relative optical gain (%)")
    for xi, g in zip(x, gains):
        ax_c.annotate(f"{g:+.1f}", (xi, g), xytext=(0, 5 if g >= 0 else -8),
                      textcoords="offset points", ha="center", fontsize=6.2)
    ax_c.set_ylim(min(-24, gains.min()-4), max(54, gains.max()+4)); style_ax(ax_c)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        add_panel_label(ax, lab)
    fig.suptitle("Optical assistance emerges with electrical distribution shift", fontsize=9.5, y=1.01)
    return save_figure(fig, out_dir, "fig5_electrical_ood_optical_gain")


def fig6_t4(source_dir: Path, out_dir: Path) -> dict:
    prof = pd.read_csv(source_dir / "fig6_t4_profile_summary.csv")
    boot = pd.read_csv(source_dir / "fig6_t4_bootstrap.csv")
    fig = plt.figure(figsize=(WIDTH_IN, mm(76)))
    gs = fig.add_gridspec(1, 3, wspace=0.40, width_ratios=[1.05, 1.05, 0.95])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, direction, color, title in [
        (ax_a, "1C_to_2C", COLORS["red"], "1C → 2C"),
        (ax_b, "2C_to_1C", COLORS["blue"], "2C → 1C"),
    ]:
        d = prof[prof["direction"] == direction].set_index("held_out_profile").loc[list(PROFILES)].reset_index()
        x = np.arange(len(d))
        ax.bar(x, d["MAE_mean"]*100, yerr=d["MAE_std"]*100, color=color, alpha=0.84,
               error_kw={"elinewidth":0.8, "capthick":0.8, "capsize":2.5}, width=0.68)
        ax.set_xticks(x, d["held_out_profile"], rotation=38, ha="right", rotation_mode="anchor")
        ax.set_ylabel("MAE (% SOC)"); ax.set_title(title, fontsize=8, pad=4); style_ax(ax)

    order = ["1C_to_2C", "2C_to_1C", "overall"]
    labels = ["1C → 2C", "2C → 1C", "Overall"]
    d = boot.set_index("scope").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    vals = d["MAE_mean"].to_numpy()*100
    lo = d["seed_cluster_bootstrap95_lo"].to_numpy()*100
    hi = d["seed_cluster_bootstrap95_hi"].to_numpy()*100
    for yi, val, l, h, color in zip(y, vals, lo, hi,
                                    [COLORS["red"], COLORS["blue"], COLORS["dark"]]):
        ax_c.plot([l, h], [yi, yi], color=color, lw=1.4)
        ax_c.plot(val, yi, "o", color=color, ms=5)
        ax_c.text(h+0.06, yi, f"{val:.2f}", va="center", fontsize=6.5)
    ax_c.set_yticks(y, labels)
    ax_c.set_xlabel("MAE (% SOC), cluster bootstrap 95% CI")
    ax_c.set_ylim(-0.6, len(y)-0.4); style_ax(ax_c)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        add_panel_label(ax, lab)
    fig.suptitle("Cross-rate unseen-profile generalization across five seeds", fontsize=9.5, y=1.01)
    return save_figure(fig, out_dir, "fig6_cross_rate_unseen_profile")


def fig7_noise_uq(source_dir: Path, pred: pd.DataFrame, out_dir: Path) -> dict:
    noise = pd.read_csv(source_dir / "fig7_noise_summary.csv")
    uq = pd.read_csv(source_dir / "fig7_uq_summary.csv")
    block_name = "NYCC_1C.xlsx::test::9432:9994"
    d = pred[pred["source_name"] == block_name].copy().reset_index(drop=True)
    if d.empty:
        d = largest_block(pred, "NYCC", "1C")
    else:
        d["sample"] = np.arange(len(d))

    fig = plt.figure(figsize=(WIDTH_IN, mm(76)))
    gs = fig.add_gridspec(1, 3, wspace=0.40, width_ratios=[0.95, 0.95, 1.25])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for direction, color, label in [
        ("1C_to_2C", COLORS["red"], "1C → 2C"),
        ("2C_to_1C", COLORS["blue"], "2C → 1C"),
    ]:
        q = noise[noise["direction"] == direction].sort_values("sigma_pm_each_wavelength")
        ax_a.plot(q["sigma_pm_each_wavelength"], q["MAE_mean"]*100,
                  marker="o", ms=4.2, lw=1.2, color=color, label=label)
        ax_b.plot(q["sigma_pm_each_wavelength"], q["Q95_AE_mean"]*100,
                  marker="o", ms=4.2, lw=1.2, color=color, label=label)
    for ax, ylabel in [(ax_a, "MAE (% SOC)"), (ax_b, "Q95 absolute error (% SOC)")]:
        ax.set_xlabel("Wavelength noise σ (pm/channel)"); ax.set_xticks([0, 0.5, 1, 2])
        ax.set_ylabel(ylabel); ax.legend(loc="best"); style_ax(ax)

    ax_c.fill_between(d["sample"].to_numpy(), d["lower"].to_numpy()*100,
                      d["upper"].to_numpy()*100, color=COLORS["blue2"], alpha=0.18,
                      label="95% interval")
    ax_c.plot(d["sample"], d["y_true"]*100, color=COLORS["dark"], lw=1.1, label="Reference")
    ax_c.plot(d["sample"], d["y_pred"]*100, color=COLORS["blue"], lw=0.9, label="Prediction")
    u95 = uq.loc[np.isclose(uq["nominal_coverage"], 0.95)].iloc[0]
    ax_c.set_title(f"95% nominal: PICP {u95['PICP']*100:.2f}%, MPIW {u95['MPIW']*100:.3f}% SOC",
                   fontsize=7, pad=4)
    ax_c.set_xlabel("Window index"); ax_c.set_ylabel("SOC (%)")
    ax_c.legend(loc="best", ncol=1); style_ax(ax_c)

    for ax, lab in zip([ax_a, ax_b, ax_c], "abc"):
        add_panel_label(ax, lab)
    fig.suptitle("FBG noise robustness and calibrated uncertainty", fontsize=9.5, y=1.01)
    return save_figure(fig, out_dir, "fig7_noise_and_uncertainty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, default=Path("paper/source_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures/generated"))
    parser.add_argument("--t1-dir", type=Path, default=None)
    parser.add_argument("--generated-source-data", type=Path,
                        default=Path("paper/figures/generated/source_data"))
    parser.add_argument("--qa-json", type=Path,
                        default=Path("paper/figures/generated/figure_manifest.json"))
    args = parser.parse_args()

    data = load_dataset(args.data_root)
    t1_pred = load_t1_95_predictions(args.source_data, args.t1_dir, args.generated_source_data)
    products = {
        "fig1": fig1_dataset_signals(data, args.out_dir),
        "fig2": fig2_representation(data, args.out_dir),
        "fig4": fig4_predictions(t1_pred, args.out_dir),
        "fig5": fig5_ood(data, args.source_data, args.out_dir),
        "fig6": fig6_t4(args.source_data, args.out_dir),
        "fig7": fig7_noise_uq(args.source_data, t1_pred, args.out_dir),
    }
    manifest = {
        "figure_width_mm": WIDTH_MM,
        "raster_dpi": DPI,
        "font_policy": "Times New Roman / Liberation Serif fallback; editable SVG text",
        "fig3": "PLACEHOLDER: methodology framework intentionally deferred",
        "products": products,
        "source_data": sorted(p.name for p in args.source_data.glob("*.csv")) +
                       sorted(p.name for p in args.generated_source_data.glob("*.csv")),
    }
    args.qa_json.parent.mkdir(parents=True, exist_ok=True)
    args.qa_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
