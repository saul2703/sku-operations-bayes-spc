from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "DataCoSupplyChainDataset.csv"
OUT_DIR = ROOT_DIR / "data"
IMG_DIR = ROOT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Dual Dark / Light Operate chart palettes (matches DESIGN.md and index.html)
THEME_PALETTES = {
    "dark": SimpleNamespace(
        suffix="",
        bg_canvas="#0b1120",
        bg_surface="#111827",
        bg_raised="#172135",
        border="#29374d",
        text_main="#f4f7fb",
        text_muted="#9db1cc",
        blue="#38bdf8",
        green="#34d399",
        amber="#fbbf24",
        pink="#f472b6",
    ),
    "light": SimpleNamespace(
        suffix="_light",
        bg_canvas="#f8fafc",
        bg_surface="#ffffff",
        bg_raised="#f1f5f9",
        border="#cbd5e1",
        text_main="#0f172a",
        text_muted="#475569",
        blue="#0284c7",
        green="#047857",
        amber="#b45309",
        pink="#be185d",
    ),
}


@dataclass(frozen=True)
class SkuSchemaContract:
    """Boundary contract ensuring every SKU record has valid counts, rates, and intervals."""

    sku_id: int
    product_name: str
    category_name: str
    n_orders: int
    late_orders: int
    raw_late_pct: float
    eb_late_pct: float
    ci_low_pct: float
    ci_high_pct: float
    total_sales: float

    def validate(self) -> None:
        if self.sku_id <= 0 or self.n_orders <= 0:
            raise ValueError(f"Invalid sku_id or order count for {self.sku_id}")
        if not (0 <= self.late_orders <= self.n_orders):
            raise ValueError(f"late_orders={self.late_orders} exceeds n_orders={self.n_orders}")
        for name, pct in [
            ("raw_late_pct", self.raw_late_pct),
            ("eb_late_pct", self.eb_late_pct),
            ("ci_low_pct", self.ci_low_pct),
            ("ci_high_pct", self.ci_high_pct),
        ]:
            if not (0.0 <= pct <= 100.0):
                raise ValueError(f"{name}={pct} out of [0, 100] for SKU {self.sku_id}")
        if not (self.ci_low_pct <= self.eb_late_pct <= self.ci_high_pct):
            raise ValueError(f"Posterior {self.eb_late_pct} outside CI [{self.ci_low_pct}, {self.ci_high_pct}]")


DATA_HYGIENE_RULES = [
    {
        "field": "order date (DateOrders)",
        "raw_anomaly": "Mixed timestamp formats: 35,498 rows use MM/DD/YYYY HH:MM and 23,396 rows use MM-DD-YYYY HH:MM.",
        "clean_rule": "Parse with explicit format='mixed' (dayfirst=False) so zero orders drop or misalign across months.",
    },
    {
        "field": "Encoding & Tail Whitespace",
        "raw_anomaly": "CSV uses latin1 encoding (non-UTF8 accent bytes) and trailing spaces in department names ('Health and Beauty ').",
        "clean_rule": "Read with encoding='latin1' and strip whitespace across categorical dimensions before grouping.",
    },
    {
        "field": "Shipping Mode SLA Paradox",
        "raw_anomaly": "First Class promises 1-day shipping but averages 2.00 real days (98.9% late); Second Class promises 2 days but averages 3.99 days (78.5% late).",
        "clean_rule": "Compute lead_time_variance_days = real - scheduled and evaluate SKU variance both overall and inside Standard Class.",
    },
    {
        "field": "Delivery Status vs. Late Risk",
        "raw_anomaly": "1,291 canceled orders (672 SUSPECTED_FRAUD + 619 CANCELED) carry Late_delivery_risk = 0 despite never delivering.",
        "clean_rule": "Track Shipping Canceled as a separate SPC p-chart alongside Late Delivery Risk so cancellations never masquerade as on-time wins.",
    },
]


def setup_chart_theme(t: SimpleNamespace) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t.bg_canvas,
            "axes.facecolor": t.bg_surface,
            "axes.edgecolor": t.border,
            "axes.labelcolor": t.text_main,
            "axes.titlecolor": t.text_main,
            "xtick.color": t.text_muted,
            "ytick.color": t.text_muted,
            "text.color": t.text_main,
            "grid.color": t.border,
            "grid.alpha": 0.45,
            "font.family": "sans-serif",
            "font.size": 9.5,
        }
    )


def style_ax(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6)


def fit_beta_binomial_prior(n_arr: np.ndarray, k_arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """Estimate Beta(alpha0, beta0) prior via volume-weighted Method of Moments."""
    n = n_arr.astype(np.float64)
    k = k_arr.astype(np.float64)
    total_n = float(n.sum())
    p_hat = float(k.sum() / total_n)
    rates = k / n

    observed_var = float(np.sum(n * (rates - p_hat) ** 2) / total_n)
    expected_noise_var = float(len(n) * p_hat * (1.0 - p_hat) / total_n)
    between_var = max(1e-6, observed_var - expected_noise_var)

    # Prior effective sample size M = alpha0 + beta0
    prior_strength = max(2.0, (p_hat * (1.0 - p_hat) / between_var) - 1.0)
    alpha0 = p_hat * prior_strength
    beta0 = (1.0 - p_hat) * prior_strength
    return alpha0, beta0, p_hat, observed_var, expected_noise_var


def beta_posterior_ci(k: np.ndarray, n: np.ndarray, alpha0: float, beta0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior mean and 95% logit-normal Beta credible interval (zero-dependency)."""
    a_post = k + alpha0
    b_post = (n - k) + beta0
    mean = a_post / (a_post + b_post)

    # Logit transform guarantees bounds stay inside (0, 1) even for small counts
    mu_logit = np.log(a_post / b_post)
    se_logit = np.sqrt((1.0 / a_post) + (1.0 / b_post))
    lo = 1.0 / (1.0 + np.exp(-(mu_logit - 1.96 * se_logit)))
    hi = 1.0 / (1.0 + np.exp(-(mu_logit + 1.96 * se_logit)))
    return mean, lo, hi


def apply_empirical_bayes(df_grp: pd.DataFrame, n_col: str = "n_orders", k_col: str = "late_orders") -> tuple[pd.DataFrame, dict]:
    """Attach Empirical Bayes posterior rates, 95% credible intervals, and shrinkage deltas."""
    n_arr = df_grp[n_col].to_numpy(dtype=np.float64)
    k_arr = df_grp[k_col].to_numpy(dtype=np.float64)
    alpha0, beta0, p_hat, obs_var, noise_var = fit_beta_binomial_prior(n_arr, k_arr)

    post_mean, ci_lo, ci_hi = beta_posterior_ci(k_arr, n_arr, alpha0, beta0)
    out = df_grp.copy()
    out["raw_late_pct"] = (k_arr / n_arr) * 100.0
    out["eb_late_pct"] = post_mean * 100.0
    out["ci_low_pct"] = ci_lo * 100.0
    out["ci_high_pct"] = ci_hi * 100.0
    out["shrinkage_pp"] = out["eb_late_pct"] - out["raw_late_pct"]
    out["shrinkage_weight"] = n_arr / (n_arr + alpha0 + beta0)

    prior_meta = {
        "alpha0": round(alpha0, 2),
        "beta0": round(beta0, 2),
        "prior_sample_size": round(alpha0 + beta0, 2),
        "baseline_rate_pct": round(p_hat * 100.0, 2),
        "observed_variance": round(obs_var, 6),
        "binomial_noise_variance": round(noise_var, 6),
        "true_between_variance": round(max(0.0, obs_var - noise_var), 6),
    }
    return out, prior_meta


def nelson_run_flags(values: np.ndarray, center: float, sigma: np.ndarray) -> list[str]:
    """Flag points violating Shewhart 3-sigma, 2-of-3 beyond 2-sigma, or 8-point shift rules."""
    z = (values - center) / np.maximum(sigma, 1e-9)
    flags = []
    run_len = 0
    prev_sign = 0

    for i in range(len(values)):
        reasons = []
        if abs(z[i]) > 3.0:
            reasons.append("Rule 1 (>3s)")

        # Rule 2: 2 of 3 consecutive points beyond 2s on the same side
        if i >= 2:
            win = z[i - 2 : i + 1]
            if (np.sum(win > 2.0) >= 2 and z[i] > 2.0) or (np.sum(win < -2.0) >= 2 and z[i] < -2.0):
                reasons.append("Rule 2 (2/3 >2s)")

        # Rule 4: 8 consecutive points on one side of the center line
        sign = 1 if z[i] >= 0 else -1
        run_len = run_len + 1 if sign == prev_sign else 1
        prev_sign = sign
        if run_len >= 8:
            reasons.append("Rule 4 (8-pt shift)")

        flags.append(" | ".join(reasons) if reasons else "In Control")
    return flags


def load_and_clean_orders() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="latin1")
    for col in ["Category Name", "Department Name", "Market", "Order Region", "Shipping Mode", "Delivery Status", "Order Status"]:
        df[col] = df[col].astype(str).str.strip()

    order_dt = pd.to_datetime(df["order date (DateOrders)"], format="mixed", dayfirst=False)
    df["order_month"] = order_dt.dt.to_period("M").astype(str)
    df["delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    df["is_canceled"] = (df["Delivery Status"] == "Shipping canceled").astype(int)
    df["late_sales"] = np.where(df["Late_delivery_risk"] == 1, df["Sales"], 0.0)
    return df


def build_monthly_spc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = (
        df.groupby("order_month", as_index=False)
        .agg(
            n_orders=("Order Item Id", "count"),
            late_orders=("Late_delivery_risk", "sum"),
            canceled_orders=("is_canceled", "sum"),
            mean_delay_days=("delay_days", "mean"),
            std_delay_days=("delay_days", "std"),
            monthly_sales=("Sales", "sum"),
            standard_share=("Shipping Mode", lambda s: float((s == "Standard Class").mean() * 100.0)),
        )
        .sort_values("order_month")
        .reset_index(drop=True)
    )

    n = m["n_orders"].to_numpy(dtype=np.float64)
    p_late = m["late_orders"].to_numpy(dtype=np.float64) / n
    p_canc = m["canceled_orders"].to_numpy(dtype=np.float64) / n
    x_bar_m = m["mean_delay_days"].to_numpy(dtype=np.float64)

    p_bar = float(m["late_orders"].sum() / n.sum())
    sig_p = np.sqrt(p_bar * (1.0 - p_bar) / n)
    m["late_pct"] = p_late * 100.0
    m["ucl_late_pct"] = np.minimum(100.0, (p_bar + 3.0 * sig_p) * 100.0)
    m["lcl_late_pct"] = np.maximum(0.0, (p_bar - 3.0 * sig_p) * 100.0)
    m["late_spc_status"] = nelson_run_flags(p_late, p_bar, sig_p)

    pc_bar = float(m["canceled_orders"].sum() / n.sum())
    sig_c = np.sqrt(pc_bar * (1.0 - pc_bar) / n)
    m["canc_pct"] = p_canc * 100.0
    m["ucl_canc_pct"] = np.minimum(100.0, (pc_bar + 3.0 * sig_c) * 100.0)
    m["lcl_canc_pct"] = np.maximum(0.0, (pc_bar - 3.0 * sig_c) * 100.0)
    m["canc_spc_status"] = nelson_run_flags(p_canc, pc_bar, sig_c)

    x_bar = float(df["delay_days"].mean())
    s_bar = float(m["std_delay_days"].mean())
    sig_x = s_bar / np.sqrt(n)
    m["ucl_delay_days"] = x_bar + 3.0 * sig_x
    m["lcl_delay_days"] = x_bar - 3.0 * sig_x
    m["delay_spc_status"] = nelson_run_flags(x_bar_m, x_bar, sig_x)

    # S-chart control bounds using standard c4 approximation (~1 for large subgroups)
    m["ucl_std_days"] = s_bar * (1.0 + 3.0 / np.sqrt(2.0 * np.maximum(1.0, n - 1.0)))
    m["lcl_std_days"] = np.maximum(0.0, s_bar * (1.0 - 3.0 / np.sqrt(2.0 * np.maximum(1.0, n - 1.0))))

    spc_meta = {
        "months_count": int(len(m)),
        "p_bar_late_pct": round(p_bar * 100.0, 2),
        "p_bar_canc_pct": round(pc_bar * 100.0, 2),
        "x_bar_delay_days": round(x_bar, 3),
        "s_bar_delay_days": round(s_bar, 3),
        "late_out_of_control_months": m.loc[m["late_spc_status"] != "In Control", "order_month"].tolist(),
        "canc_out_of_control_months": m.loc[m["canc_spc_status"] != "In Control", "order_month"].tolist(),
        "delay_out_of_control_months": m.loc[m["delay_spc_status"] != "In Control", "order_month"].tolist(),
    }
    return m, spc_meta


def generate_charts(sku_df: pd.DataFrame, lane_df: pd.DataFrame, mode_df: pd.DataFrame, spc_df: pd.DataFrame, spc_meta: dict, sku_prior: dict, std_sku_prior: dict, theme_name: str = "dark") -> None:
    t = THEME_PALETTES[theme_name]
    setup_chart_theme(t)

    # ------------------------------------------------------------------
    # 1. Hero Overview Preview (4-Panel Executive Summary)
    # ------------------------------------------------------------------
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 9.2))

    # Panel 1: Monthly Late Rate p-chart snapshot
    x_idx = np.arange(len(spc_df))
    a1.plot(x_idx, spc_df["late_pct"], color=t.blue, linewidth=2.0, marker="o", markersize=3.5, label="Monthly Late %")
    a1.plot(x_idx, spc_df["ucl_late_pct"], color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Limits")
    a1.plot(x_idx, spc_df["lcl_late_pct"], color=t.pink, linestyle="--", linewidth=1.2)
    a1.axhline(spc_meta["p_bar_late_pct"], color=t.text_muted, linestyle=":", linewidth=1.2)
    out_mask = spc_df["late_spc_status"] != "In Control"
    a1.scatter(x_idx[out_mask], spc_df.loc[out_mask, "late_pct"], color=t.amber, s=55, zorder=5, label="Out-of-Control")
    a1.set_xticks(x_idx[::6])
    a1.set_xticklabels(spc_df["order_month"].iloc[::6], rotation=20)
    a1.yaxis.set_major_formatter(mtick.PercentFormatter())
    a1.set_title("1. Monthly Late Delivery p-Chart (37 Months)", fontweight="bold")
    a1.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8)
    style_ax(a1)

    # Panel 2: Shipping Mode SLA Breakdown
    m_ord = mode_df.sort_values("late_pct")
    colors_m = [t.green if v < 45 else (t.amber if v < 70 else t.pink) for v in m_ord["late_pct"]]
    bars = a2.barh(m_ord["Shipping Mode"], m_ord["late_pct"], color=colors_m, height=0.55)
    for bar, (_, r) in zip(bars, m_ord.iterrows()):
        a2.text(bar.get_width() + 1.8, bar.get_y() + bar.get_height() / 2, f"{r['late_pct']:.1f}% ({r['mean_delay']:+.2f}d)", va="center", fontsize=8.5, color=t.text_main)
    a2.set_xlim(0, 128)
    a2.xaxis.set_major_formatter(mtick.PercentFormatter())
    a2.set_title("2. Late Rate & Avg Delay by Shipping Mode", fontweight="bold")
    style_ax(a2, "x")

    # Panel 3: Raw vs Empirical Bayes Posterior (Sample Size Funnel)
    a3.scatter(sku_df["n_orders"], sku_df["raw_late_pct"], color=t.text_muted, alpha=0.55, s=26, label="Raw Rate (Noisy)")
    a3.scatter(sku_df["n_orders"], sku_df["eb_late_pct"], color=t.blue, alpha=0.85, s=34, label="EB Posterior")
    a3.axhline(sku_prior["baseline_rate_pct"], color=t.green, linestyle="--", linewidth=1.2, label=f"Prior Mean ({sku_prior['baseline_rate_pct']}%)")
    a3.set_xscale("log")
    a3.yaxis.set_major_formatter(mtick.PercentFormatter())
    a3.set_xlabel("Orders per SKU (Log Scale)")
    a3.set_title("3. Empirical Bayes Shrinkage Across 116 SKUs", fontweight="bold")
    a3.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8)
    style_ax(a3)

    # Panel 4: Top 6 SKUs by Late Sales Exposure
    top_exp = sku_df.sort_values("late_sales", ascending=False).head(6).iloc[::-1]
    short_names = [n[:32] + ("..." if len(n) > 32 else "") for n in top_exp["Product Name"]]
    a4.barh(short_names, top_exp["late_sales"] / 1e3, color=t.amber, height=0.55)
    for idx_b, (_, r) in enumerate(top_exp.iterrows()):
        a4.text(r["late_sales"] / 1e3 + 15, idx_b, f"${r['late_sales']/1e3:,.0f}K ({r['eb_late_pct']:.1f}%)", va="center", fontsize=8.2)
    a4.set_xlim(0, top_exp["late_sales"].max() / 1e3 * 1.28)
    a4.set_xlabel("Late-Delivery Sales Exposure ($K)")
    a4.set_title("4. Top SKUs by Revenue Exposed to Late Delivery", fontweight="bold")
    style_ax(a4, "x")

    fig.suptitle("Global Store & SKU Operations — Empirical Bayes Shrinkage & SPC Overview", fontsize=13.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(IMG_DIR / f"ops_overview_preview{t.suffix}.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Empirical Bayes Shrinkage: Funnel Plot + Dumbbell Shift
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4))
    p0 = sku_prior["baseline_rate_pct"] / 100.0
    n_grid = np.geomspace(3, 10000, 300)
    se_grid = np.sqrt(p0 * (1.0 - p0) / n_grid) * 100.0

    ax1.fill_between(n_grid, p0 * 100 - 2 * se_grid, p0 * 100 + 2 * se_grid, color=t.blue, alpha=0.12, label="±2σ Binomial Funnel")
    ax1.plot(n_grid, p0 * 100 + 3 * se_grid, color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Binomial Limits")
    ax1.plot(n_grid, p0 * 100 - 3 * se_grid, color=t.pink, linestyle="--", linewidth=1.2)
    ax1.axhline(p0 * 100, color=t.green, linestyle="-", linewidth=1.2, label=f"Global Mean ({p0*100:.2f}%)")
    ax1.scatter(sku_df["n_orders"], sku_df["raw_late_pct"], color=t.amber, alpha=0.75, s=36, label="Raw SKU Rate")
    high_vol = sku_df[sku_df["n_orders"] >= 2000]
    ax1.scatter(high_vol["n_orders"], high_vol["eb_late_pct"], color=t.blue, s=55, edgecolor=t.text_main, linewidth=0.8, label="High-Volume Posterior")
    ax1.set_xscale("log")
    ax1.set_ylim(25, 92)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xlabel("Order Count per SKU (Log Scale)")
    ax1.set_ylabel("Late Delivery Rate (%)")
    ax1.set_title("Binomial Funnel: Small-Sample Noise vs. High-Volume Signal", fontweight="bold")
    ax1.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8, loc="upper right")
    style_ax(ax1)

    movers = pd.concat([
        sku_df.sort_values("shrinkage_pp").head(5),
        sku_df.sort_values("shrinkage_pp", ascending=False).head(4),
        sku_df.sort_values("eb_late_pct", ascending=False).head(3),
    ]).drop_duplicates(subset=["sku_id"]).sort_values("eb_late_pct")

    y_pos = np.arange(len(movers))
    for i, (_, r) in enumerate(movers.iterrows()):
        ax2.plot([r["raw_late_pct"], r["eb_late_pct"]], [i, i], color=t.border, linewidth=2.2, zorder=2)
    ax2.scatter(movers["raw_late_pct"], y_pos, color=t.amber, s=50, zorder=3, label="Raw Rate")
    ax2.scatter(movers["eb_late_pct"], y_pos, color=t.blue, s=60, zorder=4, label="EB Posterior")
    ax2.axvline(p0 * 100, color=t.green, linestyle=":", linewidth=1.2)
    labels = [f"{r['Product Name'][:26]} (n={r['n_orders']})" for _, r in movers.iterrows()]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8.3)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_xlabel("Late Delivery Rate (%)")
    ax2.set_title("Shrinkage Corrections: Taming Low-Volume Extremes", fontweight="bold")
    ax2.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8)
    style_ax(ax2, "x")

    fig.suptitle(
        f"Empirical Bayes Beta-Binomial Shrinkage (Prior α₀={sku_prior['alpha0']}, β₀={sku_prior['beta0']}, M={sku_prior['prior_sample_size']:.0f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMG_DIR / f"empirical_bayes_shrinkage{t.suffix}.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. SPC p-Charts: Late Delivery Rate & Cancellation Rate
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.4), sharex=True)

    ax1.plot(x_idx, spc_df["late_pct"], color=t.blue, linewidth=2.1, marker="o", markersize=4, label="Monthly Late Delivery %")
    ax1.plot(x_idx, spc_df["ucl_late_pct"], color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Control Limits")
    ax1.plot(x_idx, spc_df["lcl_late_pct"], color=t.pink, linestyle="--", linewidth=1.2)
    ax1.axhline(spc_meta["p_bar_late_pct"], color=t.green, linestyle=":", linewidth=1.3, label=f"Center Line p̄ = {spc_meta['p_bar_late_pct']}%")
    r1_l = spc_df["late_spc_status"].str.contains("Rule 1")
    r2_l = (spc_df["late_spc_status"] != "In Control") & (~r1_l)
    ax1.scatter(x_idx[r2_l], spc_df.loc[r2_l, "late_pct"], color=t.amber, s=55, zorder=4, label="Nelson Rule 2 (2/3 >2σ)")
    ax1.scatter(x_idx[r1_l], spc_df.loc[r1_l, "late_pct"], color=t.pink, s=85, zorder=5, edgecolor=t.text_main, linewidth=0.9, label="Rule 1 Breach (>3σ)")
    for idx_o in np.where(r1_l)[0]:
        row_o = spc_df.iloc[idx_o]
        ax1.annotate(f"{row_o['order_month']} ({row_o['late_pct']:.1f}%)", (idx_o, row_o["late_pct"]), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8.2, color=t.pink, fontweight="bold")
    ax1.set_ylim(51.2, 64.5)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylabel("Late Delivery Rate")
    ax1.set_title("Shewhart p-Chart: Monthly Late Delivery Rate (Jan 2015 – Jan 2018)", fontweight="bold")
    ax1.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8, loc="upper left", ncol=2)
    style_ax(ax1)

    ax2.plot(x_idx, spc_df["canc_pct"], color=t.amber, linewidth=2.1, marker="o", markersize=4, label="Monthly Shipping Canceled %")
    ax2.plot(x_idx, spc_df["ucl_canc_pct"], color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Control Limits")
    ax2.plot(x_idx, spc_df["lcl_canc_pct"], color=t.pink, linestyle="--", linewidth=1.2)
    ax2.axhline(spc_meta["p_bar_canc_pct"], color=t.green, linestyle=":", linewidth=1.3, label=f"Center Line p̄ = {spc_meta['p_bar_canc_pct']}%")
    r1_c = spc_df["canc_spc_status"].str.contains("Rule 1")
    r2_c = (spc_df["canc_spc_status"] != "In Control") & (~r1_c)
    ax2.scatter(x_idx[r2_c], spc_df.loc[r2_c, "canc_pct"], color=t.blue, s=50, zorder=4, label="Nelson Rule 2 (2/3 >2σ)")
    ax2.scatter(x_idx[r1_c], spc_df.loc[r1_c, "canc_pct"], color=t.pink, s=85, zorder=5, edgecolor=t.text_main, linewidth=0.9, label="Rule 1 Breach (>3σ)")
    for idx_o in np.where(r1_c)[0]:
        row_o = spc_df.iloc[idx_o]
        dy = 10 if row_o["canc_pct"] > spc_meta["p_bar_canc_pct"] else -18
        ax2.annotate(f"{row_o['order_month']} ({row_o['canc_pct']:.2f}%)", (idx_o, row_o["canc_pct"]), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8, color=t.pink, fontweight="bold")
    ax2.set_ylim(0.2, 4.15)
    ax2.set_xticks(x_idx[::3])
    ax2.set_xticklabels(spc_df["order_month"].iloc[::3], rotation=25)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylabel("Cancellation Rate")
    ax2.set_title("Shewhart p-Chart: Monthly Canceled / Fraud-Held Shipments (4 Rule-1 Breaches)", fontweight="bold")
    ax2.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8, loc="upper left", ncol=2)
    style_ax(ax2)

    fig.tight_layout()
    fig.savefig(IMG_DIR / f"spc_p_chart_late_delivery{t.suffix}.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. SPC X-bar & S Chart on Lead-Time Variance (real - scheduled)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.2), sharex=True)

    ax1.plot(x_idx, spc_df["mean_delay_days"], color=t.blue, linewidth=2.1, marker="o", markersize=4, label="Monthly Mean Delay (Days)")
    ax1.plot(x_idx, spc_df["ucl_delay_days"], color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Control Limits")
    ax1.plot(x_idx, spc_df["lcl_delay_days"], color=t.pink, linestyle="--", linewidth=1.2)
    ax1.axhline(spc_meta["x_bar_delay_days"], color=t.green, linestyle=":", linewidth=1.3, label=f"Center Line X̄ = +{spc_meta['x_bar_delay_days']:.3f}d")
    out_x = spc_df["delay_spc_status"] != "In Control"
    ax1.scatter(x_idx[out_x], spc_df.loc[out_x, "mean_delay_days"], color=t.amber, s=80, zorder=5, edgecolor=t.text_main, linewidth=0.8, label="Out-of-Control (>3σ)")
    for idx_o in np.where(out_x)[0]:
        row_o = spc_df.iloc[idx_o]
        dy = 9 if row_o["mean_delay_days"] > spc_meta["x_bar_delay_days"] else -18
        ax1.annotate(f"{row_o['order_month']} ({row_o['mean_delay_days']:+.2f}d)", (idx_o, row_o["mean_delay_days"]), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8.2, color=t.amber, fontweight="bold")
    ax1.set_ylim(0.41, 0.82)
    ax1.set_ylabel("Mean Lead-Time Variance (Days)")
    ax1.set_title("X̄ Control Chart: Monthly Shipping Lead-Time Variance (Real − Scheduled Days)", fontweight="bold")
    ax1.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8, loc="upper left", ncol=2)
    style_ax(ax1)

    ax2.plot(x_idx, spc_df["std_delay_days"], color=t.green, linewidth=2.0, marker="s", markersize=3.8, label="Monthly Std Dev (S)")
    ax2.plot(x_idx, spc_df["ucl_std_days"], color=t.pink, linestyle="--", linewidth=1.2, label="±3σ Limits")
    ax2.plot(x_idx, spc_df["lcl_std_days"], color=t.pink, linestyle="--", linewidth=1.2)
    ax2.axhline(spc_meta["s_bar_delay_days"], color=t.blue, linestyle=":", linewidth=1.3, label=f"Mean Spread S̄ = {spc_meta['s_bar_delay_days']:.3f}d")
    ax2.set_xticks(x_idx[::3])
    ax2.set_xticklabels(spc_df["order_month"].iloc[::3], rotation=25)
    ax2.set_ylabel("Within-Month Std Dev (Days)")
    ax2.set_title("S Control Chart: Within-Month Lead-Time Volatility Stability", fontweight="bold")
    ax2.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8, loc="upper left", ncol=3)
    style_ax(ax2)

    fig.tight_layout()
    fig.savefig(IMG_DIR / f"spc_xbar_lead_time_drift{t.suffix}.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Lane Risk Heatmap (Top Regions x Shipping Modes)
    # ------------------------------------------------------------------
    top_regions = (
        lane_df.groupby("Order Region")["n_orders"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    modes_order = ["Standard Class", "Same Day", "Second Class", "First Class"]
    piv = (
        lane_df[lane_df["Order Region"].isin(top_regions)]
        .pivot(index="Order Region", columns="Shipping Mode", values="eb_late_pct")
        .reindex(index=top_regions, columns=modes_order)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    m_sorted = mode_df.set_index("Shipping Mode").reindex(modes_order).reset_index()
    y_m = np.arange(len(m_sorted))
    bar_colors = [t.green, t.amber, t.pink, t.pink]
    bars_l = ax1.barh(y_m, m_sorted["late_pct"], height=0.52, color=bar_colors)
    for bar, (_, r) in zip(bars_l, m_sorted.iterrows()):
        ax1.text(
            bar.get_width() + 1.8,
            bar.get_y() + bar.get_height() / 2,
            f"{r['late_pct']:.1f}%  |  {r['mean_delay']:+.2f}d vs SLA",
            va="center",
            fontsize=8.5,
            color=t.text_main,
            fontweight="bold",
        )
    ax1.set_yticks(y_m)
    ax1.set_yticklabels(m_sorted["Shipping Mode"])
    ax1.set_xlabel("Late Delivery Rate (%)")
    ax1.set_xlim(0, 138)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title("SLA Promise Gap by Shipping Tier", fontweight="bold")
    style_ax(ax1, "x")

    im = ax2.imshow(piv.to_numpy(), cmap="magma", aspect="auto", vmin=30, vmax=100)
    ax2.set_xticks(np.arange(len(modes_order)))
    ax2.set_xticklabels(modes_order, rotation=15)
    ax2.set_yticks(np.arange(len(top_regions)))
    ax2.set_yticklabels(top_regions)
    for i in range(len(top_regions)):
        for j in range(len(modes_order)):
            val = piv.iloc[i, j]
            txt_color = "#0b1120" if val > 75 else "#f4f7fb"
            ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8.3, color=txt_color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.03)
    cbar.set_label("Empirical Bayes Posterior Late %")
    ax2.set_title("Top 12 Regions × Shipping Mode Posterior Late Risk", fontweight="bold")

    fig.tight_layout()
    fig.savefig(IMG_DIR / f"lane_risk_heatmap_shipping_mode{t.suffix}.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Financial Exposure Pareto + Variance Decomposition Proof
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), gridspec_kw={"width_ratios": [1.45, 0.9]})

    top10 = sku_df.sort_values("late_sales", ascending=False).head(10).copy()
    cum_pct = top10["late_sales"].cumsum() / sku_df["late_sales"].sum() * 100.0
    x_p = np.arange(len(top10))
    ax1.bar(x_p, top10["late_sales"] / 1e3, color=t.blue, width=0.58, label="Late Sales Exposure ($K)")
    ax1_t = ax1.twinx()
    ax1_t.plot(x_p, cum_pct, color=t.amber, marker="o", linewidth=2.2, label="Cumulative Share (%)")
    ax1_t.set_ylim(0, 100)
    ax1_t.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xticks(x_p)
    ax1.set_xticklabels([n[:18] + "..." if len(n) > 18 else n for n in top10["Product Name"]], rotation=32, ha="right", fontsize=8)
    ax1.set_ylabel("Late-Delivery Sales Exposure ($K)")
    ax1_t.set_ylabel("Cumulative Share of All Late Sales")
    ax1.set_title("Top 10 SKUs Drive Over 85% of Late-Delivery Revenue Exposure", fontweight="bold")
    style_ax(ax1)
    ax1_t.spines["top"].set_visible(False)

    groups = ["All Modes\n(116 SKUs)", "Standard Class Only\n(116 SKUs)", "Region × Mode\n(92 Lanes)"]
    obs_v = [sku_prior["observed_variance"] * 1e4, std_sku_prior["observed_variance"] * 1e4, 622.62]
    noi_v = [sku_prior["binomial_noise_variance"] * 1e4, std_sku_prior["binomial_noise_variance"] * 1e4, 3.81]
    x_v = np.arange(len(groups))
    ax2.bar(x_v - 0.17, obs_v, width=0.32, color=t.pink, label="Observed Variance (×10⁻⁴)")
    ax2.bar(x_v + 0.17, noi_v, width=0.32, color=t.green, label="Expected Binomial Noise (×10⁻⁴)")
    ax2.set_yscale("log")
    ax2.set_xticks(x_v)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("Weighted Rate Variance (×10⁻⁴, Log Scale)")
    ax2.set_title("Signal vs. Noise: Why Shipping Mode Dominates", fontweight="bold")
    ax2.legend(facecolor=t.bg_raised, edgecolor=t.border, fontsize=8)
    style_ax(ax2)

    fig.tight_layout()
    fig.savefig(IMG_DIR / f"sku_financial_exposure_pareto{t.suffix}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    df = load_and_clean_orders()

    # 1. SKU-level Empirical Bayes Shrinkage
    sku_raw = (
        df.groupby(["Order Item Cardprod Id", "Product Name", "Category Name", "Department Name"], as_index=False)
        .agg(
            n_orders=("Order Item Id", "count"),
            late_orders=("Late_delivery_risk", "sum"),
            canceled_orders=("is_canceled", "sum"),
            mean_delay_days=("delay_days", "mean"),
            total_sales=("Sales", "sum"),
            late_sales=("late_sales", "sum"),
            total_profit=("Benefit per order", "sum"),
        )
        .rename(columns={"Order Item Cardprod Id": "sku_id"})
    )
    sku_df, sku_prior = apply_empirical_bayes(sku_raw)

    # 2. Within-Standard-Class SKU check (proves SKU variance collapses once shipping mode is held constant)
    std_only = df[df["Shipping Mode"] == "Standard Class"]
    std_sku_raw = (
        std_only.groupby(["Order Item Cardprod Id", "Product Name"], as_index=False)
        .agg(n_orders=("Order Item Id", "count"), late_orders=("Late_delivery_risk", "sum"))
        .rename(columns={"Order Item Cardprod Id": "sku_id"})
    )
    _, std_sku_prior = apply_empirical_bayes(std_sku_raw)

    # 3. Lane-level Empirical Bayes Shrinkage (Order Region x Shipping Mode = 92 lanes)
    lane_raw = df.groupby(["Market", "Order Region", "Shipping Mode"], as_index=False).agg(
        n_orders=("Order Item Id", "count"),
        late_orders=("Late_delivery_risk", "sum"),
        canceled_orders=("is_canceled", "sum"),
        mean_delay_days=("delay_days", "mean"),
        total_sales=("Sales", "sum"),
        late_sales=("late_sales", "sum"),
        total_profit=("Benefit per order", "sum"),
    )
    lane_df, lane_prior = apply_empirical_bayes(lane_raw)

    # 4. Shipping Mode summary
    mode_df = df.groupby("Shipping Mode", as_index=False).agg(
        n_orders=("Order Item Id", "count"),
        late_orders=("Late_delivery_risk", "sum"),
        mean_delay=("delay_days", "mean"),
        avg_scheduled=("Days for shipment (scheduled)", "mean"),
        avg_real=("Days for shipping (real)", "mean"),
        total_sales=("Sales", "sum"),
    )
    mode_df["late_pct"] = (mode_df["late_orders"] / mode_df["n_orders"]) * 100.0

    # 5. Monthly SPC (37 months)
    spc_df, spc_meta = build_monthly_spc(df)

    # 6. Validate every SKU against the dataclass contract
    for _, r in sku_df.iterrows():
        SkuSchemaContract(
            sku_id=int(r["sku_id"]),
            product_name=str(r["Product Name"]),
            category_name=str(r["Category Name"]),
            n_orders=int(r["n_orders"]),
            late_orders=int(r["late_orders"]),
            raw_late_pct=round(float(r["raw_late_pct"]), 4),
            eb_late_pct=round(float(r["eb_late_pct"]), 4),
            ci_low_pct=round(float(r["ci_low_pct"]), 4),
            ci_high_pct=round(float(r["ci_high_pct"]), 4),
            total_sales=round(float(r["total_sales"]), 2),
        ).validate()

    # 7. Generate all 6 publication charts in both dark and light modes
    for theme_name in ("dark", "light"):
        generate_charts(sku_df, lane_df, mode_df, spc_df, spc_meta, sku_prior, std_sku_prior, theme_name=theme_name)

    # 8. Build Top 5 / Bottom 5 comparison matrix (Raw Extreme Noise vs. Credible EB Risk)
    def fmt_sku_rows(sub: pd.DataFrame) -> list[dict]:
        return [
            {
                "sku_id": int(r["sku_id"]),
                "product_name": str(r["Product Name"]),
                "category_name": str(r["Category Name"]),
                "n_orders": int(r["n_orders"]),
                "raw_late_pct": round(float(r["raw_late_pct"]), 2),
                "eb_late_pct": round(float(r["eb_late_pct"]), 2),
                "shrinkage_pp": round(float(r["shrinkage_pp"]), 2),
                "late_sales": round(float(r["late_sales"]), 2),
            }
            for _, r in sub.iterrows()
        ]

    payload = {
        "dataset": {
            "source_file": CSV_PATH.name,
            "total_order_items": int(len(df)),
            "unique_orders": int(df["Order Id"].nunique()),
            "unique_skus": int(len(sku_df)),
            "unique_categories": int(df["Category Name"].nunique()),
            "unique_departments": int(df["Department Name"].nunique()),
            "markets_count": int(df["Market"].nunique()),
            "regions_count": int(df["Order Region"].nunique()),
            "lanes_count": int(len(lane_df)),
            "months_count": int(len(spc_df)),
            "date_start": str(spc_df["order_month"].iloc[0]),
            "date_end": str(spc_df["order_month"].iloc[-1]),
            "total_sales": round(float(df["Sales"].sum()), 2),
            "late_sales_exposure": round(float(df["late_sales"].sum()), 2),
            "late_sales_share_pct": round(float(df["late_sales"].sum() / df["Sales"].sum() * 100.0), 2),
            "total_profit": round(float(df["Benefit per order"].sum()), 2),
            "overall_late_pct": round(float(df["Late_delivery_risk"].mean() * 100.0), 2),
            "overall_canceled_pct": round(float(df["is_canceled"].mean() * 100.0), 2),
            "overall_mean_delay_days": round(float(df["delay_days"].mean()), 3),
        },
        "empirical_bayes": {
            "sku_prior": sku_prior,
            "standard_class_sku_prior": std_sku_prior,
            "lane_prior": lane_prior,
        },
        "spc_summary": spc_meta,
        "shipping_modes": [
            {
                "shipping_mode": str(r["Shipping Mode"]),
                "n_orders": int(r["n_orders"]),
                "order_share_pct": round(float(r["n_orders"] / len(df) * 100.0), 2),
                "late_orders": int(r["late_orders"]),
                "late_pct": round(float(r["late_pct"]), 2),
                "avg_scheduled_days": round(float(r["avg_scheduled"]), 2),
                "avg_real_days": round(float(r["avg_real"]), 2),
                "mean_delay_days": round(float(r["mean_delay"]), 2),
                "total_sales": round(float(r["total_sales"]), 2),
            }
            for _, r in mode_df.sort_values("late_pct", ascending=False).iterrows()
        ],
        "data_hygiene_rules": DATA_HYGIENE_RULES,
        "top_bottom_matrix": {
            "highest_eb_risk": fmt_sku_rows(sku_df.sort_values(["eb_late_pct", "n_orders"], ascending=[False, False]).head(5)),
            "lowest_eb_risk": fmt_sku_rows(sku_df.sort_values(["eb_late_pct", "n_orders"], ascending=[True, False]).head(5)),
            "largest_downward_shrinkage": fmt_sku_rows(sku_df.sort_values("shrinkage_pp", ascending=True).head(5)),
            "largest_upward_shrinkage": fmt_sku_rows(sku_df.sort_values("shrinkage_pp", ascending=False).head(5)),
        },
        "monthly_spc": [
            {
                "order_month": str(r["order_month"]),
                "n_orders": int(r["n_orders"]),
                "late_orders": int(r["late_orders"]),
                "late_pct": round(float(r["late_pct"]), 2),
                "ucl_late_pct": round(float(r["ucl_late_pct"]), 2),
                "lcl_late_pct": round(float(r["lcl_late_pct"]), 2),
                "late_spc_status": str(r["late_spc_status"]),
                "canceled_orders": int(r["canceled_orders"]),
                "canc_pct": round(float(r["canc_pct"]), 2),
                "ucl_canc_pct": round(float(r["ucl_canc_pct"]), 2),
                "lcl_canc_pct": round(float(r["lcl_canc_pct"]), 2),
                "canc_spc_status": str(r["canc_spc_status"]),
                "mean_delay_days": round(float(r["mean_delay_days"]), 3),
                "ucl_delay_days": round(float(r["ucl_delay_days"]), 3),
                "lcl_delay_days": round(float(r["lcl_delay_days"]), 3),
                "delay_spc_status": str(r["delay_spc_status"]),
                "std_delay_days": round(float(r["std_delay_days"]), 3),
                "standard_share_pct": round(float(r["standard_share"]), 1),
                "monthly_sales": round(float(r["monthly_sales"]), 2),
            }
            for _, r in spc_df.iterrows()
        ],
        "lanes": [
            {
                "market": str(r["Market"]),
                "order_region": str(r["Order Region"]),
                "shipping_mode": str(r["Shipping Mode"]),
                "n_orders": int(r["n_orders"]),
                "late_orders": int(r["late_orders"]),
                "raw_late_pct": round(float(r["raw_late_pct"]), 2),
                "eb_late_pct": round(float(r["eb_late_pct"]), 2),
                "ci_low_pct": round(float(r["ci_low_pct"]), 2),
                "ci_high_pct": round(float(r["ci_high_pct"]), 2),
                "shrinkage_pp": round(float(r["shrinkage_pp"]), 2),
                "mean_delay_days": round(float(r["mean_delay_days"]), 2),
                "late_sales": round(float(r["late_sales"]), 2),
            }
            for _, r in lane_df.sort_values("late_sales", ascending=False).iterrows()
        ],
        "skus": [
            {
                "sku_id": int(r["sku_id"]),
                "product_name": str(r["Product Name"]),
                "category_name": str(r["Category Name"]),
                "department_name": str(r["Department Name"]),
                "n_orders": int(r["n_orders"]),
                "late_orders": int(r["late_orders"]),
                "canceled_orders": int(r["canceled_orders"]),
                "raw_late_pct": round(float(r["raw_late_pct"]), 2),
                "eb_late_pct": round(float(r["eb_late_pct"]), 2),
                "ci_low_pct": round(float(r["ci_low_pct"]), 2),
                "ci_high_pct": round(float(r["ci_high_pct"]), 2),
                "shrinkage_pp": round(float(r["shrinkage_pp"]), 2),
                "shrinkage_weight": round(float(r["shrinkage_weight"]), 3),
                "mean_delay_days": round(float(r["mean_delay_days"]), 2),
                "total_sales": round(float(r["total_sales"]), 2),
                "late_sales": round(float(r["late_sales"]), 2),
                "total_profit": round(float(r["total_profit"]), 2),
            }
            for _, r in sku_df.sort_values("late_sales", ascending=False).iterrows()
        ],
    }

    (OUT_DIR / "sku_ops_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed Empirical Bayes & SPC pipeline in {time.time() - t0:.2f}s")
    print(f"Saved 6 charts to {IMG_DIR} and JSON summary to {OUT_DIR / 'sku_ops_summary.json'}")


if __name__ == "__main__":
    main()
