from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from pandas.errors import EmptyDataError

from scripts.common import LAB_CONFIG, REPORTS_DIR, ensure_dirs, max_drawdown, profit_factor

LOGGER = logging.getLogger(__name__)


def _audit_summary() -> tuple[int, int]:
    path = REPORTS_DIR / "outbound_audit.log"
    if not path.exists():
        return 0, 0
    audit = pd.read_csv(path)
    if audit.empty:
        return 0, 0
    allowed_values = audit["allowed"].astype(str).str.lower()
    allowed = int((allowed_values == "true").sum())
    blocked = int((allowed_values == "false").sum())
    return allowed, blocked


def _signal_diagnostics_section() -> tuple[str, str]:
    diagnostics_path = REPORTS_DIR / "signal_diagnostics.csv"
    if not diagnostics_path.exists():
        return (
            "## Signal Diagnostics\n\n- Predictions generated: n/a\n- Final executable trades: n/a\n",
            "Signal diagnostics are not available yet.",
        )
    diagnostics = pd.read_csv(diagnostics_path)
    if diagnostics.empty:
        return (
            "## Signal Diagnostics\n\n- Predictions generated: 0\n- Final executable trades: 0\n",
            "No predictions were available for diagnostics.",
        )
    row = diagnostics.iloc[0]
    most_common = "n/a"
    rejection_path = REPORTS_DIR / "trade_rejections.csv"
    if rejection_path.exists():
        rejections = pd.read_csv(rejection_path)
        if not rejections.empty:
            most_common = str(rejections["reason_rejected"].value_counts().index[0])
    section = f"""## Signal Diagnostics

- Predictions generated: {int(row.get("total_predictions", 0))}
- Long opportunities found: {int(row.get("long_opportunities_found", 0))}
- Short opportunities found: {int(row.get("short_opportunities_found", 0))}
- Above confidence threshold: {int(row.get("signals_above_threshold", 0))}
- Rejected by confidence: {int(row.get("rejected_by_confidence", 0))}
- Rejected by no_trade class: {int(row.get("no_trade_predictions", 0))}
- Rejected by regime filter: {int(row.get("rejected_by_regime_filter", 0))}
- Rejected by risk engine: {int(row.get("rejected_by_risk_engine", 0))}
- Rejected by adaptation engine: {int(row.get("rejected_by_adaptation_engine", 0))}
- Final executable trades: {int(row.get("final_executable_trades", 0))}
- Most common rejection reason: {most_common}
"""
    total = max(int(row.get("total_predictions", 0)), 1)
    no_trade_share = int(row.get("no_trade_predictions", 0)) / total
    executable = int(row.get("final_executable_trades", 0))
    if executable == 0:
        diagnosis = (
            f"No trades were executed because {no_trade_share:.1%} of decisions were no-trade "
            f"and {int(row.get('signals_above_threshold', 0))} setup predictions exceeded confidence threshold."
        )
    else:
        diagnosis = (
            f"{executable} paper trades were executable. {no_trade_share:.1%} of decisions still resolved to no-trade, "
            f"so the system is selective rather than trading every candle."
        )
    return section, diagnosis


def _target_and_calibration_section() -> str:
    target_path = REPORTS_DIR / "target_distribution.csv"
    calibration_path = REPORTS_DIR / "calibration_report.md"
    walk_path = REPORTS_DIR / "walk_forward_results.csv"
    target_threshold = "n/a"
    class_distribution = "n/a"
    if walk_path.exists():
        walk = pd.read_csv(walk_path)
        if not walk.empty and "target_threshold" in walk.columns:
            target_threshold = f"{float(walk['target_threshold'].iloc[0]):.2%}"
            selected_horizon = str(walk["horizon"].iloc[0]) if "horizon" in walk.columns else None
        else:
            selected_horizon = None
    else:
        selected_horizon = None
    if target_path.exists():
        target = pd.read_csv(target_path)
        if not target.empty:
            if target_threshold != "n/a":
                selected = target[target["target_threshold"].round(6) == round(float(target_threshold.strip("%")) / 100, 6)]
            else:
                selected = target
            if selected_horizon and "horizon" in selected.columns:
                selected = selected[selected["horizon"].astype(str) == selected_horizon]
            if not selected.empty:
                row = selected.iloc[0]
                class_distribution = f"long={int(row['long_positive'])}, short={int(row['short_positive'])}, neither={int(row['neither_setup'])}"
    calibration_quality = "n/a"
    if calibration_path.exists():
        text = calibration_path.read_text(encoding="utf-8")
        lines = [line.strip("- ") for line in text.splitlines() if "ECE:" in line]
        calibration_quality = "; ".join(lines[:2]) if lines else "see calibration_report.md"
    return f"""## Setup Model Details

- Target threshold: {target_threshold}
- Class distribution: {class_distribution}
- Probability calibration quality: {calibration_quality}
"""


def _model_quality_section() -> str:
    quality_path = REPORTS_DIR / "model_quality.csv"
    regime_path = REPORTS_DIR / "regime_quality.csv"
    if not quality_path.exists():
        return "## Model Quality Gate\n\nNo model quality gate report is available yet.\n"
    quality = pd.read_csv(quality_path)
    if quality.empty:
        return "## Model Quality Gate\n\nNo model quality rows were generated.\n"
    lines = [
        (
            f"- {row['side']}: status={row['status']}, samples={int(row['samples'])}, "
            f"precision={float(row['precision']):.2%}, expectancy={float(row['expectancy']):.4f}, "
            f"ece={float(row['ece']):.4f}, reasons={row.get('reasons', '') or 'none'}"
        )
        for _, row in quality.iterrows()
    ]
    blocked_regimes = ""
    if regime_path.exists():
        try:
            regimes = pd.read_csv(regime_path)
        except EmptyDataError:
            regimes = pd.DataFrame()
        if not regimes.empty and "status" in regimes:
            blocked = regimes[regimes["status"].astype(str) == "blocked"].head(5)
            if not blocked.empty:
                blocked_regimes = "\nBlocked regimes: " + "; ".join(
                    f"{row['side']} {row['regime']} ({row['reasons']})" for _, row in blocked.iterrows()
                )
    return "## Model Quality Gate\n\n" + "\n".join(lines) + blocked_regimes + "\n"


def _probability_summary_section() -> str:
    path = REPORTS_DIR / "probability_buckets.csv"
    if not path.exists():
        return "## Probability Buckets Summary\n\nNo probability bucket report is available yet.\n"
    buckets = pd.read_csv(path)
    if buckets.empty:
        return "## Probability Buckets Summary\n\nNo probability buckets were generated.\n"
    important = buckets[buckets["predictions_count"] > 0].sort_values("predictions_count", ascending=False).head(3)
    if important.empty:
        return "## Probability Buckets Summary\n\nNo populated probability buckets.\n"
    lines = [
        f"- {row['bucket']}: predictions={int(row['predictions_count'])}, up={int(row['predicted_up'])}, down={int(row['predicted_down'])}, no_trade={int(row['predicted_no_trade'])}, avg_return={row['avg_future_return']:.4f}"
        for _, row in important.iterrows()
    ]
    return "## Probability Buckets Summary\n\n" + "\n".join(lines) + "\n"


def _robustness_section() -> str:
    exec_path = REPORTS_DIR / "realistic_execution.csv"
    mc_path = REPORTS_DIR / "monte_carlo.csv"
    rwf_path = REPORTS_DIR / "rolling_walkforward.csv"
    verdict_path = REPORTS_DIR / "edge_verdict.csv"
    bootstrap_path = REPORTS_DIR / "bootstrap_summary.csv"
    benchmark_path = REPORTS_DIR / "benchmark_results.csv"
    realistic_return = "n/a"
    if exec_path.exists():
        execution = pd.read_csv(exec_path)
        if not execution.empty:
            conservative = execution.sort_values(["fee_pct", "slippage_pct"], ascending=False).iloc[0]
            realistic_return = f"{float(conservative['total_return']):.2%}"
    mc_median = "n/a"
    mc_loss = "n/a"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        if not mc.empty:
            mc_median = f"{float(mc['total_return'].median()):.2%}"
            mc_loss = f"{float((mc['total_return'] < 0).mean()):.2%}"
    rolling_score = "n/a"
    if rwf_path.exists():
        rwf = pd.read_csv(rwf_path)
        if "stability_score" in rwf and not rwf.empty:
            rolling_score = f"{float(rwf['stability_score'].mean()):.3f}"
    verdict = "n/a"
    trades_accumulated = "n/a"
    if verdict_path.exists():
        verdict_df = pd.read_csv(verdict_path)
        if not verdict_df.empty:
            verdict = str(verdict_df["verdict"].iloc[0])
            trades_accumulated = str(int(verdict_df["trades"].iloc[0])) if "trades" in verdict_df else "n/a"
    bootstrap_ci = "n/a"
    if bootstrap_path.exists():
        boot = pd.read_csv(bootstrap_path)
        row = boot[boot["metric"] == "total_return"]
        if not row.empty:
            bootstrap_ci = f"{float(row['p05'].iloc[0]):.2%} to {float(row['p95'].iloc[0]):.2%}"
    benchmark_comparison = "n/a"
    if benchmark_path.exists():
        bench = pd.read_csv(benchmark_path)
        if not bench.empty:
            best = bench.sort_values("total_return", ascending=False).iloc[0]
            benchmark_comparison = f"best={best['benchmark']} return={float(best['total_return']):.2%}"
    return f"""## Robustness

- Trades accumulated: {trades_accumulated}
- Required trades: 100
- Sample sufficiency: {"sufficient" if trades_accumulated != "n/a" and int(trades_accumulated) >= 100 else "insufficient"}
- Realistic execution return: {realistic_return}
- Monte Carlo median return: {mc_median}
- Probability of losing money: {mc_loss}
- Bootstrap confidence interval: {bootstrap_ci}
- Rolling stability score: {rolling_score}
- Benchmark comparison: {benchmark_comparison}
- Edge verdict: {verdict}
"""


def generate_report() -> str:
    ensure_dirs()
    trades_path = REPORTS_DIR / "trades.csv"
    try:
        trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    except EmptyDataError:
        trades = pd.DataFrame()
    if not trades.empty:
        if "timestamp" not in trades.columns and "closed_at" in trades.columns:
            trades["timestamp"] = trades["closed_at"]
        if "timestamp" in trades.columns:
            trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce", utc=True)
    if trades.empty:
        balance = LAB_CONFIG.initial_balance
        pnl_day = 0.0
        winrate = 0.0
        pf = 0.0
        dd = 0.0
        best = "n/a"
        worst = "n/a"
    else:
        balance = float(trades["balance_after"].iloc[-1])
        last_day = trades["timestamp"].max().date()
        day_trades = trades[trades["timestamp"].dt.date == last_day]
        pnl_day = float(day_trades["pnl_usd"].sum())
        winrate = float((trades["pnl_usd"] > 0).mean())
        pf = profit_factor(trades["pnl_pct"])
        dd = max_drawdown([LAB_CONFIG.initial_balance] + trades["balance_after"].tolist())
        best_row = trades.sort_values("pnl_usd", ascending=False).iloc[0]
        worst_row = trades.sort_values("pnl_usd", ascending=True).iloc[0]
        best = f"{best_row['side']} {best_row['pnl_usd']:.2f} USD confidence={best_row['confidence']:.3f}"
        worst = f"{worst_row['side']} {worst_row['pnl_usd']:.2f} USD confidence={worst_row['confidence']:.3f}"

    feature_path = REPORTS_DIR / "feature_importance.csv"
    if feature_path.exists():
        top_features = ", ".join(pd.read_csv(feature_path).head(5)["feature"].astype(str).tolist())
    else:
        top_features = "n/a"
    allowed, blocked = _audit_summary()
    signal_section, diagnosis = _signal_diagnostics_section()
    probability_section = _probability_summary_section()
    target_section = _target_and_calibration_section()
    model_quality_section = _model_quality_section()
    robustness_section = _robustness_section()
    degraded = "unknown"
    if not trades.empty:
        cutoff = trades["timestamp"].max() - pd.Timedelta(days=7)
        recent = trades[trades["timestamp"] >= cutoff]
        degraded = "yes" if not recent.empty and recent["pnl_usd"].sum() < 0 else "no"

    report = f"""# Daily Market Intelligence Report

Generated: {datetime.now(timezone.utc).isoformat()}

- Current balance: {balance:.2f} USD
- Daily PnL: {pnl_day:.2f} USD
- Trades: {0 if trades.empty else len(trades)}
- Winrate: {winrate:.2%}
- Profit factor: {pf:.2f}
- Max drawdown: {dd:.2%}
- Active configuration: paper-only, balance={LAB_CONFIG.initial_balance:.2f}, max risk per trade={LAB_CONFIG.risk_per_trade:.2%}
- Top features: {top_features}
- Best signal: {best}
- Worst signal: {worst}
- Outbound audit: allowed={allowed}, blocked={blocked}
- Blocked outbound requests present: {"yes" if blocked else "no"}
- Performance degraded over last 7 days: {degraded}

{signal_section}

{target_section}

{model_quality_section}

{probability_section}

{robustness_section}

## Diagnosis

{diagnosis}

This is a local research/demo report, not financial advice.
"""
    (REPORTS_DIR / "daily_report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Saved daily report.")
    return report


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(generate_report())
