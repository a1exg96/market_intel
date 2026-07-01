from __future__ import annotations

import logging

import pandas as pd

from scripts.common import REPORTS_DIR, ensure_dirs

LOGGER = logging.getLogger(__name__)
REQUIRED_TRADES = 100


def _trade_source() -> pd.DataFrame:
    for filename in ["forward_results.csv", "realistic_execution_trades.csv", "trades.csv"]:
        path = REPORTS_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if not df.empty:
                df["_source"] = filename
                return df
    return pd.DataFrame()


def build_edge_verdict() -> str:
    ensure_dirs()
    reasons: list[str] = []
    trades = _trade_source()
    trade_count = int(len(trades))
    source = str(trades["_source"].iloc[0]) if not trades.empty and "_source" in trades else "none"
    reasons.append(f"trade_source={source}")
    reasons.append(f"trades={trade_count}")

    if trade_count < REQUIRED_TRADES:
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"minimum_required_trades={REQUIRED_TRADES}")
    else:
        pf = 0.0
        if "pnl_pct" in trades:
            gains = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].sum()
            losses = trades.loc[trades["pnl_pct"] < 0, "pnl_pct"].sum()
            pf = float("inf") if losses == 0 and gains > 0 else float(gains / abs(losses)) if losses != 0 else 0.0
        elif "trade_return" in trades:
            gains = trades.loc[trades["trade_return"] > 0, "trade_return"].sum()
            losses = trades.loc[trades["trade_return"] < 0, "trade_return"].sum()
            pf = float("inf") if losses == 0 and gains > 0 else float(gains / abs(losses)) if losses != 0 else 0.0
        winrate = float((trades.get("pnl", trades.get("pnl_usd", pd.Series(dtype=float))) > 0).mean()) if not trades.empty else 0.0
        reasons.append(f"profit_factor={pf:.4f}")
        reasons.append(f"winrate={winrate:.2%}")

        mc_stable = False
        mc_path = REPORTS_DIR / "monte_carlo.csv"
        if mc_path.exists():
            mc = pd.read_csv(mc_path)
            if not mc.empty:
                mc_stable = float((mc["total_return"] < 0).mean()) < 0.25 and float((mc["profit_factor"] > 1.2).mean()) > 0.6
                reasons.append(f"monte_carlo_probability_loss={float((mc['total_return'] < 0).mean()):.2%}")

        rolling_stable = False
        rwf_path = REPORTS_DIR / "rolling_walkforward.csv"
        if rwf_path.exists():
            rwf = pd.read_csv(rwf_path)
            insufficient = "note" in rwf.columns and rwf["note"].astype(str).str.contains("insufficient", case=False, na=False).any()
            rolling_stable = (not insufficient) and len(rwf) >= 3 and "profit_factor" in rwf and float((rwf["profit_factor"] > 1.2).mean()) >= 0.5
            if insufficient:
                reasons.append("rolling_walkforward_insufficient_history=true")

        if pf < 1:
            verdict = "NO_EDGE"
        elif pf < 1.2:
            verdict = "WEAK_EDGE"
        elif mc_stable and rolling_stable:
            verdict = "PROMISING_EDGE"
        else:
            verdict = "WEAK_EDGE"

    report = "# Edge Verdict\n\n" f"Verdict: **{verdict}**\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
    (REPORTS_DIR / "edge_verdict.md").write_text(report, encoding="utf-8")
    pd.DataFrame([{"verdict": verdict, "required_trades": REQUIRED_TRADES, "trades": trade_count, "reasons": "; ".join(reasons)}]).to_csv(REPORTS_DIR / "edge_verdict.csv", index=False)
    LOGGER.info("Saved edge verdict=%s", verdict)
    return verdict


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    print(build_edge_verdict())
