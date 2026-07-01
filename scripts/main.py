from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import CONFIG_DIR, KB_DIR, assert_dry_run_config, ensure_dirs, market_symbols, setup_logging
from scripts.privacy_audit import privacy_check

LOGGER = logging.getLogger(__name__)


def init_knowledge_base() -> None:
    import pandas as pd

    ensure_dirs()
    events_path = KB_DIR / "events.parquet"
    decisions_path = KB_DIR / "model_decisions.parquet"
    notes_path = KB_DIR / "research_notes.md"
    if not events_path.exists():
        pd.DataFrame(columns=["timestamp", "event_type", "market_context", "actual_outcome"]).to_parquet(events_path, index=False)
    if not decisions_path.exists():
        pd.DataFrame(columns=["timestamp", "model_version", "prediction", "actual", "was_correct", "config_change"]).to_parquet(decisions_path, index=False)
    if not notes_path.exists():
        notes_path.write_text("# Local Research Notes\n\n", encoding="utf-8")


def run_backtest() -> int:
    from scripts.install_freqtrade import run_freqtrade_dry_command

    config_path = CONFIG_DIR / "config.backtest.json"
    assert_dry_run_config(config_path)
    return run_freqtrade_dry_command(
        [
            "backtesting",
            "--config",
            str(config_path),
            "--userdir",
            "user_data",
            "--strategy",
            "MarketIntelStrategy",
        ]
    )


def run_full() -> None:
    from scripts.adaptation_engine import adapt
    from scripts.baseline_ml import train
    from scripts.collector import collect
    from scripts.feature_engineering import build_features
    from scripts.label_distribution_report import build_label_distribution_report
    from scripts.model_audit import run_model_audit
    from scripts.paper_trader import run_paper
    from scripts.probability_report import build_probability_report
    from scripts.recommendation_engine import build_recommendations
    from scripts.regime_engine import build_regime_labels
    from scripts.research_engine import run_research
    from scripts.signal_diagnostics import run_signal_diagnostics
    from scripts.stats_report import generate_report
    from scripts.target_audit import run_target_audit
    from scripts.trading_audit_report import build_trading_audit_report
    from scripts.trade_rejection_report import build_trade_rejection_report

    privacy_check()
    for symbol in market_symbols():
        collect(symbol=symbol)
        build_features(symbol=symbol)
    run_target_audit()
    build_regime_labels()
    train()
    build_trade_rejection_report()
    run_signal_diagnostics()
    build_probability_report()
    build_label_distribution_report()
    build_recommendations()
    run_model_audit()
    run_paper()
    run_research()
    adapt()
    generate_report()
    build_trading_audit_report()
    init_knowledge_base()


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Market Intelligence Trading Lab v0.1")
    parser.add_argument(
        "command",
        choices=[
            "privacy-check",
            "db-init",
            "collect",
            "collector-service",
            "features",
            "regime",
            "train",
            "audit",
            "integrity-audit",
            "execution",
            "monte-carlo",
            "bootstrap",
            "benchmarks",
            "forward-paper",
            "rolling-walkforward",
            "edge-verdict",
            "target-audit",
            "diagnostics",
            "probability-report",
            "label-report",
            "rejection-report",
            "recommendations",
            "backtest",
            "paper",
            "research",
            "research-service",
            "telegram-notifier-service",
            "dashboard",
            "adapt",
            "report",
            "trading-audit",
            "full",
        ],
    )
    args = parser.parse_args(argv)
    ensure_dirs()

    if args.command == "privacy-check":
        privacy_check()
        return 0

    if args.command == "db-init":
        from scripts.db import init_db

        init_db()
        return 0

    if args.command == "collector-service":
        from scripts.collector_service import run_collector_service

        run_collector_service()
        return 0

    if args.command == "research-service":
        from scripts.research_service import run_research_service

        run_research_service()
        return 0

    if args.command == "telegram-notifier-service":
        from scripts.telegram_notifier import run_telegram_notifier_service

        run_telegram_notifier_service()
        return 0

    if args.command == "dashboard":
        import uvicorn

        uvicorn.run("scripts.dashboard_app:app", host="0.0.0.0", port=8000)
        return 0

    init_knowledge_base()

    if args.command == "collect":
        from scripts.collector import collect

        for symbol in market_symbols():
            collect(symbol=symbol)
    elif args.command == "features":
        from scripts.feature_engineering import build_features

        for symbol in market_symbols():
            build_features(symbol=symbol)
    elif args.command == "regime":
        from scripts.regime_engine import build_regime_labels

        build_regime_labels()
    elif args.command == "train":
        from scripts.baseline_ml import train

        train()
    elif args.command == "audit":
        from scripts.model_audit import run_model_audit

        run_model_audit()
    elif args.command == "integrity-audit":
        from scripts.integrity_audit import run_integrity_audit

        print(run_integrity_audit())
    elif args.command == "execution":
        from scripts.execution_engine import run_execution_sweep

        run_execution_sweep()
    elif args.command == "monte-carlo":
        from scripts.monte_carlo import run_monte_carlo

        run_monte_carlo()
    elif args.command == "bootstrap":
        from scripts.bootstrap_engine import run_bootstrap

        run_bootstrap()
    elif args.command == "benchmarks":
        from scripts.benchmark_engine import run_benchmarks

        run_benchmarks()
    elif args.command == "forward-paper":
        from scripts.forward_paper_engine import run_forward_paper_engine

        run_forward_paper_engine()
    elif args.command == "rolling-walkforward":
        from scripts.rolling_walkforward import run_rolling_walkforward

        run_rolling_walkforward()
    elif args.command == "edge-verdict":
        from scripts.edge_verdict import build_edge_verdict

        print(build_edge_verdict())
    elif args.command == "target-audit":
        from scripts.target_audit import run_target_audit

        run_target_audit()
    elif args.command == "diagnostics":
        from scripts.signal_diagnostics import run_signal_diagnostics

        run_signal_diagnostics()
    elif args.command == "probability-report":
        from scripts.probability_report import build_probability_report

        build_probability_report()
    elif args.command == "label-report":
        from scripts.label_distribution_report import build_label_distribution_report

        build_label_distribution_report()
    elif args.command == "rejection-report":
        from scripts.trade_rejection_report import build_trade_rejection_report

        build_trade_rejection_report()
    elif args.command == "recommendations":
        from scripts.recommendation_engine import build_recommendations

        print(build_recommendations())
    elif args.command == "backtest":
        return run_backtest()
    elif args.command == "paper":
        from scripts.paper_trader import run_paper

        assert_dry_run_config(CONFIG_DIR / "config.paper.json")
        run_paper()
    elif args.command == "research":
        from scripts.research_engine import run_research

        run_research()
    elif args.command == "adapt":
        from scripts.adaptation_engine import adapt

        print(adapt())
    elif args.command == "report":
        from scripts.stats_report import generate_report

        print(generate_report())
    elif args.command == "trading-audit":
        from scripts.trading_audit_report import build_trading_audit_report

        print(build_trading_audit_report())
    elif args.command == "full":
        run_full()
    else:
        LOGGER.error("Unknown command: %s", args.command)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
