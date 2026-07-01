from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

LOGGER = logging.getLogger(__name__)


def database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://market:market@postgres:5432/market_intel")


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(database_url(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS market_ticks (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price DOUBLE PRECISION,
            quantity DOUBLE PRECISION,
            raw JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS candles (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            quote_volume DOUBLE PRECISION,
            source TEXT,
            UNIQUE (ts, exchange, symbol, timeframe)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS funding (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            funding_rate DOUBLE PRECISION,
            UNIQUE (ts, exchange, symbol)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS open_interest (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_interest DOUBLE PRECISION,
            UNIQUE (ts, exchange, symbol)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS liquidations (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            quantity DOUBLE PRECISION,
            price DOUBLE PRECISION,
            raw JSONB DEFAULT '{}'::jsonb
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS news (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            source TEXT,
            title TEXT,
            sentiment DOUBLE PRECISION,
            raw JSONB DEFAULT '{}'::jsonb
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS signals (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            regime TEXT,
            signal TEXT,
            confidence DOUBLE PRECISION,
            model_version TEXT,
            raw JSONB DEFAULT '{}'::jsonb,
            UNIQUE (ts, symbol, model_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            position_size DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            pnl_usd DOUBLE PRECISION,
            pnl_pct DOUBLE PRECISION,
            balance_after DOUBLE PRECISION,
            reason TEXT,
            model_version TEXT,
            UNIQUE (ts, symbol, side, entry_price, exit_price, model_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS equity_curve (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            balance DOUBLE PRECISION NOT NULL,
            source TEXT,
            UNIQUE (ts, balance, source)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_reports (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT now(),
            markdown TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT now(),
            service TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            raw JSONB DEFAULT '{}'::jsonb
        )
        """,
    ]
    with get_conn() as conn:
        conn.execute("SELECT pg_advisory_lock(2026063001)")
        try:
            for statement in statements:
                conn.execute(statement)
        finally:
            conn.execute("SELECT pg_advisory_unlock(2026063001)")
    LOGGER.info("PostgreSQL schema ready.")


def execute_many(conn: psycopg.Connection, query: str, rows: list[tuple[Any, ...]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(query, rows)


def log_event(service: str, level: str, message: str, raw: dict[str, Any] | None = None) -> None:
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO system_logs (service, level, message, raw) VALUES (%s, %s, %s, %s)",
                (service, level, message, Json(raw or {})),
            )
    except Exception as exc:
        LOGGER.warning("Could not write system log: %s", exc)


def upsert_candles(df: pd.DataFrame, exchange: str = "binance", timeframe: str = "5m") -> int:
    if df.empty:
        return 0
    rows = [
        (
            row["timestamp"],
            exchange,
            row["symbol"],
            timeframe,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            float(row.get("quote_volume", row.get("quote_asset_volume", 0.0))),
            row.get("source", exchange),
        )
        for _, row in df.iterrows()
    ]
    with get_conn() as conn:
        execute_many(
            conn,
            """
            INSERT INTO candles (ts, exchange, symbol, timeframe, open, high, low, close, volume, quote_volume, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ts, exchange, symbol, timeframe)
            DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
                          volume=EXCLUDED.volume, quote_volume=EXCLUDED.quote_volume, source=EXCLUDED.source
            """,
            rows,
        )
    return len(rows)


def upsert_futures_context(df: pd.DataFrame, exchange: str = "binance", symbol: str = "BTCUSDT") -> int:
    if df.empty:
        return 0
    funding_rows = []
    oi_rows = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if "funding_rate" in row and pd.notna(row["funding_rate"]):
            funding_rows.append((ts, exchange, symbol, float(row["funding_rate"])))
        if "open_interest" in row and pd.notna(row["open_interest"]):
            oi_rows.append((ts, exchange, symbol, float(row["open_interest"])))
    with get_conn() as conn:
        if funding_rows:
            execute_many(
                conn,
                """
                INSERT INTO funding (ts, exchange, symbol, funding_rate)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (ts, exchange, symbol) DO UPDATE SET funding_rate=EXCLUDED.funding_rate
                """,
                funding_rows,
            )
        if oi_rows:
            execute_many(
                conn,
                """
                INSERT INTO open_interest (ts, exchange, symbol, open_interest)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (ts, exchange, symbol) DO UPDATE SET open_interest=EXCLUDED.open_interest
                """,
                oi_rows,
            )
    return len(funding_rows) + len(oi_rows)


def insert_signals(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (
            row["timestamp"],
            row["symbol"],
            row.get("regime", "UNKNOWN"),
            row.get("predicted_direction", row.get("signal", "flat")),
            float(row.get("confidence", 0.0)),
            row.get("model_version", "unknown"),
            Json({k: str(v) for k, v in row.items()}),
        )
        for _, row in df.iterrows()
    ]
    with get_conn() as conn:
        execute_many(
            conn,
            """
            INSERT INTO signals (ts, symbol, regime, signal, confidence, model_version, raw)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ts, symbol, model_version) DO NOTHING
            """,
            rows,
        )
    return len(rows)


def insert_daily_report(markdown: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO daily_reports (markdown) VALUES (%s)", (markdown,))


def insert_paper_trades(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (
            row["timestamp"],
            row["symbol"],
            row["side"],
            float(row["entry_price"]),
            float(row["exit_price"]),
            float(row["position_size"]),
            float(row["confidence"]),
            float(row["pnl_usd"]),
            float(row["pnl_pct"]),
            float(row["balance_after"]),
            row.get("reason", ""),
            row.get("model_version", "unknown"),
        )
        for _, row in df.iterrows()
    ]
    with get_conn() as conn:
        execute_many(
            conn,
            """
            INSERT INTO paper_trades
            (ts, symbol, side, entry_price, exit_price, position_size, confidence, pnl_usd, pnl_pct, balance_after, reason, model_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ts, symbol, side, entry_price, exit_price, model_version) DO NOTHING
            """,
            rows,
        )
        for row in rows:
            conn.execute(
                "INSERT INTO equity_curve (ts, balance, source) VALUES (%s,%s,%s) ON CONFLICT (ts, balance, source) DO NOTHING",
                (row[0], row[9], "paper_trader"),
            )
    return len(rows)


def latest_dashboard_snapshot() -> dict[str, Any]:
    with get_conn() as conn:
        trade_stats = conn.execute(
            """
            SELECT
              COUNT(*)::int AS trades,
              COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),0)::int AS wins,
              COALESCE(SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END),0)::int AS losses,
              COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END),0) AS gross_profit,
              COALESCE(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END),0) AS gross_loss,
              COALESCE(SUM(pnl_usd),0) AS pnl
            FROM paper_trades
            """
        ).fetchone()
        latest_balance = conn.execute(
            "SELECT balance_after FROM paper_trades ORDER BY ts DESC, id DESC LIMIT 1"
        ).fetchone()
        last_signal = conn.execute("SELECT * FROM signals ORDER BY ts DESC LIMIT 1").fetchone()
        trades = conn.execute("SELECT * FROM paper_trades ORDER BY ts DESC LIMIT 100").fetchall()
        logs = conn.execute("SELECT * FROM system_logs ORDER BY ts DESC LIMIT 100").fetchall()
        equity = conn.execute("SELECT ts, balance FROM equity_curve ORDER BY ts ASC LIMIT 500").fetchall()
    wins = trade_stats["wins"] or 0
    total = trade_stats["trades"] or 0
    gross_profit = float(trade_stats["gross_profit"] or 0)
    gross_loss = abs(float(trade_stats["gross_loss"] or 0))
    balances = [float(row["balance"]) for row in equity]
    if balances:
        peaks: list[float] = []
        current_peak = balances[0]
        for value in balances:
            current_peak = max(current_peak, value)
            peaks.append(current_peak)
        max_dd = min((value - peak) / peak for value, peak in zip(balances, peaks) if peak) if peaks else 0.0
    else:
        max_dd = 0.0
    if gross_loss:
        profit_factor_value: float | None = float(gross_profit / gross_loss)
    elif gross_profit:
        profit_factor_value = None
    else:
        profit_factor_value = 0.0
    return {
        "stats": {
            "balance": float(latest_balance["balance_after"]) if latest_balance else 1000.0,
            "pnl": float(trade_stats["pnl"] or 0),
            "trades": int(total),
            "wins": int(wins),
            "losses": int(trade_stats["losses"] or 0),
            "winrate": float(wins / total) if total else 0.0,
            "profit_factor": profit_factor_value,
            "max_drawdown": float(max_dd),
        },
        "last_signal": dict(last_signal) if last_signal else None,
        "trades": [dict(row) for row in trades],
        "logs": [dict(row) for row in logs],
        "equity": [dict(row) for row in equity],
    }
