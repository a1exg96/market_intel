from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from freqtrade.strategy import IStrategy
except Exception:
    class IStrategy:  # type: ignore[no-redef]
        pass


class MarketIntelStrategy(IStrategy):
    """Freqtrade strategy bridge that consumes local paper/research signals only."""

    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 0.02}
    stoploss = -0.03
    process_only_new_candles = True

    def _signals(self) -> pd.DataFrame:
        path = Path(__file__).resolve().parents[2] / "data" / "processed" / "predictions.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "predicted_direction"])
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[["timestamp", "predicted_direction", "confidence"]]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        signals = self._signals()
        if signals.empty:
            dataframe["mi_signal"] = "flat"
            dataframe["mi_confidence"] = 0.0
            return dataframe
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
        merged = pd.merge_asof(dataframe.sort_values("date"), signals.sort_values("timestamp"), left_on="date", right_on="timestamp", direction="backward")
        dataframe["mi_signal"] = merged["predicted_direction"].fillna("flat").to_numpy()
        dataframe["mi_confidence"] = merged["confidence"].fillna(0.0).to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["mi_signal"] == "up", "enter_long"] = 1
        dataframe.loc[dataframe["mi_signal"] == "down", "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["mi_signal"] == "down", "exit_long"] = 1
        dataframe.loc[dataframe["mi_signal"] == "up", "exit_short"] = 1
        return dataframe

