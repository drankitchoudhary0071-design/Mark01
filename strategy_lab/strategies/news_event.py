"""
18. News / Event-driven trading — STUB.

Requires an external news/economic-calendar feed. This module does **not**
fabricate headlines or synthetic event timestamps. Until a feed is wired,
``generate_signals`` returns an empty list and reports are flagged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp
from strategy_lab.engine import indicators as ind


class NewsEventStrategy(Strategy):
    name = "news_event"
    library = "stub — external news feed required"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "events_csv": "",  # path to CSV: timestamp,event,bias(long|short|flat),importance
            "min_importance": 2,  # 1=low 2=med 3=high — only if feed provided
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 2.0,
            "max_hold_bars": 24,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[EXTERNAL FEED REQUIRED] news_event emits zero signals unless "
            "`events_csv` points to a real calendar/news file.",
            "[DO NOT FABRICATE] No synthetic headlines or event times are generated.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        path = str(self.params.get("events_csv") or "").strip()
        if not path:
            return []
        pth = Path(path)
        if not pth.exists():
            return []

        events = pd.read_csv(pth, parse_dates=["timestamp"])
        if events["timestamp"].dt.tz is None:
            events["timestamp"] = events["timestamp"].dt.tz_localize("UTC")
        else:
            events["timestamp"] = events["timestamp"].dt.tz_convert("UTC")

        out = df.copy()
        out["atr"] = ind.atr(out, int(self.params["atr_period"]))
        ts_to_i = {
            pd.Timestamp(t).tz_convert("UTC"): i
            for i, t in enumerate(out["timestamp"])
        }
        # Map event to nearest bar at or after event time
        bar_times = list(ts_to_i.keys())

        signals: list[Signal] = []
        for _, ev in events.iterrows():
            if int(ev.get("importance", 0)) < int(self.params["min_importance"]):
                continue
            bias = str(ev.get("bias", "flat")).lower()
            if bias not in ("long", "short"):
                continue
            et = pd.Timestamp(ev["timestamp"]).tz_convert("UTC")
            # find first bar >= event
            idx = None
            for bt in bar_times:
                if bt >= et:
                    idx = ts_to_i[bt]
                    break
            if idx is None:
                continue
            row = out.iloc[idx]
            if pd.isna(row["atr"]) or row["atr"] <= 0:
                continue
            entry = float(row["close"])
            stop, tp = atr_stop_tp(
                entry,
                bias,
                float(row["atr"]),
                self.params["stop_atr"],
                self.params["tp_atr"],
            )
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(row["timestamp"]),
                    direction=bias,  # type: ignore[arg-type]
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    pattern=f"news:{ev.get('event', 'event')}",
                    max_hold_bars=int(self.params["max_hold_bars"]),
                    meta={"source": path},
                )
            )
        return signals
