"""
CLI runner — backtest all (or selected) strategy families with walk-forward OOS.

Examples
--------
  # All strategies on default pairs (PAXGUSDT, BTCUSDT), 1h, 365d
  python -m strategy_lab.run_all

  # Custom pair / timeframe
  python -m strategy_lab.run_all --symbols PAXGUSDT BTCUSDT --interval 1h --days 180

  # Subset
  python -m strategy_lab.run_all --strategies trend_following mean_reversion scalping

  # Holdout instead of rolling walk-forward
  python -m strategy_lab.run_all --no-walk-forward --oos-frac 0.25

  # Refresh Binance cache
  python -m strategy_lab.run_all --refresh-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab import config as cfg
from strategy_lab.data.fetch import load_or_fetch, load_pair
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import (
    compute_metrics,
    format_metrics,
    overfitting_assessment,
    trades_to_frame,
)
from strategy_lab.engine.walk_forward import (
    folds_summary_frame,
    run_holdout,
    run_walk_forward,
)
from strategy_lab.strategies import SPECIAL_SIM, STRATEGY_REGISTRY
from strategy_lab.strategies.grid_trading import simulate_grid
from strategy_lab.strategies.market_making import simulate_market_making
from strategy_lab.strategies.statistical_arbitrage import simulate_pairs
from strategy_lab.strategies.cointegration_pairs import simulate_coint_pairs
from strategy_lab.strategies.vwap_twap import simulate_vwap_twap

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="strategy_lab multi-family backtester")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=list(cfg.DEFAULT_SYMBOLS),
        help="Binance symbols (default: PAXGUSDT BTCUSDT)",
    )
    p.add_argument("--interval", default=cfg.DEFAULT_INTERVAL)
    p.add_argument("--days", type=int, default=cfg.DEFAULT_DAYS)
    p.add_argument("--scalp-interval", default=cfg.SCALP_INTERVAL)
    p.add_argument("--scalp-days", type=int, default=cfg.SCALP_DAYS)
    p.add_argument("--capital", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--risk", type=float, default=cfg.RISK_PER_TRADE)
    p.add_argument("--commission", type=float, default=cfg.COMMISSION_RATE)
    p.add_argument("--half-spread", type=float, default=cfg.HALF_SPREAD)
    p.add_argument("--slippage", type=float, default=cfg.SLIPPAGE)
    p.add_argument("--strategies", nargs="*", default=None, help="Subset of strategy names")
    p.add_argument("--no-walk-forward", action="store_true")
    p.add_argument("--oos-frac", type=float, default=cfg.HOLDOUT_OOS_FRAC)
    p.add_argument("--wf-train", type=int, default=cfg.WF_TRAIN_BARS)
    p.add_argument("--wf-test", type=int, default=cfg.WF_TEST_BARS)
    p.add_argument("--wf-step", type=int, default=cfg.WF_STEP_BARS)
    p.add_argument("--refresh-data", action="store_true")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR))
    return p.parse_args(argv)


def _costs(args: argparse.Namespace) -> CostModel:
    return CostModel(
        commission_rate=args.commission,
        half_spread=args.half_spread,
        slippage=args.slippage,
    )


def _scale_wf_for_interval(interval: str, args: argparse.Namespace) -> tuple[int, int, int]:
    """Roughly scale 1h bar counts to other timeframes."""
    mult = {
        "1m": 60,
        "3m": 20,
        "5m": 12,
        "15m": 4,
        "30m": 2,
        "1h": 1,
        "4h": 0.25,
        "1d": 1 / 24,
    }.get(interval, 1)
    return (
        max(50, int(args.wf_train * mult)),
        max(20, int(args.wf_test * mult)),
        max(20, int(args.wf_step * mult)),
    )


def _run_signal_strategy(
    name: str,
    df: pd.DataFrame,
    symbol: str,
    args: argparse.Namespace,
    costs: CostModel,
) -> dict[str, Any]:
    cls = STRATEGY_REGISTRY[name]
    strat = cls()
    signal_fn = strat.generate_signals
    train, test, step = _scale_wf_for_interval(args.interval if name != "scalping" else args.scalp_interval, args)

    if args.no_walk_forward:
        is_res, oos_res, is_m, oos_m = run_holdout(
            df,
            signal_fn,
            oos_frac=args.oos_frac,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            costs=costs,
            strategy_name=name,
            symbol=symbol,
            params=strat.params,
            curve_fit_flags=strat.curve_fit_flags,
        )
        report = {
            "mode": "holdout",
            "is": is_m,
            "oos": oos_m,
            "assessment": overfitting_assessment(is_m, oos_m),
            "folds": None,
            "oos_result": oos_res,
            "is_result": is_res,
        }
    else:
        wf = run_walk_forward(
            df,
            signal_fn,
            train_bars=train,
            test_bars=test,
            step_bars=step,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            costs=costs,
            strategy_name=name,
            symbol=symbol,
            params=strat.params,
            curve_fit_flags=strat.curve_fit_flags,
        )
        report = {
            "mode": "walk_forward",
            "is": wf.aggregate_is_metrics,
            "oos": wf.combined_oos_metrics,
            "assessment": overfitting_assessment(wf.aggregate_is_metrics, wf.combined_oos_metrics),
            "folds": folds_summary_frame(wf),
            "oos_result": wf.combined_oos,
            "is_result": None,
        }
    report["describe"] = strat.describe()
    return report


def _run_special(
    name: str,
    dfs: dict[str, pd.DataFrame],
    symbols: list[str],
    args: argparse.Namespace,
    costs: CostModel,
) -> dict[str, Any]:
    """Holdout split for specialized simulators (grid / MM / pairs / VWAP)."""
    if name in ("statistical_arbitrage", "cointegration_pairs"):
        if len(symbols) < 2:
            raise ValueError(f"{name} needs two symbols")
        a, b = load_pair(symbols[0], symbols[1], args.interval, args.days)
        n = len(a)
        cut = int(n * (1 - args.oos_frac))
        sim = simulate_coint_pairs if name == "cointegration_pairs" else simulate_pairs
        is_res = sim(
            a.iloc[:cut].reset_index(drop=True),
            b.iloc[:cut].reset_index(drop=True),
            initial_capital=args.capital,
            costs=costs,
            symbol=f"{symbols[0]}/{symbols[1]}",
        )
        oos_res = sim(
            a.iloc[cut:].reset_index(drop=True),
            b.iloc[cut:].reset_index(drop=True),
            initial_capital=args.capital,
            costs=costs,
            symbol=f"{symbols[0]}/{symbols[1]}",
        )
        symbol = f"{symbols[0]}/{symbols[1]}"
    else:
        symbol = symbols[0]
        df = dfs[symbol]
        n = len(df)
        cut = int(n * (1 - args.oos_frac))
        is_df = df.iloc[:cut].reset_index(drop=True)
        oos_df = df.iloc[cut:].reset_index(drop=True)
        if name == "grid_trading":
            sim = simulate_grid
        elif name == "market_making":
            sim = simulate_market_making
        elif name == "vwap_twap":
            sim = simulate_vwap_twap
        else:
            raise ValueError(f"No special simulator for {name}")
        is_res = sim(is_df, initial_capital=args.capital, costs=costs, symbol=symbol)
        oos_res = sim(oos_df, initial_capital=args.capital, costs=costs, symbol=symbol)

    is_m = compute_metrics(is_res)
    oos_m = compute_metrics(oos_res)
    cls = STRATEGY_REGISTRY[name]
    strat = cls()
    return {
        "mode": "holdout_special",
        "is": is_m,
        "oos": oos_m,
        "assessment": overfitting_assessment(is_m, oos_m),
        "folds": None,
        "oos_result": oos_res,
        "is_result": is_res,
        "describe": strat.describe(),
        "symbol": symbol,
    }


def _save_report(
    out_dir: Path,
    name: str,
    symbol: str,
    report: dict[str, Any],
) -> None:
    safe_sym = symbol.replace("/", "-")
    base = out_dir / f"{name}_{safe_sym}"
    base.parent.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    payload = {
        "mode": report["mode"],
        "describe": report["describe"],
        "is": {k: v for k, v in report["is"].items() if k != "params"},
        "oos": {k: v for k, v in report["oos"].items() if k != "params"},
        "params": report["oos"].get("params") or report["is"].get("params"),
        "curve_fit_flags": report["oos"].get("curve_fit_flags")
        or report["is"].get("curve_fit_flags"),
        "assessment": report["assessment"],
    }
    # Make JSON safe
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return obj

    with open(f"{base}_metrics.json", "w") as f:
        json.dump(_sanitize(payload), f, indent=2, default=str)

    text = []
    text.append(report["describe"])
    text.append("")
    text.append(format_metrics(report["is"], title=f"{name} {symbol} — IN SAMPLE / TRAIN"))
    text.append("")
    text.append(format_metrics(report["oos"], title=f"{name} {symbol} — OUT OF SAMPLE"))
    text.append("")
    text.append(report["assessment"])
    with open(f"{base}_report.txt", "w") as f:
        f.write("\n".join(text))

    oos_res = report["oos_result"]
    if oos_res is not None and oos_res.trades:
        trades_to_frame(oos_res.trades).to_csv(f"{base}_oos_trades.csv", index=False)
    if report.get("folds") is not None:
        report["folds"].to_csv(f"{base}_wf_folds.csv", index=False)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = _costs(args)

    names = args.strategies or list(STRATEGY_REGISTRY.keys())
    for n in names:
        if n not in STRATEGY_REGISTRY:
            print(f"Unknown strategy: {n}")
            print("Available:", ", ".join(STRATEGY_REGISTRY))
            return 2

    # Prefetch data
    dfs: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        dfs[sym] = load_or_fetch(
            sym, args.interval, args.days, refresh=args.refresh_data
        )

    scalp_dfs: dict[str, pd.DataFrame] = {}
    if "scalping" in names:
        for sym in args.symbols:
            scalp_dfs[sym] = load_or_fetch(
                sym, args.scalp_interval, args.scalp_days, refresh=args.refresh_data
            )

    summary_rows = []
    print("\n" + "=" * 72)
    print("strategy_lab — walk-forward / OOS backtests")
    print(f"symbols={args.symbols} interval={args.interval} days={args.days}")
    print(
        f"costs: commission={costs.commission_rate} half_spread={costs.half_spread} "
        f"slippage={costs.slippage} (RT≈{costs.round_trip_friction()*100:.2f}%)"
    )
    print("=" * 72)

    for name in names:
        print(f"\n--- {name} ---")
        if name in SPECIAL_SIM:
            try:
                report = _run_special(name, dfs, args.symbols, args, costs)
            except Exception as e:
                print(f"FAILED {name}: {e}")
                continue
            symbol = report.get("symbol", args.symbols[0])
            _save_report(out_dir, name, symbol, report)
            print(format_metrics(report["oos"], title=f"OOS {name} {symbol}"))
            print(report["assessment"])
            summary_rows.append(
                {
                    "strategy": name,
                    "symbol": symbol,
                    "mode": report["mode"],
                    **{f"oos_{k}": report["oos"][k] for k in (
                        "total_trades", "win_rate", "expectancy_R",
                        "max_drawdown", "sharpe_ratio", "profit_factor", "total_return",
                    )},
                }
            )
            continue

        target_symbols = args.symbols
        for sym in target_symbols:
            df = scalp_dfs[sym] if name == "scalping" else dfs[sym]
            # Temporarily override interval scaling for scalp via args copy
            try:
                report = _run_signal_strategy(name, df, sym, args, costs)
            except Exception as e:
                print(f"FAILED {name} {sym}: {e}")
                continue
            _save_report(out_dir, name, sym, report)
            print(format_metrics(report["oos"], title=f"OOS {name} {sym}"))
            print(report["assessment"])
            summary_rows.append(
                {
                    "strategy": name,
                    "symbol": sym,
                    "mode": report["mode"],
                    **{f"oos_{k}": report["oos"][k] for k in (
                        "total_trades", "win_rate", "expectancy_R",
                        "max_drawdown", "sharpe_ratio", "profit_factor", "total_return",
                    )},
                }
            )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        # Rank by OOS expectancy_R then Sharpe for side-by-side comparison
        summary = summary.sort_values(
            by=["oos_expectancy_R", "oos_sharpe_ratio"], ascending=False
        ).reset_index(drop=True)
        summary.insert(0, "rank", range(1, len(summary) + 1))
        summary_path = out_dir / "summary_comparison.csv"
        summary.to_csv(summary_path, index=False)
        print("\n" + "=" * 72)
        print("SUMMARY (OOS) — ranked by expectancy_R, then Sharpe")
        print(summary.to_string(index=False))
        print(f"\nWrote {summary_path}")
        print(f"Per-strategy reports in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
