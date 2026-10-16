"""Command-line entrypoints: setup db, train, backtest, walk-forward, ablation, replay,
leader poll. Run with ``python -m app.cli <command>``.

Examples
--------
python -m app.cli setup-db
python -m app.cli poll-leaders --once
python -m app.cli backtest --synthetic
python -m app.cli ablation --synthetic
python -m app.cli walk-forward --synthetic --calibrate
python -m app.cli replay --synthetic
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

from .config import BASE_DIR, get_settings
from .logging import setup_logging


def _settings(args) -> None:
    setup_logging(log_dir=BASE_DIR / "logs")


async def cmd_setup_db(args) -> None:
    from .config import get_settings
    from .db.base import Database

    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings)
    await db.create_all()
    print(f"Database ready at {settings.effective_db_url}")


async def cmd_backtest(args) -> None:
    from .backtest.runner import BacktestRunner, ProbScorer, TransactionCostModel
    from .execution.killswitch import KillSwitch
    from .features.timeframe import get_profile
    from .risk.engine import RiskEngine
    from .risk.limiter import RateLimiter
    from .risk.prop_profiles import load_profile

    settings = get_settings()
    profile = load_profile(settings.prop_profile_path)
    timeframe = (getattr(args, "timeframe", None) or settings.market_timeframe).upper()
    tfp = get_profile(timeframe)
    ohlcv = await _load_ohlcv(settings, args, timeframe=timeframe)
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter())
    costs = TransactionCostModel()
    result = BacktestRunner(
        risk, scorer=ProbScorer(), cost_model=costs, stop_points=tfp.stop_points
    ).run(ohlcv, start_capital=profile.capital)
    _print_backtest(result)


def _print_backtest(result) -> None:
    print("-" * 52)
    print(f"trades:                {result.n_trades}  (win {result.win_rate:.1%})")
    print(f"total return:          {result.total_return_pct:.2f}%")
    print(f"max drawdown:          {result.max_drawdown_pct:.2%}")
    print(f"sharpe (approx):       {result.sharpe:.2f}")
    print(f"profit factor:         {result.profit_factor:.2f}")
    print(f"brier score:           {result.brier_score:.4f}")
    print(f"commission paid:       ${result.total_commission_paid:.2f}")
    print(f"slippage cost:         ${result.total_slippage_paid:.2f}")
    print(f"swap cost:             ${result.total_swap_paid:.2f}")


async def cmd_ablation(args) -> None:
    from .backtest.ablation import AblationEngine
    from .backtest.runner import TransactionCostModel
    from .backtest.walkforward import time_slices
    from .execution.killswitch import KillSwitch
    from .risk.engine import RiskEngine
    from .risk.limiter import RateLimiter
    from .risk.prop_profiles import load_profile

    settings = get_settings()
    profile = load_profile(settings.prop_profile_path)
    ohlcv = await _load_ohlcv(settings, args)
    risk = RiskEngine(profile, KillSwitch(True), RateLimiter())
    costs = TransactionCostModel()
    ablation = AblationEngine(risk, cost_model=costs)

    slices = time_slices(
        ohlcv,
        train_end=settings.train_end,
        val_end=settings.validation_end,
        test_end=settings.test_end,
    )
    df_results = ablation.run_all_ablations(slices)
    print("\n" + "=" * 92)
    print("                      ARIES ABLATION COMPARISON REPORT                      ")
    print("=" * 92)
    print(df_results.to_string(index=False))
    print("=" * 92 + "\n")


async def cmd_walk_forward(args) -> None:
    from .backtest.leakage import run_leakage_report
    from .backtest.walkforward import walk_forward, write_fold_report
    from .features.timeframe import get_profile

    settings = get_settings()
    timeframe = (getattr(args, "timeframe", None) or settings.market_timeframe).upper()
    tfp = get_profile(timeframe)
    ohlcv = await _load_ohlcv(settings, args, timeframe=timeframe)

    leakage = run_leakage_report(ohlcv, settings.test_start)
    print(f"leakage report: causal={leakage.features_causal} kronos_overlap={leakage.kronos_test_overlap}")
    for note in leakage.notes:
        print("  -", note)

    out_dir = BASE_DIR / "data" / "models"
    fold_name = "wf_main" if timeframe == "M15" else f"wf_main_{timeframe.lower()}"
    print(f"timeframe={timeframe} horizon_bars={tfp.horizon_bars} "
          f"label_threshold={tfp.label_threshold} min_move_pct={tfp.min_move_pct} -> {fold_name}")
    folds = walk_forward(
        ohlcv,
        train_end=settings.train_end,
        val_end=settings.validation_end,
        test_end=settings.test_end,
        horizon_bars=tfp.horizon_bars,
        min_move_pct=tfp.min_move_pct,
        threshold=tfp.label_threshold,
        fold_name=fold_name,
        out_dir=out_dir,
    )
    report_name = "wf_report.json" if timeframe == "M15" else f"wf_report_{timeframe.lower()}.json"
    write_fold_report(folds, BASE_DIR / "data" / report_name)
    for fold in folds:
        print(f"\n{fold.fold_name}: winner={fold.winner} logloss={fold.test_logloss:.4f} "
              f"auc={fold.test_auc:.3f} Brier={fold.brier_score:.4f} ECE={fold.ece:.4f}")
        if args.calibrate and fold.calibration.reliability_curve:
            print("\n--- Reliability Curve (Calibration Table) ---")
            print(f"{'Bin':<5} {'Pred Prob':<12} {'True WinRate':<14} {'Sample Count':<12}")
            for b in fold.calibration.reliability_curve:
                print(f"{b.bin_index:<5} {b.mean_predicted_prob:<12.3f} {b.true_fraction:<14.3f} {b.count:<12}")


async def cmd_replay(args) -> None:
    from .backtest.replay import run_replay
    from .execution.base import PaperBroker
    from .execution.executor import Executor
    from .execution.killswitch import KillSwitch
    from .metamodel.predict import MetaModelPredictor
    from .risk.engine import RiskEngine
    from .risk.limiter import RateLimiter
    from .risk.prop_profiles import load_profile
    from .strategy.engine import StrategyEngine

    settings = get_settings()
    ohlcv = await _load_ohlcv(settings, args)
    profile = load_profile(settings.prop_profile_path)
    db = __import__("app.db.base", fromlist=["Database"]).Database(settings)
    await db.create_all()
    kill = KillSwitch(True)
    risk = RiskEngine(profile, kill, RateLimiter())
    predictor = MetaModelPredictor(BASE_DIR / "data" / "models", threshold=settings.meta_model_prob_threshold)
    strategy = StrategyEngine(settings, predictor)
    executor = Executor(settings, risk, PaperBroker(), kill)
    stats = await run_replay(db, strategy, risk, executor, ohlcv)
    print(f"replay: bars={stats.bars_seen} signals={stats.signals_fired} "
          f"filled={stats.orders_filled} rejected={stats.rejected}")


async def cmd_poll_once(args) -> None:
    from .config import get_settings
    from .db.base import Database
    from .leaders.myfxbook import MyfxbookClient
    from .leaders.poll import poll_leader_trades

    settings = get_settings()
    if not settings.myfxbook_email:
        raise SystemExit("MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")
    db = Database(settings)
    await db.create_all()
    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)
    inserted = await poll_leader_trades(settings, db.session_factory, client)
    print(f"polled live-tier leaders; inserted {inserted} events")


async def cmd_scan_leaders(args) -> None:
    from .config import get_settings
    from .db.base import Database
    from .leaders.discovery import LeaderRoster
    from .leaders.myfxbook import MyfxbookClient
    from .leaders.poll import refresh_leader_roster

    settings = get_settings()
    roster = LeaderRoster(
        settings.roster_path,
        max_live=settings.live_leader_pool_size,
        max_watch=settings.max_watch_leaders,
        min_observations=settings.leader_min_observations,
    )

    if args.add:
        ok = roster.add(args.add, name=args.name or "", source="manual", url=args.url or "")
        print(("added" if ok else "already known") + f": {args.add}")
    if args.remove:
        print("removed" if roster.remove(args.remove) else f"not found: {args.remove}")
    if args.live:
        roster.set_live([s.strip() for s in args.live.split(",") if s.strip()])
        print(f"live tier set to: {roster.live_ids()}")

    if args.action == "list":
        _print_roster(settings, roster)
        return

    if not settings.myfxbook_email:
        raise SystemExit("MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not configured")

    db = Database(settings)
    await db.create_all()
    client = MyfxbookClient(settings.myfxbook_email, settings.myfxbook_password)

    if args.action == "scan":
        result = await refresh_leader_roster(settings, client, db.session_factory, apply=args.apply)
        summary = result.get("error") or f"{result['watched']} candidates"
        print(f"watch list scan: {summary}")
        print(f"promoted -> live: {result['promoted']}")
        print(f"demoted -> watch: {result['demoted']}")
        print(result["note"])
        _print_roster(settings, roster)

    elif args.action == "add-check":
        if not args.id:
            raise SystemExit("usage: scan-leaders add-check --id <accountId>")
        stats = await client.get_account_stats(args.id)
        print(f"{stats.account_id}  {stats.name or '(no name)'}  "
              f"gain={stats.gain_pct:.2f}%  dd={stats.drawdown_pct:.2f}%  trades={stats.trades}")
        print("run `scan-leaders --add <id>` to put it on the watch tier")

    elif args.action == "discover":
        from .leaders.screening import discover_leaders

        seeds = [s.strip() for s in (args.seeds or "").split(",") if s.strip()]
        result = await discover_leaders(
            settings, client, roster, seed_ids=seeds, apply=args.apply, rotate=args.rotate
        )
        print(f"screened {result['candidates']} candidates -> "
              f"{len(result['qualified'])} qualified"
              + ("" if result["applied"] else "  (dry run; pass --apply to persist)"))
        for row in sorted(result["results"], key=lambda r: r["composite"], reverse=True):
            flag = "OK " if row["qualified"] else "no "
            reason = "" if row["qualified"] else f"  [{','.join(row['reasons'])}]"
            print(f"  {flag}{row['id']:<14} score={row['composite']:.3f} "
                  f"{row['strategy']:<14} pf={row['metrics'].get('profit_factor')} "
                  f"dd={row['metrics'].get('drawdown_pct')}%{reason}")
        if result["rotation"] is not None:
            print(f"rotation: promoted={result['rotation']['promoted']} "
                  f"demoted={result['rotation']['demoted']}")
            print(result["rotation"]["note"])
        if result["applied"]:
            _print_roster(settings, roster)


def _print_roster(settings, roster) -> None:
    rows = sorted(roster.all_ids(), key=lambda i: (roster.get(i).watch_score, roster.get(i).observations), reverse=True)
    print("-" * 78)
    print(f"{'tier':<6} {'id':<16} {'score':>6} {'obs':>4}  name")
    print("-" * 78)
    for leader_id in rows:
        c = roster.get(leader_id)
        print(f"{c.tier:<6} {leader_id:<16} {c.watch_score:>6.3f} {c.observations:>4}  {c.name or ''}")
    print(f"live={len(roster.live_ids())}/{settings.live_leader_pool_size}  "
          f"watch={len(roster.watch_ids())}/{settings.max_watch_leaders}  file={settings.roster_path}")


async def _load_ohlcv(settings, args, timeframe: str | None = None):
    from datetime import datetime, timezone

    from .data.ohlcv import CSVProvider, OHLCVRequest, SyntheticProvider

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    tf = (timeframe or settings.market_timeframe).upper()
    if args.synthetic:
        provider = SyntheticProvider(seed=args.seed)
    else:
        provider = CSVProvider(str(BASE_DIR / "data" / "raw"))
    df = await provider.get_ohlcv(
        OHLCVRequest(symbol=settings.market_symbol, timeframe=tf, start=start, end=end)
    )
    df.attrs["symbol"] = settings.market_symbol
    df.attrs["timeframe"] = tf
    print(f"loaded {len(df)} bars ({tf})  {df.index[0]} .. {df.index[-1]}")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gold-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-db", help="create the database schema")

    sp = sub.add_parser("poll-leaders", help="poll Myfxbook live-tier leaders once")
    sp.add_argument("--once", action="store_true")

    sp = sub.add_parser("scan-leaders", help="discover, rank and rotate leaders (two-tier)")
    sp.add_argument("action", nargs="?", default="scan", choices=["scan", "list", "add-check", "discover"],
                    help="scan=rank the watch list (default), discover=screen candidates, "
                         "list=show roster, add-check=verify one id")
    sp.add_argument("--apply", action="store_true", help="persist the rotation (default: dry run)")
    sp.add_argument("--add", help="add a candidate account id to the watch tier")
    sp.add_argument("--name", help="name for --add")
    sp.add_argument("--url", help="myfxbook URL for --add")
    sp.add_argument("--remove", help="remove an account id from the roster")
    sp.add_argument("--live", help="comma-separated ids to force as the live tier")
    sp.add_argument("--id", help="account id for add-check")
    sp.add_argument("--seeds", help="comma-separated candidate ids to screen (added to the seed file)")
    sp.add_argument("--rotate", action="store_true", help="rotate the live tier after discovery")

    sp = sub.add_parser("backtest", help="run the hybrid-signal backtest")
    sp.add_argument("--synthetic", action="store_true", help="use deterministic synthetic data")
    sp.add_argument("--days", type=int, default=730, help="lookback window in days")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--timeframe", default=None, help="override market timeframe (M1, M5, M15, ...)")

    sp = sub.add_parser("ablation", help="run 6-way ablation comparison across splits")
    sp.add_argument("--synthetic", action="store_true", help="use synthetic data")
    sp.add_argument("--days", type=int, default=1800, help="lookback window in days")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--timeframe", default=None, help="override market timeframe (M1, M5, M15, ...)")

    sp = sub.add_parser("walk-forward", help="walk-forward train/validate/test")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--calibrate", action="store_true", help="output Brier score & reliability curve")
    sp.add_argument("--days", type=int, default=800)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--timeframe", default=None, help="override market timeframe (M1, M5, M15, ...)")

    sp = sub.add_parser("replay", help="replay historical bars through the live stack")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--days", type=int, default=180)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--timeframe", default=None, help="override market timeframe (M1, M5, M15, ...)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _settings(args)
    if args.command == "setup-db":
        asyncio.run(cmd_setup_db(args))
    elif args.command == "poll-leaders":
        asyncio.run(cmd_poll_once(args))
    elif args.command == "scan-leaders":
        asyncio.run(cmd_scan_leaders(args))
    elif args.command == "backtest":
        asyncio.run(cmd_backtest(args))
    elif args.command == "ablation":
        asyncio.run(cmd_ablation(args))
    elif args.command == "walk-forward":
        asyncio.run(cmd_walk_forward(args))
    elif args.command == "replay":
        asyncio.run(cmd_replay(args))


if __name__ == "__main__":
    main()
