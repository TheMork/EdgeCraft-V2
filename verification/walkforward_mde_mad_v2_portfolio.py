from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = PROJECT_ROOT / "verification" / "backtest_server_mde_mad_v2_compare.py"


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _parse_list_env(name: str, default_values: List[str]) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default_values
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or default_values


def _extract_summary_path(output_text: str) -> Path:
    pattern = re.compile(r"^Saved:\s+(verification/server_vs_local_mde_mad_v2_portfolio_summary_[^\s]+\.csv)\s*$", re.MULTILINE)
    m = pattern.search(output_text)
    if not m:
        raise RuntimeError("Could not find summary CSV path in backtest output.")
    return PROJECT_ROOT / m.group(1)


def _build_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[Fold]:
    folds: List[Fold] = []
    fold_start = start
    fold_id = 1

    while True:
        train_end_excl = fold_start + pd.DateOffset(months=train_months)
        test_end_excl = train_end_excl + pd.DateOffset(months=test_months)
        if test_end_excl > end:
            break

        train_end = train_end_excl - pd.Timedelta(seconds=1)
        test_start = train_end_excl
        test_end = test_end_excl - pd.Timedelta(seconds=1)

        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=fold_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        fold_start = fold_start + pd.DateOffset(months=step_months)

    return folds


def _run_period(
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    timeframe: str,
    runs: str,
    executions: str,
) -> tuple[pd.DataFrame, float, str]:
    env = os.environ.copy()
    env["EDGECRAFT_START_DATE"] = period_start.strftime("%Y-%m-%dT%H:%M:%S")
    env["EDGECRAFT_END_DATE"] = period_end.strftime("%Y-%m-%dT%H:%M:%S")
    env["EDGECRAFT_TIMEFRAME"] = timeframe
    env["EDGECRAFT_COMPARE_RUNS"] = runs
    env["EDGECRAFT_COMPARE_EXECUTIONS"] = executions

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(
            "Backtest failed "
            f"({period_start.isoformat()} -> {period_end.isoformat()}) code={proc.returncode}\n{msg}"
        )

    merged_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary_path = _extract_summary_path(merged_out)
    if not summary_path.exists():
        raise RuntimeError(f"Summary file not found: {summary_path}")

    return pd.read_csv(summary_path), elapsed, summary_path.name


def run_walkforward() -> None:
    global_start = pd.Timestamp(os.getenv("EDGECRAFT_WF_START", "2024-02-22T00:00:00"), tz="UTC")
    global_end = pd.Timestamp(os.getenv("EDGECRAFT_WF_END", "2026-02-22T00:00:00"), tz="UTC")

    train_months = int(os.getenv("EDGECRAFT_WF_TRAIN_MONTHS", "6"))
    test_months = int(os.getenv("EDGECRAFT_WF_TEST_MONTHS", "3"))
    step_months = int(os.getenv("EDGECRAFT_WF_STEP_MONTHS", "3"))

    timeframe = os.getenv("EDGECRAFT_WF_TIMEFRAME", "8h")
    runs = os.getenv("EDGECRAFT_COMPARE_RUNS", "local_mde_mad_v2_default")
    executions = os.getenv("EDGECRAFT_COMPARE_EXECUTIONS", "strict_next_open")

    _ = _parse_list_env("EDGECRAFT_COMPARE_RUNS", ["local_mde_mad_v2_default"])
    _ = _parse_list_env("EDGECRAFT_COMPARE_EXECUTIONS", ["strict_next_open"])

    folds = _build_folds(
        start=global_start,
        end=global_end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    if not folds:
        raise RuntimeError("No folds generated. Check date range and month settings.")

    print(
        f"Walk-forward: timeframe={timeframe}, train={train_months}m, test={test_months}m, step={step_months}m, "
        f"folds={len(folds)}"
    )
    print(f"Runs: {runs}")
    print(f"Executions: {executions}")

    all_rows: List[pd.DataFrame] = []

    for fold in folds:
        print(
            f"\nFold {fold.fold_id}: train {fold.train_start.strftime('%Y-%m-%d')} -> {fold.train_end.strftime('%Y-%m-%d')}, "
            f"test {fold.test_start.strftime('%Y-%m-%d')} -> {fold.test_end.strftime('%Y-%m-%d')}"
        )

        train_df, train_elapsed, train_file = _run_period(
            period_start=fold.train_start,
            period_end=fold.train_end,
            timeframe=timeframe,
            runs=runs,
            executions=executions,
        )
        train_df["phase"] = "train"
        train_df["fold_id"] = fold.fold_id
        train_df["fold_train_start"] = fold.train_start.isoformat()
        train_df["fold_train_end"] = fold.train_end.isoformat()
        train_df["fold_test_start"] = fold.test_start.isoformat()
        train_df["fold_test_end"] = fold.test_end.isoformat()
        train_df["elapsed_sec"] = train_elapsed
        train_df["source_summary_file"] = train_file
        all_rows.append(train_df)

        test_df, test_elapsed, test_file = _run_period(
            period_start=fold.test_start,
            period_end=fold.test_end,
            timeframe=timeframe,
            runs=runs,
            executions=executions,
        )
        test_df["phase"] = "test"
        test_df["fold_id"] = fold.fold_id
        test_df["fold_train_start"] = fold.train_start.isoformat()
        test_df["fold_train_end"] = fold.train_end.isoformat()
        test_df["fold_test_start"] = fold.test_start.isoformat()
        test_df["fold_test_end"] = fold.test_end.isoformat()
        test_df["elapsed_sec"] = test_elapsed
        test_df["source_summary_file"] = test_file
        all_rows.append(test_df)

        print(f"  train done in {train_elapsed:.1f}s ({train_file})")
        print(f"  test  done in {test_elapsed:.1f}s ({test_file})")

    detail = pd.concat(all_rows, ignore_index=True)

    test_only = detail[detail["phase"] == "test"].copy()
    grouped = test_only.groupby(["run", "execution", "timeframe"], as_index=False).agg(
        folds=("fold_id", "nunique"),
        mean_test_return=("total_return", "mean"),
        median_test_return=("total_return", "median"),
        worst_test_return=("total_return", "min"),
        best_test_return=("total_return", "max"),
        win_rate_test=("total_return", lambda s: float((s > 0).mean())),
        mean_test_drawdown=("max_drawdown", "mean"),
        worst_test_drawdown=("max_drawdown", "max"),
        mean_test_sharpe=("sharpe_ratio", "mean"),
        mean_test_sortino=("sortino_ratio", "mean"),
        mean_test_trades=("total_trades", "mean"),
    )

    def _compound_return(series: pd.Series) -> float:
        vals = [float(x) for x in series if pd.notna(x)]
        gross = 1.0
        for r in vals:
            gross *= (1.0 + r)
        return gross - 1.0

    comp = (
        test_only.sort_values(["run", "execution", "timeframe", "fold_id"]) 
        .groupby(["run", "execution", "timeframe"]) ["total_return"]
        .apply(_compound_return)
        .reset_index(name="compound_test_return")
    )

    summary = grouped.merge(comp, on=["run", "execution", "timeframe"], how="left")
    summary = summary.sort_values(["compound_test_return", "mean_test_return"], ascending=[False, False]).reset_index(drop=True)

    ts = pd.Timestamp.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / f"walkforward_mde_mad_v2_portfolio_detail_{ts}.csv"
    summary_path = out_dir / f"walkforward_mde_mad_v2_portfolio_summary_{ts}.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nWalk-forward summary (test phase):")
    print(
        summary[
            [
                "run",
                "execution",
                "timeframe",
                "folds",
                "mean_test_return",
                "median_test_return",
                "worst_test_return",
                "win_rate_test",
                "compound_test_return",
                "mean_test_drawdown",
                "worst_test_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    run_walkforward()
