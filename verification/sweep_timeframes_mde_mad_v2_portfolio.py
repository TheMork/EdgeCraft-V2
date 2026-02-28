from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = PROJECT_ROOT / "verification" / "backtest_server_mde_mad_v2_compare.py"


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


def run_sweep() -> None:
    timeframes = _parse_list_env("EDGECRAFT_SWEEP_TIMEFRAMES", ["2h", "4h", "8h", "12h", "1d"])
    runs = os.getenv("EDGECRAFT_COMPARE_RUNS", "local_mde_mad_v2_default")
    executions = os.getenv("EDGECRAFT_COMPARE_EXECUTIONS", "legacy_close,strict_next_open")

    rows: List[pd.DataFrame] = []

    print(f"Sweep timeframes: {', '.join(timeframes)}")
    print(f"Runs: {runs}")
    print(f"Executions: {executions}")

    for tf in timeframes:
        env = os.environ.copy()
        env["EDGECRAFT_TIMEFRAME"] = tf
        env["EDGECRAFT_COMPARE_RUNS"] = runs
        env["EDGECRAFT_COMPARE_EXECUTIONS"] = executions

        print(f"\n[{tf}] Running portfolio backtest ...")
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
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"Backtest failed for timeframe={tf} with code {proc.returncode}")

        merged_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        summary_path = _extract_summary_path(merged_out)
        if not summary_path.exists():
            raise RuntimeError(f"Summary file not found: {summary_path}")

        df = pd.read_csv(summary_path)
        df["timeframe"] = tf
        df["sweep_elapsed_sec"] = elapsed
        rows.append(df)

        print(f"[{tf}] done in {elapsed:.1f}s -> {summary_path.name}")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["run", "execution", "timeframe"]).reset_index(drop=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "verification" / f"server_vs_local_mde_mad_v2_timeframe_sweep_{ts}.csv"
    combined.to_csv(out_path, index=False)

    best = combined.sort_values("total_return", ascending=False).head(10)

    print("\nTop rows by total_return:")
    print(best[["run", "execution", "timeframe", "total_return", "final_equity", "max_drawdown", "total_trades"]].to_string(index=False))
    print(f"\nSaved sweep: {out_path}")


if __name__ == "__main__":
    run_sweep()
