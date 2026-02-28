import pandas as pd
df = pd.read_csv("verification/mde_v2_comprehensive_sweep_20260227_072244.csv")
print("Top 10 overall by total_return:")
print(df.sort_values("total_return", ascending=False)[["run", "timeframe", "total_return", "max_drawdown", "total_trades", "score_balanced", "phase"]].head(10).to_string())
print("\nTop 10 overall by score_balanced:")
print(df.sort_values("score_balanced", ascending=False)[["run", "timeframe", "total_return", "max_drawdown", "total_trades", "score_balanced", "phase"]].head(10).to_string())
