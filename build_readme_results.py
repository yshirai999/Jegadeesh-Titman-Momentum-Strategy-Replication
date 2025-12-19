"""
Aggregate per-strategy summary CSVs into one README results table.

Expected input files:
  results/summary_statistics_{J}{K}{skip}.csv

Each summary_statistics file should include a row with Portfolio == 'WML' and columns:
  - 'Mean Return (%)'
  - 'Volatility (%)'
  - 't-statistic'

This script:
  1) Reads all 32 files
  2) Extracts WML stats
  3) Builds a Markdown table like the README screenshot (grouped by J and K, with Skip=0 and Skip=1)
  4) Inserts/replaces a block in README.md between markers.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


# ----------------------------
# CONFIG
# ----------------------------
# REPO_ROOT = Path(__file__).resolve().parents[1]
# RESULTS_DIR = RESULTS_DIR = REPO_ROOT / "momentum_replication" / "results"
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent  # build_readme_results.py is in repo root (per your screenshot)

print(f"[DEBUG] __file__      = {__file__}")
print(f"[DEBUG] resolved file = {THIS_FILE}")
print(f"[DEBUG] repo root     = {REPO_ROOT}")

# Find a results directory anywhere under repo root that is exactly named "results"
results_dirs = [p for p in REPO_ROOT.rglob("results") if p.is_dir()]

print("[DEBUG] candidate results dirs found:")
for p in results_dirs:
    print(f"  - {p}")

if not results_dirs:
    raise FileNotFoundError(
        f"No 'results' directory found anywhere under: {REPO_ROOT}"
    )

# Prefer momentum_replication/results if present, otherwise take the first found
preferred = REPO_ROOT / "momentum_replication" / "results"
RESULTS_DIR = preferred if preferred in results_dirs else results_dirs[0]

print(f"[DEBUG] using RESULTS_DIR = {RESULTS_DIR}")

README_PATH = REPO_ROOT / "README.md"

FORMATION_PERIODS = [3, 6, 9, 12]
HOLDING_PERIODS = [3, 6, 9, 12]
SKIP_PERIODS = [0, 1]

# Markers in README.md (the script will replace content between them)
BEGIN_MARKER = "<!-- BEGIN: MOMENTUM_RESULTS_TABLE -->"
END_MARKER   = "<!-- END: MOMENTUM_RESULTS_TABLE -->"


# ----------------------------
# HELPERS
# ----------------------------
def parse_j_k_skip_from_filename(filename: str) -> tuple[int, int, int]:
    """
    Parse J, K, skip from filename.

    Supports:
      - summary_statistics_{J}{K}0.csv
      - summary_statistics_{J}{K}1.csv
      - summary_statistics_{J}{K}True.csv
      - summary_statistics_{J}{K}False.csv
    """
    m = re.match(r"^summary_statistics_(\d+)(True|False|0|1)\.csv$", filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")

    jk_part = m.group(1)
    skip_part = m.group(2)

    # Convert skip flag to int {0,1}
    if skip_part in ("1", "True"):
        skip = 1
    elif skip_part in ("0", "False"):
        skip = 0
    else:
        raise ValueError(f"Invalid skip flag in filename: {filename}")

    # Recover J and K by matching known values
    for j in FORMATION_PERIODS:
        for k in HOLDING_PERIODS:
            if jk_part == f"{j}{k}":
                return j, k, skip

    raise ValueError(f"Could not parse (J,K) from filename '{filename}'")



def read_wml_stats(stats_csv_path: Path) -> dict:
    """
    Read a summary_statistics CSV and extract WML metrics.
    """
    df = pd.read_csv(stats_csv_path)

    if "Portfolio" not in df.columns:
        raise ValueError(f"Missing 'Portfolio' column in {stats_csv_path.name}")

    wml = df.loc[df["Portfolio"] == "WML"]
    if wml.empty:
        raise ValueError(f"No 'WML' row found in {stats_csv_path.name}")

    # Extract numeric values
    mean_ret = float(wml["Mean Return (%)"].iloc[0])
    vol = float(wml["Volatility (%)"].iloc[0])
    tstat = str(wml["t-statistic"].iloc[0])

    return {"mean": mean_ret, "vol": vol, "tstat": tstat}


def fmt(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}"


def build_markdown_table(rows: list[dict]) -> str:
    """
    Build a Markdown table matching your README screenshot style:
      Formation (J) | Holding (K) | Skip=0 Return | Skip=0 Vol | Skip=0 t-stat | Skip=1 Return | Skip=1 Vol | Skip=1 t-stat

    'rows' should have dicts with keys: J, K, skip0_mean, skip0_vol, skip0_t, skip1_mean, skip1_vol, skip1_t
    """
    header = (
        "| Formation (J) | Holding (K) | Skip=0 Return(%) | Skip=0 Vol(%) | Skip=0 t-stat | "
        "Skip=1 Return(%) | Skip=1 Vol(%) | Skip=1 t-stat |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    lines = []
    for r in rows:
        lines.append(
            f"| J={r['J']} | K={r['K']} | "
            f"{fmt(r['skip0_mean'])} | {fmt(r['skip0_vol'])} | {r['skip0_t']} | "
            f"{fmt(r['skip1_mean'])} | {fmt(r['skip1_vol'])} | {r['skip1_t']} |"
        )

    return header + "\n".join(lines) + "\n"


def replace_block_in_readme(readme_text: str, new_block: str) -> str:
    """
    Replace content between BEGIN_MARKER and END_MARKER. If markers don't exist, append at end.
    """
    if BEGIN_MARKER in readme_text and END_MARKER in readme_text:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            flags=re.DOTALL
        )
        return pattern.sub(f"{BEGIN_MARKER}\n\n{new_block}\n{END_MARKER}", readme_text)

    # If missing markers, append them
    return readme_text.rstrip() + "\n\n" + BEGIN_MARKER + "\n\n" + new_block + "\n" + END_MARKER + "\n"


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    stats_files = sorted(RESULTS_DIR.glob("summary_statistics_*.csv"))
    if not stats_files:
        raise FileNotFoundError(f"No summary_statistics_*.csv files found in {RESULTS_DIR}")

    # Collect stats by (J,K,skip)
    stats = {}  # (J,K,skip) -> metrics dict
    for p in stats_files:
        j, k, s = parse_j_k_skip_from_filename(p.name)
        stats[(j, k, s)] = read_wml_stats(p)

    # Build final rows for each (J,K) combining skip=0 and skip=1
    out_rows = []
    missing = []
    for j in FORMATION_PERIODS:
        for k in HOLDING_PERIODS:
            key0 = (j, k, 0)
            key1 = (j, k, 1)
            if key0 not in stats or key1 not in stats:
                missing.append((j, k))
                continue

            out_rows.append({
                "J": j,
                "K": k,
                "skip0_mean": stats[key0]["mean"],
                "skip0_vol":  stats[key0]["vol"],
                "skip0_t":    stats[key0]["tstat"],
                "skip1_mean": stats[key1]["mean"],
                "skip1_vol":  stats[key1]["vol"],
                "skip1_t":    stats[key1]["tstat"],
            })

    if missing:
        raise RuntimeError(
            "Missing results for some (J,K) pairs (need both skip=0 and skip=1):\n"
            + "\n".join([f"  J={j}, K={k}" for j, k in missing])
        )

    md_table = build_markdown_table(out_rows)

    # Update README
    readme_text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    updated = replace_block_in_readme(readme_text, md_table)
    README_PATH.write_text(updated, encoding="utf-8")

    print(f"✅ README updated: {README_PATH}")
    print(f"✅ Table built from {len(stats_files)} summary files")


if __name__ == "__main__":
    main()
