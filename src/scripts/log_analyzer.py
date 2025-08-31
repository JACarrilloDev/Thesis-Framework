#!/usr/bin/env python3
"""
Log analyzer for RL training metrics.

Features:
- Load and merge logs/training_metrics.csv and optional .bak CSV.
- Fill missing columns per RLTrainer._csv_header.
- Compute rolling trends/deltas (reward, episode_len, hits, KL, entropy, losses, throughput).
- Detect potential curriculum/phase shifts (regime changes) and annotate.
- Auto-split multiple runs inside concatenated CSV/BAK when counters reset.
- Output:
  - Terminal summary (last N iters + overall).
  - Plots (PNG) saved to logs/analysis/run_*/.
  - Markdown summary with key findings.

Usage examples:
  python src/scripts/log_analyzer.py --csv logs/training_metrics.csv --bak logs/training_metrics.csv.bak --split-runs
  python src/scripts/log_analyzer.py --split-runs --run-index -1 --curriculum-boundaries 400,900
  python src/scripts/log_analyzer.py --split-runs --run-index -1 --no-plots
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib is optional at runtime if --no-plots
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


CSV_HEADER = [
    # Keep in sync with RLTrainer._csv_header
    "training_iteration",
    "timesteps_total",
    "timesteps_this_iter",
    "episodes_total",
    "episodes_this_iter",
    "episode_len_mean",
    "episode_reward_mean",
    "episode_reward_min",
    "episode_reward_max",
    "success_mean",
    "success_phase1_mean",
    "success_qtr_mean",
    "full_success_mean",
    "phase2_started_mean",
    "nav_hits_mean_mean",
    "nav_hits_r0_mean",
    "nav_hits_r1_mean",
    "dual_stagnation_steps_mean",
    "entropy",
    "policy_loss",
    "vf_loss",
    "vf_explained_var",
    "kl",
    "cur_kl_coeff",
    "entropy_coeff_used",
    "iter_time_s",
    "wall_time_s_total",
]

DERIVED_COLS = [
    "reward_range",
    "throughput_ts_per_s",
    "steps_per_episode_est",
    "nav_hits_sum",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace({0: np.nan})


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce known numeric columns
    for col in CSV_HEADER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CSV_HEADER:
        if col not in df.columns:
            df[col] = np.nan
    # Derived metrics
    df["reward_range"] = df["episode_reward_max"] - df["episode_reward_min"]
    df["throughput_ts_per_s"] = _safe_div(df["timesteps_this_iter"], df["iter_time_s"])
    df["steps_per_episode_est"] = _safe_div(df["timesteps_this_iter"], df["episodes_this_iter"])
    # Combine nav hits if present
    hits_cols = [c for c in ["nav_hits_mean_mean", "nav_hits_r0_mean", "nav_hits_r1_mean"] if c in df.columns]
    if hits_cols:
        df["nav_hits_sum"] = df[hits_cols].sum(axis=1, min_count=1)
    else:
        df["nav_hits_sum"] = np.nan
    return df


def merge_runs(current_csv: str, bak_csv: Optional[str]) -> pd.DataFrame:
    parts = []
    if bak_csv and os.path.exists(bak_csv):
        parts.append(load_csv(bak_csv))
    parts.append(load_csv(current_csv))
    # Keep original file order; do NOT drop duplicate iterations yet (we may have multiple runs)
    df = pd.concat(parts, ignore_index=True, sort=False)
    df = ensure_columns(df)
    df = df.reset_index(drop=True)
    return df


def assign_run_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect separate runs in a concatenated CSV stream when counters reset:
    - training_iteration decreases
    - timesteps_total decreases
    - wall_time_s_total decreases
    """
    run_id = 0
    run_ids = [run_id]
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        ti_reset = (pd.notna(prev["training_iteration"]) and pd.notna(cur["training_iteration"])
                    and int(cur["training_iteration"]) < int(prev["training_iteration"]))
        ts_reset = (pd.notna(prev["timesteps_total"]) and pd.notna(cur["timesteps_total"])
                    and float(cur["timesteps_total"]) < float(prev["timesteps_total"]))
        wt_reset = (pd.notna(prev["wall_time_s_total"]) and pd.notna(cur["wall_time_s_total"])
                    and float(cur["wall_time_s_total"]) < float(prev["wall_time_s_total"]))
        if ti_reset or ts_reset or wt_reset:
            run_id += 1
        run_ids.append(run_id)

    out = df.copy()
    out["run_id"] = run_ids
    # Sort within runs to make plots consistent
    out = out.sort_values(["run_id", "training_iteration"], kind="mergesort").reset_index(drop=True)
    return out


def rolling_metrics(df: pd.DataFrame, window: int) -> pd.DataFrame:
    r = df.copy()
    for col in [
        "episode_reward_mean", "episode_len_mean", "reward_range",
        "kl", "entropy", "policy_loss", "vf_loss", "vf_explained_var",
        "throughput_ts_per_s", "nav_hits_sum",
    ]:
        if col in r.columns:
            r[f"{col}_roll{window}"] = r[col].rolling(window, min_periods=max(3, window // 3)).mean()
            r[f"{col}_delta"] = r[col].diff()
    return r


def detect_phase_shifts(
    df: pd.DataFrame,
    window: int,
    curriculum_boundaries: Optional[List[int]] = None
) -> List[Tuple[int, str]]:
    """
    Heuristic phase/regime change detector:
    - Significant relative change in episode_len_mean or reward_range over a short span.
    - Introduction of nav_hits_* (first non-null or first > 0).
    - Large change in episodes_this_iter (e.g., batch sizing changes).
    - Optional explicit curriculum boundaries.
    """
    shifts: List[Tuple[int, str]] = []
    w = max(5, window // 2)
    ep = df["episode_len_mean"].rolling(w, min_periods=3).mean()
    rr = df["reward_range"].rolling(w, min_periods=3).mean()

    for i in range(w, len(df)):
        ti = int(df.iloc[i]["training_iteration"])

        def rel_jump(series, i, back=10, thr=0.12):
            j = max(0, i - back)
            prev = series.iloc[j]
            cur = series.iloc[i]
            if not np.isnan(prev) and not np.isnan(cur) and prev != 0:
                return abs(cur - prev) / (abs(prev) + 1e-9) >= thr
            return False

        if rel_jump(ep, i) and rel_jump(rr, i, thr=0.15):
            shifts.append((ti, "Regime change: episode length & reward range shift"))

        if i > 0:
            prev_epi = df.iloc[i - 1]["episodes_this_iter"]
            epi = df.iloc[i]["episodes_this_iter"]
            if not (pd.isna(prev_epi) or pd.isna(epi)) and epi != prev_epi:
                shifts.append((ti, f"Episodes/iter changed {int(prev_epi)} -> {int(epi)}"))

    if "nav_hits_sum" in df.columns:
        mask_any = df["nav_hits_sum"].notna()
        if mask_any.any():
            pos = int(np.flatnonzero(mask_any.values)[0])
            ti = int(df["training_iteration"].iloc[pos])
            shifts.append((ti, "Navigation hits metrics start"))
        gt0 = (df["nav_hits_sum"].fillna(0) > 0)
        if gt0.any():
            pos = int(np.flatnonzero(gt0.values)[0])
            ti = int(df["training_iteration"].iloc[pos])
            shifts.append((ti, "Navigation hits > 0"))

    if "phase2_started_mean" in df.columns:
        m2 = (df["phase2_started_mean"].fillna(0) > 0)
        if m2.any():
            pos = int(np.flatnonzero(m2.values)[0])
            ti = int(df["training_iteration"].iloc[pos])
            shifts.append((ti, "Phase-2 started"))

    if curriculum_boundaries:
        for b in curriculum_boundaries:
            try:
                shifts.append((int(b), "Curriculum boundary"))
            except Exception:
                pass

    # Deduplicate
    seen = set()
    dedup = []
    for ti, msg in shifts:
        if (ti, msg) not in seen:
            dedup.append((ti, msg))
            seen.add((ti, msg))
    return dedup


def summarize_slice(df: pd.DataFrame, lo: int, hi: int) -> Dict[str, float]:
    s = df[(df["training_iteration"] >= lo) & (df["training_iteration"] <= hi)]
    out: Dict[str, float] = {}

    def put(name, val):
        if isinstance(val, (float, int)) and (not np.isnan(val)):
            out[name] = float(val)

    if len(s) == 0:
        return out

    put("iters", len(s))
    put("episode_len_mean", s["episode_len_mean"].mean())
    put("episode_reward_mean", s["episode_reward_mean"].mean())
    put("reward_range_mean", s["reward_range"].mean())
    put("kl_mean", s["kl"].mean())
    put("entropy_mean", s["entropy"].mean())
    put("policy_loss_mean", s["policy_loss"].mean())
    put("vf_loss_mean", s["vf_loss"].mean())
    put("vf_explained_var_mean", s["vf_explained_var"].mean())
    put("throughput_ts_per_s_mean", s["throughput_ts_per_s"].mean())
    # Added success-related summaries
    put("success_mean", s["success_mean"].mean())
    put("success_phase1_mean", s["success_phase1_mean"].mean())
    put("success_qtr_mean", s["success_qtr_mean"].mean())
    put("full_success_mean", s["full_success_mean"].mean())
    put("phase2_started_mean", s["phase2_started_mean"].mean())
    if "nav_hits_sum" in s.columns:
        put("nav_hits_sum_mean", s["nav_hits_sum"].mean())
    return out


def print_summary(df: pd.DataFrame, tail_n: int = 200):
    lo = int(df["training_iteration"].min())
    hi = int(df["training_iteration"].max())
    tail_lo = max(lo, hi - tail_n + 1)

    full = summarize_slice(df, lo, hi)
    tail = summarize_slice(df, tail_lo, hi)

    def fmt(k, v):
        return f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"

    print("== Overall ==")
    for k, v in full.items():
        print(" -", fmt(k, v))

    print(f"\n== Last {tail_n} iters ({tail_lo}-{hi}) ==")
    for k, v in tail.items():
        print(" -", fmt(k, v))

    if "nav_hits_sum" in df.columns:
        # Use positional indices to avoid label/iloc mismatch
        mask_any = df["nav_hits_sum"].notna()
        if mask_any.any():
            pos = int(np.flatnonzero(mask_any.values)[0])
            ti = int(df["training_iteration"].iloc[pos])
            print(f"\nFirst available navigation hits metrics at iteration: {ti}")
        gt0 = (df["nav_hits_sum"].fillna(0) > 0)
        if gt0.any():
            pos = int(np.flatnonzero(gt0.values)[0])
            ti = int(df["training_iteration"].iloc[pos])
            print(f"First nav_hits_sum > 0 at iteration: {ti}")

def plot_series(df: pd.DataFrame, outdir: str, shifts: List[Tuple[int, str]], window: int):
    if not HAS_PLT:
        print("matplotlib not available; skipping plots.")
        return

    os.makedirs(outdir, exist_ok=True)
    x = df["training_iteration"]

    def has_data(col: str) -> bool:
        return col in df.columns and pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().any()

    def annotate(ax):
        for ti, msg in shifts:
            ax.axvline(ti, color="gray", linestyle="--", alpha=0.5)
            ax.text(ti, ax.get_ylim()[1], msg, rotation=90, va="top", ha="right", fontsize=8, alpha=0.7)

    # Reward and episode length
    plt.figure(figsize=(11, 6))
    if has_data("episode_reward_mean"):
        plt.plot(x, df["episode_reward_mean"], label="reward_mean", alpha=0.7)
        roll_col = f"episode_reward_mean_roll{window}"
        if roll_col in df.columns and has_data(roll_col):
            plt.plot(x, df[roll_col], label=roll_col, linewidth=2)
    if has_data("episode_len_mean"):
        plt.plot(x, df["episode_len_mean"], label="episode_len_mean", alpha=0.6)
    annotate(plt.gca())
    plt.legend(); plt.title("Rewards and Episode Length"); plt.xlabel("iteration"); plt.ylabel("value")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "reward_and_len.png")); plt.close()

    # Diagnostics: losses, KL, entropy, explained var
    plt.figure(figsize=(11, 6))
    plotted = False
    for col in ["kl", "entropy", "policy_loss", "vf_loss", "vf_explained_var"]:
        if has_data(col):
            plt.plot(x, df[col], label=col, alpha=0.7); plotted = True
    if plotted:
        annotate(plt.gca())
        plt.legend(); plt.title("Diagnostics"); plt.xlabel("iteration"); plt.ylabel("value")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "diagnostics.png"))
    plt.close()

    # Throughput
    if "throughput_ts_per_s" in df.columns and has_data("throughput_ts_per_s"):
        plt.figure(figsize=(11, 5))
        plt.plot(x, df["throughput_ts_per_s"], label="ts/s", alpha=0.8)
        annotate(plt.gca())
        plt.legend(); plt.title("Throughput (timesteps/s)"); plt.xlabel("iteration"); plt.ylabel("ts/s")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "throughput.png")); plt.close()

    # Hits (nav collisions)
    if "nav_hits_sum" in df.columns and has_data("nav_hits_sum"):
        plt.figure(figsize=(11, 5))
        plt.plot(x, df["nav_hits_sum"], label="nav_hits_sum", alpha=0.8)
        annotate(plt.gca())
        plt.legend(); plt.title("Navigation Hits"); plt.xlabel("iteration"); plt.ylabel("hits")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "nav_hits.png")); plt.close()

    for col in ["full_success_mean", "success_mean"]:
        if has_data(col):
            plt.figure(figsize=(11, 4))
            plt.plot(x, 100.0 * df[col], label=f"{col} (% of episodes)", color="tab:green")
            annotate(plt.gca())
            plt.ylim(-2, 102)
            plt.legend(); plt.title("Success Rate per Iteration"); plt.xlabel("iteration"); plt.ylabel("%")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "success_rate.png")); plt.close()

        # Success rates (add phase-1 and quarter metrics if present)
    for col, color in [
        ("full_success_mean", "tab:green"),
        ("success_mean", "tab:olive"),
        ("success_phase1_mean", "tab:blue"),
        ("success_qtr_mean", "tab:orange"),
    ]:
        if has_data(col):
            plt.figure(figsize=(11, 4))
            plt.plot(x, 100.0 * df[col], label=f"{col} (% episodes)", color=color, alpha=0.85)
            roll_col = f"{col}_roll{window}"
            if roll_col in df.columns and has_data(roll_col):
                plt.plot(x, 100.0 * df[roll_col], label=f"{roll_col} (%)", linewidth=2, color=color, alpha=0.5)
            annotate(plt.gca())
            plt.ylim(-2, 102)
            plt.legend(); plt.title("Success Rates per Iteration"); plt.xlabel("iteration"); plt.ylabel("%")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{col}.png")); plt.close()

def _col_or_jsonl(df: pd.DataFrame, name: str) -> Optional[str]:
    if name in df.columns:
        return name
    alt = f"{name}_jsonl"
    return alt if alt in df.columns else None

def plot_episode_scatter(df: pd.DataFrame, outdir: str):
    if not HAS_PLT:
        return
    rewards_col = _col_or_jsonl(df, "hist_episode_reward")
    if not rewards_col:
        return  # no per-episode histograms available
    success_map = None
    # Optional per-episode success arrays inside a dict column
    if "hist_custom_metrics" in df.columns or "hist_custom_metrics_jsonl" in df.columns:
        cm_col = _col_or_jsonl(df, "hist_custom_metrics")
        # pick one key if available
        if cm_col:
            # prefer "full_success" then "success"
            def pick_key(dct):
                for k in ["full_success", "success", "success_r0", "success_r1"]:
                    if k in dct:
                        return k
                return None
            # find a key on any row
            for d in df[cm_col].dropna():
                key = pick_key(d)
                if key:
                    success_map = key
                    break

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    rng = np.random.default_rng(0)

    X, Y_succ, Y_fail = [], [], []
    for _, row in df.iterrows():
        ti = row.get("training_iteration")
        lst = row.get(rewards_col)
        if isinstance(lst, list) and len(lst) > 0 and pd.notna(ti):
            xvals = (ti + rng.uniform(-0.35, 0.35, size=len(lst)))
            if success_map and isinstance(row.get(_col_or_jsonl(df, "hist_custom_metrics")), dict):
                succ_list = row[_col_or_jsonl(df, "hist_custom_metrics")].get(success_map)
                if isinstance(succ_list, list) and len(succ_list) == len(lst):
                    succ_mask = (np.array(succ_list, dtype=float) > 0.5)
                    plt.scatter(xvals[~succ_mask], np.array(lst)[~succ_mask], s=10, alpha=0.35, label="_nolegend_", color="tab:orange")
                    plt.scatter(xvals[succ_mask], np.array(lst)[succ_mask], s=12, alpha=0.6, label="_nolegend_", color="tab:green")
                    continue
            # fallback: no success flags, plot all as neutral
            plt.scatter(xvals, lst, s=10, alpha=0.35, color="tab:blue")

    plt.title("Per-Iteration Episode Rewards (scatter)")
    plt.xlabel("training_iteration")
    plt.ylabel("episode_reward")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "episodes_scatter.png"))
    plt.close()

def write_markdown_report(df: pd.DataFrame, outdir: str, shifts: List[Tuple[int, str]], window: int, tail_n: int = 200):
    os.makedirs(outdir, exist_ok=True)
    lo = int(df["training_iteration"].min())
    hi = int(df["training_iteration"].max())
    tail_lo = max(lo, hi - tail_n + 1)
    full = summarize_slice(df, lo, hi)
    tail = summarize_slice(df, tail_lo, hi)

    lines: List[str] = []
    lines.append("# RL Training Log Analysis")
    lines.append("")
    lines.append(f"- Iteration range: {lo} - {hi}")
    lines.append(f"- Rolling window: {window}")
    lines.append("")
    lines.append("## Detected curriculum/phase shifts")
    if shifts:
        for ti, msg in shifts:
            lines.append(f"- Iter {ti}: {msg}")
    else:
        lines.append("- None detected by heuristics")

    def kv_table(title: str, d: Dict[str, float]):
        lines.append("")
        lines.append(f"## {title}")
        for k, v in d.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

    kv_table("Overall", full)
    kv_table(f"Last {tail_n} iterations ({tail_lo}-{hi})", tail)

    lines.append("")
    lines.append("## Notes")
    lines.append("- success_* metrics may be 0.0 due to the 2-phase cycle; focus on reward, episode length, hits, and diagnostics.")
    if "nav_hits_sum" in df.columns:
        mask_any = df["nav_hits_sum"].notna().values
        if mask_any.any():
            pos = int(np.flatnonzero(mask_any)[0])
            ti = int(df["training_iteration"].iloc[pos])
            lines.append(f"- Navigation hits appear starting near iter {ti}; earlier iterations show NaN for hits.")
    lines.append("- Drops after a curriculum shift are expected; look for rolling averages to recover.")

    lines.append("")
    lines.append("## Plots")
    lines.append("## Plots")
    for name in ["reward_and_len.png", "diagnostics.png", "throughput.png", "nav_hits.png", "success_rate.png", "episodes_scatter.png"]:
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            lines.append(f"- {name}")

    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    def _extract_learner_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        learner = result.get("info", {}).get("learner", {}) or {}
        # Try common keys; fallback to first policy present
        for key in ["shared_policy", "default_policy"]:
            if key in learner:
                return learner[key].get("learner_stats", {}) or {}
        if learner:
            first = next(iter(learner.values()))
            return first.get("learner_stats", {}) or {}
        return {}


def maybe_load_jsonl(pattern: Optional[str]) -> Optional[pd.DataFrame]:
    if not pattern:
        return None
    files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    rows = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)

                        # Prefer sampler_results.* (present in RLlib), fallback to top-level keys
                        sr = (j.get("sampler_results") or {})
                        hs = (sr.get("hist_stats") or (j.get("hist_stats") or {}))
                        cm = (sr.get("custom_metrics") or (j.get("custom_metrics") or {}))
                        info = (j.get("info") or {})
                        learner_all = (info.get("learner") or {})
                        # policy key may be "default_policy" or your named policy "shared_policy"
                        pol_key = next(iter(learner_all.keys()), None)
                        learner_stats = (learner_all.get(pol_key, {}) or {}).get("learner_stats", {}) if pol_key else {}

                        row = {
                            "training_iteration": j.get("training_iteration"),
                            # basic means (often redundant with CSV, but used when CSV missing)
                            "episode_reward_mean": sr.get("episode_reward_mean", j.get("episode_reward_mean")),
                            "episode_len_mean": sr.get("episode_len_mean", j.get("episode_len_mean")),
                            "episodes_this_iter": sr.get("episodes_this_iter", j.get("episodes_this_iter")),
                            "timesteps_total": j.get("timesteps_total"),
                            "timesteps_this_iter": j.get("num_env_steps_sampled_this_iter") or j.get("num_steps_trained_this_iter"),
                            # diagnostic learner stats
                            "kl": learner_stats.get("kl"),
                            "cur_kl_coeff": learner_stats.get("cur_kl_coeff"),
                            "entropy": learner_stats.get("entropy"),
                            "policy_loss": learner_stats.get("policy_loss"),
                            "vf_loss": learner_stats.get("vf_loss"),
                            "vf_explained_var": learner_stats.get("vf_explained_var"),
                            # success/fail rates (means)
                            "success_mean": cm.get("success_mean"),
                            "full_success_mean": cm.get("full_success_mean"),
                            "success_phase1_mean": cm.get("success_phase1_mean"),
                            "success_qtr_mean": cm.get("success_qtr_mean"),
                            "phase2_started_mean": cm.get("phase2_started_mean"),
                            # episode-level histograms (lists) for scatter plots
                            "hist_episode_reward": hs.get("episode_reward"),
                            "hist_episode_lengths": hs.get("episode_lengths"),
                        }
                        # Optional: if you logged per-episode custom metrics and enabled keep_per_episode_custom_metrics,
                        # you can capture them as lists (e.g., "success") to color scatter points by success.
                        # RLlib stores them inside hist_stats under keys like "custom_metrics/<name>".
                        # Try a few common keys:
                        custom_hist = {}
                        for k in ["success", "full_success", "success_r0", "success_r1"]:
                            v = hs.get(f"custom_metrics/{k}")
                            if v is not None:
                                custom_hist[k] = v
                        if custom_hist:
                            row["hist_custom_metrics"] = custom_hist

                        if row["training_iteration"] is not None:
                            rows.append(row)
                    except Exception:
                        continue
        except Exception:
            continue
    if not rows:
        return None
    dfj = pd.DataFrame(rows)
    dfj = dfj.dropna(subset=["training_iteration"])
    dfj["training_iteration"] = pd.to_numeric(dfj["training_iteration"], errors="coerce")
    return dfj

def analyze_one(df_run: pd.DataFrame, outdir: str, window: int, tail_n: int, curriculum_boundaries: Optional[List[int]], no_plots: bool):
    r = rolling_metrics(df_run, window)
    shifts = detect_phase_shifts(r, window, curriculum_boundaries)
    print_summary(r, tail_n=tail_n)
    if not no_plots:
        plot_series(r, outdir, shifts, window)
        # Episode scatter (from JSONL hist_stats)
        try:
            plot_episode_scatter(r, outdir)
        except Exception as e:
            print(f"episode scatter plot failed: {e}")
    write_markdown_report(r, outdir, shifts, window, tail_n=tail_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/training_metrics.csv", help="Path to current CSV")
    ap.add_argument("--bak", default=None, help="Path to previous CSV (optional)")
    ap.add_argument("--merge-bak", action="store_true", help="If set, merge --csv and --bak when --bak exists")
    ap.add_argument("--jsonl-pattern", default=None, help="Glob for jsonl logs to enrich (optional)")
    ap.add_argument("--window", type=int, default=25, help="Rolling window size")
    ap.add_argument("--tail-n", type=int, default=200, help="Tail window used in summaries")
    ap.add_argument("--no-plots", action="store_true", help="Disable plotting")
    ap.add_argument("--outdir", default="logs/analysis", help="Output directory for analysis")
    ap.add_argument("--split-runs", action="store_true", help="Split concatenated logs into runs when counters reset")
    ap.add_argument("--run-index", type=int, default=None, help="Which run to analyze (0-based, -1 = last). Omit to analyze all.")
    ap.add_argument("--curriculum-boundaries", type=str, default=None, help="Comma-separated iteration boundaries to annotate (e.g. '400,900').")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    bak_path = args.bak if (args.merge_bak and args.bak and os.path.exists(args.bak)) else None
    df = merge_runs(args.csv, bak_path)

    # Optionally merge jsonl metrics (left-join by training_iteration)
    dfj = maybe_load_jsonl(args.jsonl_pattern)
    if dfj is not None:
        df = df.merge(dfj, on="training_iteration", how="left", suffixes=("", "_jsonl"))

    # Assign run ids
    if args.split_runs:
        df = assign_run_ids(df)
    else:
        df["run_id"] = 0

    boundaries = None
    if args.curriculum_boundaries:
        boundaries = [int(x.strip()) for x in args.curriculum_boundaries.split(",") if x.strip()]

    run_ids = sorted(df["run_id"].unique())
    if args.run_index is not None:
        idx = args.run_index if args.run_index >= 0 else (len(run_ids) - 1)
        if idx < 0 or idx >= len(run_ids):
            raise SystemExit(f"Invalid --run-index {args.run_index}; available runs: {run_ids}")
        run_ids = [run_ids[idx]]

    # Analyze each selected run
    for rid in run_ids:
        df_run = df[df["run_id"] == rid].copy()
        if df_run.empty:
            continue
        outdir = os.path.join(args.outdir, f"run_{rid}")
        analyze_one(df_run, outdir, args.window, args.tail_n, boundaries, args.no_plots)


if __name__ == "__main__":
    main()