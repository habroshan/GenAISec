from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 52, 62, 72, 82]
PROFILE = "balanced"

BASE_SCRIPT = Path("train_phase3_model_comparison_leakage_safe.py")
BASE_OUTPUT = Path("data/multiseed_leakage_safe")

MODEL_ORDER = [
    "prompt_lr_text",
    "cumulative_lr_text",
    "cumulative_lr_text_numeric",
    "gru_sequence",
    "bilstm_sequence",
    "transformer_sequence",
]

DISPLAY = {
    "prompt_lr_text": "Prompt LR",
    "cumulative_lr_text": "Cumulative LR",
    "cumulative_lr_text_numeric": "Cumulative LR + Numeric",
    "gru_sequence": "GRU",
    "bilstm_sequence": "BiLSTM",
    "transformer_sequence": "Transformer",
}


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    if n < 2:
        return mean, sd, np.nan, np.nan
    sem = stats.sem(values)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, sd, lo, hi


def patch_script_for_seed(seed: int, run_dir: Path) -> Path:
    text = BASE_SCRIPT.read_text(encoding="utf-8")

                                                                          
    original = 'output_dir = "data/phase3_model_comparison_leakage_safe"'
    replacement = f'output_dir = r"{run_dir.as_posix()}"'
    if original not in text:
        raise RuntimeError(
            "Could not find the expected output_dir assignment in "
            "train_phase3_model_comparison_leakage_safe.py"
        )
    text = text.replace(original, replacement)

    tmp_script = BASE_OUTPUT / f"_tmp_phase3_seed_{seed}.py"
    tmp_script.write_text(text, encoding="utf-8")
    return tmp_script


def run_seed(seed: int):
    run_dir = BASE_OUTPUT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tmp_script = patch_script_for_seed(seed, run_dir)

    cmd = [
        sys.executable,
        str(tmp_script),
        "--profile", PROFILE,
        "--seed", str(seed),
    ]

    
    print(f"Running seed {seed}")
    

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {result.returncode}")

                                                                                   
    candidates = [
        run_dir / "phase3_model_comparison_leakage_safe_summary.json",
        run_dir / "phase3_model_comparison_summary.json",
    ]
    summary = next((p for p in candidates if p.exists()), None)
    if summary is None:
        raise FileNotFoundError(
            f"No summary JSON found for seed {seed} in {run_dir}"
        )

    return summary


def load_seed_metrics(seed: int, summary_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

                                                   
    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected summary format for seed {seed}: expected a list."
        )

    by_model = {row["model"]: row for row in data}
    rows = []

    for model in MODEL_ORDER:
        if model not in by_model:
            raise KeyError(f"Model {model} missing from seed {seed} summary.")
        m = by_model[model]
        rows.append({
            "seed": seed,
            "model": model,
            "display_name": DISPLAY[model],
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "roc_auc": float(m["roc_auc"]),
        })

    return rows


def paired_tests(df, metric: str, reference_model="cumulative_lr_text_numeric"):
    out = []

    ref = (
        df[df["model"] == reference_model]
        .sort_values("seed")[metric]
        .to_numpy()
    )

    for model in MODEL_ORDER:
        if model == reference_model:
            continue

        vals = (
            df[df["model"] == model]
            .sort_values("seed")[metric]
            .to_numpy()
        )

        if len(vals) != len(ref):
            raise ValueError("Seed alignment mismatch in paired comparison.")

        t_stat, t_p = stats.ttest_rel(ref, vals)

        try:
            w_stat, w_p = stats.wilcoxon(ref, vals, zero_method="wilcox")
            w_stat, w_p = float(w_stat), float(w_p)
        except Exception:
            w_stat, w_p = None, None

        out.append({
            "metric": metric,
            "reference_model": reference_model,
            "comparison_model": model,
            "mean_difference_reference_minus_comparison": float(np.mean(ref - vals)),
            "paired_t_statistic": float(t_stat),
            "paired_t_pvalue": float(t_p),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_p,
        })

    return out


def main():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"{BASE_SCRIPT} not found. Put this wrapper in the same folder "
            "as train_phase3_model_comparison_leakage_safe.py."
        )

    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for seed in SEEDS:
        summary = run_seed(seed)
        all_rows.extend(load_seed_metrics(seed, summary))

    df = pd.DataFrame(all_rows)

    raw_csv = BASE_OUTPUT / "multiseed_raw_metrics.csv"
    df.to_csv(raw_csv, index=False)

    summary_rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]

        f1_mean, f1_sd, f1_lo, f1_hi = ci95(sub["f1"].values)
        auc_mean, auc_sd, auc_lo, auc_hi = ci95(sub["roc_auc"].values)

        summary_rows.append({
            "model": model,
            "display_name": DISPLAY[model],
            "n_seeds": len(sub),
            "f1_mean": f1_mean,
            "f1_sd": f1_sd,
            "f1_ci95_low": f1_lo,
            "f1_ci95_high": f1_hi,
            "roc_auc_mean": auc_mean,
            "roc_auc_sd": auc_sd,
            "roc_auc_ci95_low": auc_lo,
            "roc_auc_ci95_high": auc_hi,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = BASE_OUTPUT / "multiseed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    tests = (
        paired_tests(df, "f1")
        + paired_tests(df, "roc_auc")
    )

    stats_json = BASE_OUTPUT / "multiseed_statistics.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": SEEDS,
            "profile": PROFILE,
            "reference_model_for_tests": "cumulative_lr_text_numeric",
            "summary": summary_rows,
            "paired_tests": tests,
        }, f, indent=2)

    
    print("Results")
    

    for _, r in summary_df.iterrows():
        print(
            f"{r['display_name']}: "
            f"F1={r['f1_mean']:.4f} ± {r['f1_sd']:.4f} "
            f"(95% CI {r['f1_ci95_low']:.4f}-{r['f1_ci95_high']:.4f}); "
            f"AUC={r['roc_auc_mean']:.4f} ± {r['roc_auc_sd']:.4f} "
            f"(95% CI {r['roc_auc_ci95_low']:.4f}-{r['roc_auc_ci95_high']:.4f})"
        )

    print("\nPaired tests use Cumulative LR + Numeric as the reference.")
    for t in tests:
        print(
            f"{t['metric']} | ref vs {DISPLAY[t['comparison_model']]}: "
            f"paired-t p={t['paired_t_pvalue']:.6g}, "
            f"Wilcoxon p={t['wilcoxon_pvalue']}"
        )

    print(f"\nSaved raw metrics: {raw_csv}")
    print(f"Saved summary:     {summary_csv}")
    print(f"Saved statistics:  {stats_json}")


if __name__ == "__main__":
    main()
