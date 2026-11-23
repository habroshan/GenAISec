
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix
)

CURRENT_TEMPLATES = {
    "spear_phishing": {
        "identity": 0.30,
        "authority": 0.25,
        "business": 0.25,
        "timing": 0.20,
    },
    "invoice_fraud": {
        "vendor": 0.30,
        "financial": 0.30,
        "authority": 0.20,
        "timing": 0.20,
    },
    "access_pretext": {
        "identity": 0.25,
        "technical": 0.30,
        "authority": 0.25,
        "timing": 0.20,
    },
}

EQUAL_WEIGHT_TEMPLATES = {
    name: {ctx: 1.0 / len(weights) for ctx in weights}
    for name, weights in CURRENT_TEMPLATES.items()
}

THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    required = {
        "session_id", "turn_id", "prompt_text", "cumulative_contexts",
        "attack_ready_label"
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    return rows


def score_template(contexts, weights):
    contexts = set(contexts)
    total = float(sum(weights.values()))
    return sum(w for c, w in weights.items() if c in contexts) / total


def label_from_contexts(contexts, templates, threshold):
    scores = {
        name: score_template(contexts, weights)
        for name, weights in templates.items()
    }
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    return int(best_score >= threshold), best_name, best_score


def sensitivity(args):
    rows = load_rows(args.input)
    original = np.array([int(r["attack_ready_label"]) for r in rows], dtype=int)

    configs = [
        ("current_weights", CURRENT_TEMPLATES),
        ("equal_weights", EQUAL_WEIGHT_TEMPLATES),
    ]

    results = []
    for cfg_name, templates in configs:
        for threshold in THRESHOLDS:
            pred = []
            for r in rows:
                y, _, _ = label_from_contexts(
                    r.get("cumulative_contexts", []), templates, threshold
                )
                pred.append(y)
            pred = np.asarray(pred, dtype=int)

            results.append({
                "weight_setting": cfg_name,
                "threshold": threshold,
                "positive_prefix_rate": float(pred.mean()),
                "agreement_with_original": float((pred == original).mean()),
                "cohen_kappa_with_original": float(cohen_kappa_score(original, pred)),
                "num_flipped_prefixes": int((pred != original).sum()),
            })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(outdir / "label_sensitivity.csv", index=False)

                                                                   
    session_results = []
    sessions = defaultdict(list)
    for r in rows:
        sessions[int(r["session_id"])].append(r)

    for cfg_name, templates in configs:
        for threshold in THRESHOLDS:
            ready_sessions = 0
            for _, srows in sessions.items():
                is_ready = False
                for r in srows:
                    y, _, _ = label_from_contexts(
                        r.get("cumulative_contexts", []), templates, threshold
                    )
                    if y == 1:
                        is_ready = True
                        break
                ready_sessions += int(is_ready)
            session_results.append({
                "weight_setting": cfg_name,
                "threshold": threshold,
                "attack_ready_session_rate": ready_sessions / len(sessions),
                "attack_ready_sessions": ready_sessions,
                "num_sessions": len(sessions),
            })

    pd.DataFrame(session_results).to_csv(
        outdir / "session_prevalence_sensitivity.csv", index=False
    )
    print(df.to_string(index=False))
    print(f"\nSaved results to: {outdir.resolve()}")


def build_prefix_text(srows, turn_id):
    ordered = sorted(srows, key=lambda r: int(r["turn_id"]))
    texts = [
        str(r.get("prompt_text", "")).strip()
        for r in ordered
        if int(r["turn_id"]) <= int(turn_id)
    ]
    return "\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(texts))


def prepare_expert(args):
    rows = load_rows(args.input)
    rng = np.random.default_rng(args.seed)

    sessions = defaultdict(list)
    for r in rows:
        sessions[int(r["session_id"])].append(r)

    candidates = []
    for r in rows:
        score = float(r.get("attack_readiness_score", np.nan))
        candidates.append({
            "session_id": int(r["session_id"]),
            "turn_id": int(r["turn_id"]),
            "framework_label": int(r["attack_ready_label"]),
            "framework_score": score,
            "prefix_text": build_prefix_text(
                sessions[int(r["session_id"])], int(r["turn_id"])
            ),
        })

    df = pd.DataFrame(candidates)

                                                                            
    if df["framework_score"].notna().all():
        df["distance_to_threshold"] = (df["framework_score"] - 0.75).abs()
        strata = []
        groups = [
            df[(df.framework_label == 0) & (df.distance_to_threshold <= 0.20)],
            df[(df.framework_label == 1) & (df.distance_to_threshold <= 0.20)],
            df[(df.framework_label == 0) & (df.distance_to_threshold > 0.20)],
            df[(df.framework_label == 1) & (df.distance_to_threshold > 0.20)],
        ]
        per_group = max(1, args.n // len(groups))
        for g in groups:
            if len(g):
                take = min(per_group, len(g))
                strata.append(g.sample(n=take, random_state=args.seed))
        sampled = pd.concat(strata, ignore_index=True)
        if len(sampled) < args.n:
            remaining = df.drop(sampled.index, errors="ignore")
            if len(remaining):
                extra = remaining.sample(
                    n=min(args.n - len(sampled), len(remaining)),
                    random_state=args.seed + 1
                )
                sampled = pd.concat([sampled, extra], ignore_index=True)
    else:
                                                          
        pos = df[df.framework_label == 1]
        neg = df[df.framework_label == 0]
        n_pos = min(args.n // 2, len(pos))
        n_neg = min(args.n - n_pos, len(neg))
        sampled = pd.concat([
            pos.sample(n=n_pos, random_state=args.seed),
            neg.sample(n=n_neg, random_state=args.seed + 1)
        ], ignore_index=True)

    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "case_id", np.arange(1, len(sampled) + 1))
    sampled["expert_1"] = ""
    sampled["expert_2"] = ""
    sampled["expert_3"] = ""
    sampled["expert_notes"] = ""

                                                        
    blinded = sampled[
        ["case_id", "session_id", "turn_id", "prefix_text",
         "expert_1", "expert_2", "expert_3", "expert_notes"]
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    blinded.to_csv(out, index=False)

                                                       
    key = sampled[
        ["case_id", "framework_label", "framework_score"]
    ]
    key.to_csv(out.with_name(out.stem + "_framework_key.csv"), index=False)

    print(f"Prepared {len(blinded)} blinded cases: {out.resolve()}")
    print(f"Framework key: {out.with_name(out.stem + '_framework_key.csv').resolve()}")


def fleiss_kappa_binary(ratings):
                                                                   
    ratings = np.asarray(ratings, dtype=int)
    n_items, n_raters = ratings.shape
    counts = np.zeros((n_items, 2), dtype=float)
    for i in range(n_items):
        counts[i, 0] = np.sum(ratings[i] == 0)
        counts[i, 1] = np.sum(ratings[i] == 1)
    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p = counts.sum(axis=0) / (n_items * n_raters)
    P_e = np.sum(p**2)
    if np.isclose(1 - P_e, 0):
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def analyse_expert(args):
    completed = pd.read_csv(args.input)
    key_path = Path(args.input).with_name(
        Path(args.input).stem.replace("_completed", "") + "_framework_key.csv"
    )
    if not key_path.exists():
        raise FileNotFoundError(
            f"Framework key not found: {key_path}. "
            "Keep the key created by prepare-expert in the same directory."
        )
    key = pd.read_csv(key_path)
    df = completed.merge(key, on="case_id", how="inner", validate="one_to_one")

    expert_cols = [c for c in ["expert_1", "expert_2", "expert_3"] if c in df.columns]
    if len(expert_cols) < 2:
        raise ValueError("At least two expert columns are required.")

    for c in expert_cols:
        if not set(df[c].dropna().astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{c} must contain only 0/1 values.")
    df = df.dropna(subset=expert_cols).copy()
    ratings = df[expert_cols].astype(int).to_numpy()

    majority = (ratings.sum(axis=1) >= (ratings.shape[1] / 2)).astype(int)
    framework = df["framework_label"].astype(int).to_numpy()

    result = {
        "n_cases": int(len(df)),
        "n_experts": int(len(expert_cols)),
        "fleiss_kappa_between_experts": fleiss_kappa_binary(ratings),
        "framework_vs_expert_majority": {
            "accuracy": float(accuracy_score(majority, framework)),
            "precision": float(precision_score(majority, framework, zero_division=0)),
            "recall": float(recall_score(majority, framework, zero_division=0)),
            "f1": float(f1_score(majority, framework, zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(majority, framework)),
            "confusion_matrix": confusion_matrix(majority, framework).tolist(),
        },
        "pairwise_expert_cohen_kappa": {},
    }

    for i in range(len(expert_cols)):
        for j in range(i + 1, len(expert_cols)):
            a, b = expert_cols[i], expert_cols[j]
            result["pairwise_expert_cohen_kappa"][f"{a}_vs_{b}"] = float(
                cohen_kappa_score(df[a].astype(int), df[b].astype(int))
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sensitivity")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="remaining_validation")
    p.set_defaults(func=sensitivity)

    p = sub.add_parser("prepare-expert")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=prepare_expert)

    p = sub.add_parser("analyse-expert")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=analyse_expert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
