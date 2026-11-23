                      
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_phase3_model_comparison_leakage_safe import (
    TEXT_COL, LABEL_COL, SESSION_COL, TURN_COL, NUMERIC_FEATURES,
    load_json, enrich_missing_numeric_features, make_session_splits, split_rows,
    numeric_matrix, labels, build_turn_feature_transformers, make_turn_features,
    make_sequence_pack, SequencePack, RNNClassifier, train_torch_model,
    set_seed, ensure_dir
)

def choose_threshold(y_val, p_val):
    ts = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(scores))])

def metrics_from_probs(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "threshold": float(threshold),
    }

def train_prompt_lr(train_rows, val_rows, test_rows, use_numeric):
    train_text = [str(r.get(TEXT_COL, "")) for r in train_rows]
    val_text = [str(r.get(TEXT_COL, "")) for r in val_rows]
    test_text = [str(r.get(TEXT_COL, "")) for r in test_rows]
    y_train, y_val, y_test = labels(train_rows), labels(val_rows), labels(test_rows)

    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr_t = vec.fit_transform(train_text)
    Xva_t = vec.transform(val_text)
    Xte_t = vec.transform(test_text)

    if use_numeric:
        scaler = StandardScaler()
        Xtr_n = scaler.fit_transform(numeric_matrix(train_rows))
        Xva_n = scaler.transform(numeric_matrix(val_rows))
        Xte_n = scaler.transform(numeric_matrix(test_rows))
        Xtr, Xva, Xte = hstack([Xtr_t, Xtr_n]), hstack([Xva_t, Xva_n]), hstack([Xte_t, Xte_n])
    else:
        Xtr, Xva, Xte = Xtr_t, Xva_t, Xte_t

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(Xtr, y_train)
    p_val = clf.predict_proba(Xva)[:, 1]
    threshold = choose_threshold(y_val, p_val)
    p_test = clf.predict_proba(Xte)[:, 1]
    return metrics_from_probs(y_test, p_test, threshold)

def build_text_only_turn_features(train_rows, val_rows, test_rows):
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit([str(r.get(TEXT_COL, "")) for r in train_rows])
    tr = vec.transform([str(r.get(TEXT_COL, "")) for r in train_rows]).toarray().astype(np.float32)
    va = vec.transform([str(r.get(TEXT_COL, "")) for r in val_rows]).toarray().astype(np.float32)
    te = vec.transform([str(r.get(TEXT_COL, "")) for r in test_rows]).toarray().astype(np.float32)
    return tr, va, te

def shuffled_pack(pack: SequencePack, seed: int) -> SequencePack:
    rng = np.random.default_rng(seed)
    X = pack.X.copy()
    for i, length in enumerate(pack.lengths):
        length = int(length)
        if length > 1:
            perm = rng.permutation(length)
            X[i, :length, :] = X[i, perm, :]
    return SequencePack(X=X, lengths=pack.lengths.copy(), y=pack.y.copy())

def train_gru_condition(name, train_pack, val_pack, test_pack, args):
    set_seed(args.seed)
    model = RNNClassifier(
        input_dim=train_pack.X.shape[-1],
        hidden_dim=args.hidden_dim,
        rnn_type="gru",
        bidirectional=False,
        dropout=0.25,
    )
    return train_torch_model(
        name=name, model=model,
        train_pack=train_pack, val_pack=val_pack, test_pack=test_pack,
        output_dir="data/ablation_leakage_safe/models",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/profiles/balanced/prompt_sessions_labeled.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--max_len", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_json(args.input)
    rows = enrich_missing_numeric_features(rows)
    rows = sorted(rows, key=lambda r: (int(r[SESSION_COL]), int(r[TURN_COL])))

    train_ids, val_ids, test_ids = make_session_splits(rows, seed=args.seed)
    train_rows, val_rows, test_rows = split_rows(rows, train_ids, val_ids, test_ids)

    results = {}
    print("\nPrompt text only")
    results["prompt_text_only"] = train_prompt_lr(train_rows, val_rows, test_rows, use_numeric=False)

    print("\nPrompt text + numeric")
    results["prompt_text_numeric"] = train_prompt_lr(train_rows, val_rows, test_rows, use_numeric=True)

    tr_txt, va_txt, te_txt = build_text_only_turn_features(train_rows, val_rows, test_rows)
    tr_pack_txt = make_sequence_pack(train_rows, tr_txt, args.max_len)
    va_pack_txt = make_sequence_pack(val_rows, va_txt, args.max_len)
    te_pack_txt = make_sequence_pack(test_rows, te_txt, args.max_len)

    print("\nGRU text only")
    results["sequence_text_only"] = train_gru_condition(
        "ablation_gru_text_only", tr_pack_txt, va_pack_txt, te_pack_txt, args
    )

    vec, scaler = build_turn_feature_transformers(train_rows, max_features=1000)
    tr_num = make_turn_features(train_rows, vec, scaler)
    va_num = make_turn_features(val_rows, vec, scaler)
    te_num = make_turn_features(test_rows, vec, scaler)
    tr_pack_num = make_sequence_pack(train_rows, tr_num, args.max_len)
    va_pack_num = make_sequence_pack(val_rows, va_num, args.max_len)
    te_pack_num = make_sequence_pack(test_rows, te_num, args.max_len)

    print("\nGRU text + numeric")
    results["sequence_text_numeric"] = train_gru_condition(
        "ablation_gru_text_numeric", tr_pack_num, va_pack_num, te_pack_num, args
    )

                                                                                 
                                                        
    tr_shuf = shuffled_pack(tr_pack_num, args.seed + 101)
    va_shuf = shuffled_pack(va_pack_num, args.seed + 202)
    te_shuf = shuffled_pack(te_pack_num, args.seed + 303)

    print("\nGRU shuffled order")
    results["sequence_shuffled_history"] = train_gru_condition(
        "ablation_gru_shuffled", tr_shuf, va_shuf, te_shuf, args
    )

    outdir, figdir = "data/ablation_leakage_safe", "figures"
    ensure_dir(outdir)
    ensure_dir(figdir)

    out_json = os.path.join(outdir, "ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "seed": args.seed,
            "numeric_features": NUMERIC_FEATURES,
            "split": "session-level 70/15/15",
            "shuffled_control": "within-prefix order permutation; content, length, and target label preserved",
            "results": results
        }, f, indent=2)

    keys = ["prompt_text_only","prompt_text_numeric","sequence_text_only","sequence_text_numeric","sequence_shuffled_history"]
    names = ["Prompt\nText","Prompt\nText+Num","GRU\nText","GRU\nText+Num","GRU\nShuffled"]
    f1_vals = [results[k]["f1"] for k in keys]
    auc_vals = [results[k]["roc_auc"] for k in keys]

    x = np.arange(len(keys)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x-width/2, f1_vals, width, label="F1")
    ax.bar(x+width/2, auc_vals, width, label="ROC-AUC")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Leakage-Safe Ablation Study"); ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(figdir, "ablation_results_leakage_safe.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nAblation results")
    for k in keys:
        r = results[k]
        print(f"{k}: Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, F1={r['f1']:.4f}, AUC={r['roc_auc']:.4f}")
    print(f"\nSaved results: {out_json}")
    print(f"Saved figure: {fig_path}")

if __name__ == "__main__":
    main()
