from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from transformers import AutoModelForCausalLM, AutoTokenizer

from train_phase3_model_comparison_leakage_safe import (
    TEXT_COL, LABEL_COL, SESSION_COL, TURN_COL,
    load_json, enrich_missing_numeric_features,
    make_session_splits, split_rows, build_cumulative_text,
    numeric_matrix, labels, build_turn_feature_transformers,
    make_turn_features, make_sequence_pack, RNNClassifier, set_seed
)


def best_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 181)
    vals = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(vals))])


def metric_dict(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def fit_cumulative_lr(train_rows, val_rows):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    scaler = StandardScaler()

    Xtr_t = vec.fit_transform(build_cumulative_text(train_rows))
    Xva_t = vec.transform(build_cumulative_text(val_rows))
    Xtr_n = scaler.fit_transform(numeric_matrix(train_rows))
    Xva_n = scaler.transform(numeric_matrix(val_rows))

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42
    )
    clf.fit(hstack([Xtr_t, Xtr_n]), labels(train_rows))
    pva = clf.predict_proba(hstack([Xva_t, Xva_n]))[:, 1]
    return vec, scaler, clf, best_threshold(labels(val_rows), pva)


def predict_cumulative_lr(rows, vec, scaler, clf):
    Xt = vec.transform(build_cumulative_text(rows))
    Xn = scaler.transform(numeric_matrix(rows))
    return clf.predict_proba(hstack([Xt, Xn]))[:, 1]


def choose_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def stratified_subset(rows, n, seed):
    if n <= 0 or n >= len(rows):
        return list(range(len(rows)))

    y = labels(rows)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_pos = max(1, int(round(n * len(pos) / len(rows))))
    n_neg = n - n_pos
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))

    chosen = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    rng.shuffle(chosen)
    return chosen.tolist()


SYSTEM_TEXT = (
    "You are a cybersecurity risk classifier for enterprise Generative AI usage. "
    "Classify the cumulative employee interaction history available so far. "
    "RISK means that the disclosed information, considered together, is sufficiently "
    "specific and contextual to materially enable a targeted enterprise cyberattack "
    "such as spear-phishing, invoice fraud, or an access-pretext/social-engineering attack. "
    "SAFE means the accumulated disclosure is not yet sufficiently actionable. "
    "Judge cumulative context rather than whether a single sentence contains a sensitive word. "
    "Return only SAFE or RISK."
)


def target_prompt(history):
    return (
        "CUMULATIVE INTERACTION HISTORY:\n"
        f"{history}\n\n"
        "Classification:"
    )


def rag_prompt(history, examples):
    chunks = [
        "Below are examples retrieved from labelled training interactions. "
        "Use them only as demonstrations of the classification task."
    ]
    for i, (h, lab) in enumerate(examples, 1):
        chunks.append(
            f"\nExample {i}\n"
            f"CUMULATIVE INTERACTION HISTORY:\n{h}\n"
            f"Classification: {'RISK' if lab == 1 else 'SAFE'}"
        )
    chunks.append(
        "\nNow classify this interaction:\n"
        f"CUMULATIVE INTERACTION HISTORY:\n{history}\n\n"
        "Classification:"
    )
    return "\n".join(chunks)


def chat_base(tokenizer, user_text):
    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user_text},
    ]
                                                                                       
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def candidate_logprob_batch(model, tokenizer, bases, candidate, device):
    


       
    fulls = [b + candidate for b in bases]
    base_enc = tokenizer(
        bases, add_special_tokens=False, padding=True, truncation=True,
        max_length=4096, return_tensors="pt"
    )
    full_enc = tokenizer(
        fulls, add_special_tokens=False, padding=True, truncation=True,
        max_length=4096, return_tensors="pt"
    )

    input_ids = full_enc["input_ids"].to(device)
    attn = full_enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits

    logp = torch.log_softmax(logits, dim=-1)
    scores = []

                                                               
    for i in range(len(bases)):
        base_len = int(base_enc["attention_mask"][i].sum().item())
        full_len = int(full_enc["attention_mask"][i].sum().item())

                                                    
        cand_ids = full_enc["input_ids"][i, base_len:full_len].to(device)
        if len(cand_ids) == 0:
            scores.append(float("-inf"))
            continue

                                                              
        positions = torch.arange(base_len - 1, full_len - 1, device=device)
        tok_lp = logp[i, positions, cand_ids]
        scores.append(float(tok_lp.sum().detach().cpu().item()))

    return np.asarray(scores, dtype=np.float64)


def llm_probabilities(model, tokenizer, prompts, device, batch_size):
    probs = []
    tokenizer.padding_side = "right"

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        bases = [chat_base(tokenizer, x) for x in chunk]

                                                                  
        s_safe = candidate_logprob_batch(model, tokenizer, bases, " SAFE", device)
        s_risk = candidate_logprob_batch(model, tokenizer, bases, " RISK", device)

                                   
        m = np.maximum(s_safe, s_risk)
        e0 = np.exp(s_safe - m)
        e1 = np.exp(s_risk - m)
        p = e1 / np.maximum(e0 + e1, 1e-12)
        probs.extend(p.tolist())

        print(f"LLM scored {min(start + batch_size, len(prompts))}/{len(prompts)}")

    return np.asarray(probs, dtype=float)


def build_retriever(train_rows, k):
    histories = build_cumulative_text(train_rows)
    y = labels(train_rows)

    vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X = vec.fit_transform(histories)
    nn = NearestNeighbors(
        n_neighbors=min(k, len(histories)), metric="cosine", algorithm="brute"
    )
    nn.fit(X)
    return histories, y, vec, nn


def retrieve_examples(histories, y_train, vec, nn, target_histories, k):
    Xq = vec.transform(target_histories)
    _, idx = nn.kneighbors(Xq, n_neighbors=k)
    out = []
    for row in idx:
        out.append([(histories[j], int(y_train[j])) for j in row])
    return out


def load_gru(train_rows, checkpoint, args, device="cpu"):
    vec, scaler = build_turn_feature_transformers(train_rows, args.tfidf_features)
    tr_feat = make_turn_features(train_rows, vec, scaler)
    input_dim = tr_feat.shape[1]

    model = RNNClassifier(
        input_dim, args.hidden_dim, "gru", False, 0.25
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return vec, scaler, model


def gru_predict(rows, vec, scaler, model, args, device="cpu"):
    feat = make_turn_features(rows, vec, scaler)
    pack = make_sequence_pack(rows, feat, args.max_len)
    X = torch.tensor(pack.X, dtype=torch.float32)
    L = torch.tensor(pack.lengths, dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            p = torch.sigmoid(
                model(X[i:i+256].to(device), L[i:i+256].to(device))
            )
            out.extend(p.cpu().numpy().tolist())
    return np.asarray(out), pack.y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--balanced", required=True)
    p.add_argument("--gru", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sample-size", type=int, default=1200,
                   help="0 means full test set")
    p.add_argument("--retrieval-k", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tfidf-features", type=int, default=1000)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--max-len", type=int, default=14)
    p.add_argument("--outdir", default="data/llm_rag_comparator")
    args = p.parse_args()

    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    rows = enrich_missing_numeric_features(load_json(args.balanced))
    rows = sorted(rows, key=lambda r: (int(r[SESSION_COL]), int(r[TURN_COL])))
    tr_ids, va_ids, te_ids = make_session_splits(rows, args.seed)
    train_rows, val_rows, test_rows = split_rows(rows, tr_ids, va_ids, te_ids)

    subset_idx = stratified_subset(test_rows, args.sample_size, args.seed + 991)
    eval_rows = [test_rows[i] for i in subset_idx]
    eval_y = labels(eval_rows)
    eval_hist = build_cumulative_text(eval_rows)

    print(
        f"Evaluation subset: n={len(eval_rows)}, "
        f"positive_rate={eval_y.mean():.4f}"
    )

                                                                     
                                                                
                                                                     
    lr_vec, lr_scaler, lr_clf, lr_t = fit_cumulative_lr(train_rows, val_rows)
    p_lr = predict_cumulative_lr(eval_rows, lr_vec, lr_scaler, lr_clf)

    gvec, gscaler, gru = load_gru(train_rows, args.gru, args, device="cpu")
    pva_gru, yva_gru = gru_predict(val_rows, gvec, gscaler, gru, args, device="cpu")
    gru_t = best_threshold(yva_gru, pva_gru)
    p_gru, y_gru = gru_predict(eval_rows, gvec, gscaler, gru, args, device="cpu")

                                                                     
                                 
                                                                     
    device, dtype = choose_device()
    print(f"Loading {args.model} on {device} with dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    zero_prompts = [target_prompt(h) for h in eval_hist]
    p_llm_zero = llm_probabilities(
        model, tokenizer, zero_prompts, device, args.batch_size
    )

                                                                               
                                               
    zero_t = 0.5

                                                                     
                                                      
                                                                     
    tr_histories, ytr, rvec, rnn = build_retriever(
        train_rows, args.retrieval_k
    )
    retrieved = retrieve_examples(
        tr_histories, ytr, rvec, rnn, eval_hist, args.retrieval_k
    )
    rag_prompts = [
        rag_prompt(h, ex) for h, ex in zip(eval_hist, retrieved)
    ]
    p_llm_rag = llm_probabilities(
        model, tokenizer, rag_prompts, device, args.batch_size
    )
    rag_t = 0.5

                                                                     
             
                                                                     
    results = [
        {"model": "Local LLM zero-shot", **metric_dict(eval_y, p_llm_zero, zero_t)},
        {"model": f"Retrieval-augmented LLM (k={args.retrieval_k})",
         **metric_dict(eval_y, p_llm_rag, rag_t)},
        {"model": "Cumulative LR + Numeric",
         **metric_dict(eval_y, p_lr, lr_t)},
        {"model": "GRU",
         **metric_dict(y_gru, p_gru, gru_t)},
    ]

    out_csv = os.path.join(args.outdir, "llm_rag_comparator_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    pred_df = pd.DataFrame({
        "subset_index": subset_idx,
        "session_id": [int(r[SESSION_COL]) for r in eval_rows],
        "turn_id": [int(r[TURN_COL]) for r in eval_rows],
        "y_true": eval_y,
        "p_llm_zero": p_llm_zero,
        "p_llm_rag": p_llm_rag,
        "p_cumulative_lr_numeric": p_lr,
        "p_gru": p_gru,
    })
    pred_df.to_csv(
        os.path.join(args.outdir, "llm_rag_comparator_predictions.csv"),
        index=False
    )

    manifest = {
        "llm_model": args.model,
        "sample_size": len(eval_rows),
        "sample_selection": "stratified random subset of seed-42 test prefixes",
        "test_positive_rate": float(eval_y.mean()),
        "retrieval_k": args.retrieval_k,
        "retrieval_corpus": "training prefixes only",
        "llm_threshold": 0.5,
        "device": device,
        "notes": [
            "The LLM comparator receives cumulative text history only.",
            "The retrieval-augmented LLM retrieves labelled exemplars only from the training partition.",
            "This is a task-aligned contemporary LLM comparator, not a reimplementation of a published cumulative-disclosure detector.",
            "No attack-template scores or label-derived structured features are provided to the proposed models."
        ],
    }
    with open(os.path.join(args.outdir, "llm_rag_comparator_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nResults")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
