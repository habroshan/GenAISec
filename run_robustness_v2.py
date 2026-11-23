                      


   

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import time
from collections import defaultdict
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

try:
    import psutil
except ImportError:
    psutil = None

from train_phase3_model_comparison_leakage_safe import (
    TEXT_COL, LABEL_COL, SESSION_COL, TURN_COL, NUMERIC_FEATURES,
    set_seed, ensure_dir, load_json, enrich_missing_numeric_features,
    make_session_splits, split_rows, build_cumulative_text, numeric_matrix, labels,
    build_turn_feature_transformers, make_turn_features, make_sequence_pack,
    RNNClassifier, TransformerClassifier
)


                                                                             
           
                                                                             

def metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None,
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def best_threshold(y, probs):
    ts = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(scores))])


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def by_session(rows):
    d = defaultdict(list)
    for r in rows:
        d[int(r[SESSION_COL])].append(r)
    for sid in d:
        d[sid] = sorted(d[sid], key=lambda x: int(x[TURN_COL]))
    return d


                                                                             
                        
                                                                             

def fit_cumulative_lr(train_rows, val_rows):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    scaler = StandardScaler()

    Xtr_t = vec.fit_transform(build_cumulative_text(train_rows))
    Xva_t = vec.transform(build_cumulative_text(val_rows))
    Xtr_n = scaler.fit_transform(numeric_matrix(train_rows))
    Xva_n = scaler.transform(numeric_matrix(val_rows))

    Xtr = hstack([Xtr_t, Xtr_n])
    Xva = hstack([Xva_t, Xva_n])

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42
    )
    clf.fit(Xtr, labels(train_rows))
    pva = clf.predict_proba(Xva)[:, 1]
    threshold = best_threshold(labels(val_rows), pva)
    return vec, scaler, clf, threshold


def cumulative_lr_predict(rows, vec, scaler, clf):
    Xt = vec.transform(build_cumulative_text(rows))
    Xn = scaler.transform(numeric_matrix(rows))
    X = hstack([Xt, Xn])
    return clf.predict_proba(X)[:, 1]


                                                                             
                                   
                                                                             

def neural_components(train_rows, args):
    vec, scaler = build_turn_feature_transformers(train_rows, max_features=args.tfidf_features)
    train_feat = make_turn_features(train_rows, vec, scaler)
    input_dim = train_feat.shape[1]
    return vec, scaler, input_dim


def make_model(kind, input_dim, args):
    if kind == "gru":
        return RNNClassifier(input_dim, args.hidden_dim, "gru", False, 0.25)
    if kind == "bilstm":
        return RNNClassifier(input_dim, args.hidden_dim, "lstm", True, 0.25)
    if kind == "transformer":
        return TransformerClassifier(
            input_dim, d_model=128, nhead=4, num_layers=2,
            max_len=args.max_len, dropout=0.25
        )
    raise ValueError(kind)


def load_model(kind, ckpt, input_dim, args, device="cpu"):
    model = make_model(kind, input_dim, args)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def neural_predict(rows, vec, scaler, model, args, device="cpu", batch_size=256):
    feat = make_turn_features(rows, vec, scaler)
    pack = make_sequence_pack(rows, feat, args.max_len)

    X = torch.tensor(pack.X, dtype=torch.float32)
    L = torch.tensor(pack.lengths, dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            lb = L[i:i+batch_size].to(device)
            p = torch.sigmoid(model(xb, lb)).cpu().numpy()
            out.extend(p.tolist())
    return np.asarray(out, dtype=float), pack.y, pack.lengths


                                                                             
                      
                                                                             

DLP_PATTERNS = [
    r"\bproject\b", r"\binvoice\b", r"\bbudget\b", r"\bpayment\b",
    r"\bdeadline\b", r"\baccess\b", r"\bcredential", r"\bpassword\b",
    r"\binternal\b", r"\bvendor\b", r"\bsupplier\b",
    r"£\s?\d+", r"\$\s?\d+", r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
]
DLP_RE = re.compile("|".join(DLP_PATTERNS), flags=re.I)


def regex_score(rows):
    return np.asarray([len(DLP_RE.findall(str(r.get(TEXT_COL, "")))) for r in rows], dtype=float)


def entity_score(rows):
                                                                                     
    return np.asarray([len(r.get("referenced_entities", []) or []) for r in rows], dtype=float)


def tune_score_threshold(y_val, score_val):
    unique = sorted(set(score_val.tolist()))
    candidates = [-0.5] + [u - 1e-9 for u in unique] + [u + 1e-9 for u in unique]
    best = None
    for t in candidates:
        pred = (score_val > t).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        if best is None or f > best[0]:
            best = (f, float(t))
    return best[1]


def score_to_probs(score):
                                                                                         
    if len(score) == 0:
        return score
    mx = max(float(score.max()), 1.0)
    return np.asarray(score, dtype=float) / mx


def retrieval_baseline(train_rows, val_rows, test_rows, k=7):
    tr_text = build_cumulative_text(train_rows)
    va_text = build_cumulative_text(val_rows)
    te_text = build_cumulative_text(test_rows)

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text)
    Xva = vec.transform(va_text)
    Xte = vec.transform(te_text)
    ytr = labels(train_rows)

    nn = NearestNeighbors(n_neighbors=min(k, len(train_rows)), metric="cosine", algorithm="brute")
    nn.fit(Xtr)

    def pred(X):
        dist, idx = nn.kneighbors(X)
        sim = 1.0 - dist
        yy = ytr[idx]
        denom = np.maximum(sim.sum(axis=1), 1e-9)
        return (sim * yy).sum(axis=1) / denom

    pva = pred(Xva)
    pte = pred(Xte)
    t = best_threshold(labels(val_rows), pva)
    return pte, t


def prompt_lr_baseline(train_rows, val_rows, test_rows):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([r.get(TEXT_COL, "") for r in train_rows])
    Xva = vec.transform([r.get(TEXT_COL, "") for r in val_rows])
    Xte = vec.transform([r.get(TEXT_COL, "") for r in test_rows])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    clf.fit(Xtr, labels(train_rows))
    pva = clf.predict_proba(Xva)[:,1]
    pte = clf.predict_proba(Xte)[:,1]
    return pte, best_threshold(labels(val_rows), pva)


                                                                             
                            
                                                                             

SYNONYMS = {
    "summarise": "condense",
    "summarize": "condense",
    "rewrite": "rephrase",
    "improve": "refine",
    "clearer": "easier to understand",
    "professional": "business-appropriate",
    "message": "note",
    "update": "status note",
    "deadline": "due date",
    "supplier": "provider",
    "vendor": "provider",
    "project": "initiative",
    "issue": "matter",
    "internal": "in-house",
}

CODE_MIX = {
    "please": "por favor",
    "project": "proyecto",
    "message": "mensaje",
    "update": "actualización",
    "deadline": "fecha límite",
    "supplier": "proveedor",
    "payment": "pago",
    "access": "acceso",
    "system": "sistema",
    "team": "equipo",
    "internal": "interno",
}


def replace_words(text, mapping):
    out = text
    for a, b in mapping.items():
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.I)
    return out


def abbreviate_entity(value):
    toks = re.findall(r"[A-Za-z0-9]+", str(value))
    if len(toks) >= 2:
        return "".join(t[0] for t in toks if t) + "."
    v = str(value)
    if len(v) <= 4:
        return v[0] + "***"
    return v[:2] + "***" + v[-1:]


def transform_rows(rows, condition):
    out = copy.deepcopy(rows)

    if condition == "original":
        return out

    if condition == "synonym":
        for r in out:
            r[TEXT_COL] = replace_words(str(r.get(TEXT_COL, "")), SYNONYMS)
        return out

    if condition == "code_mixed":
        for r in out:
            r[TEXT_COL] = replace_words(str(r.get(TEXT_COL, "")), CODE_MIX)
        return out

    if condition == "entity_text_obfuscation":
                                                                        
                                                                      
                                                      
        for r in out:
            txt = str(r.get(TEXT_COL, ""))
            ents = r.get("referenced_entities", []) or []
            for e in ents:
                val = str(e.get("value", ""))
                if val:
                    txt = txt.replace(val, abbreviate_entity(val))
            r[TEXT_COL] = txt
        return out

    if condition == "entity_extraction_failure":
                                                                             
                                                                             
                                                        
        for r in out:
            txt = str(r.get(TEXT_COL, ""))
            ents = r.get("referenced_entities", []) or []
            for e in ents:
                val = str(e.get("value", ""))
                if val:
                    txt = txt.replace(val, abbreviate_entity(val))
            r[TEXT_COL] = txt
            r["referenced_entities"] = []
        return enrich_missing_numeric_features_force(out)

    if condition == "delayed":
        return delayed_rows(out)

    raise ValueError(condition)


def enrich_missing_numeric_features_force(rows):
                                                                           
    sessions = by_session(rows)
    enriched = []
    for sid, srows in sessions.items():
        seen_values, seen_types = set(), set()
        for r in srows:
            rr = dict(r)
            ents = rr.get("referenced_entities", []) or []
            current_values = {(e.get("type"), e.get("value")) for e in ents}
            current_types = {e.get("type") for e in ents if e.get("type")}
            new_values = current_values - seen_values
            new_types = current_types - seen_types
            seen_values.update(current_values)
            seen_types.update(current_types)

            rr["new_entity_count"] = len(new_values)
            rr["new_entity_type_count"] = len(new_types)
            rr["cumulative_entity_count"] = len(seen_values)
            rr["cumulative_entity_type_count"] = len(seen_types)
            enriched.append(rr)
    return sorted(enriched, key=lambda r:(int(r[SESSION_COL]), int(r[TURN_COL])))


def delayed_rows(rows):
    


       
    sessions = by_session(rows)
    out = []
    new_prompt_id = max(int(r.get("prompt_id", 0)) for r in rows) + 1

    for sid, srows in sessions.items():
        new_turn = 0
        for j, r in enumerate(srows):
            rr = dict(r)
            rr[TURN_COL] = new_turn
            out.append(rr)
            new_turn += 1

            if j < len(srows) - 1:
                noise = dict(r)
                noise["prompt_id"] = new_prompt_id
                new_prompt_id += 1
                noise[TURN_COL] = new_turn
                noise[TEXT_COL] = "Please improve the wording without adding any new information."
                noise["prompt_type"] = "noise"
                noise["referenced_entities"] = []
                                                                                               
                noise[LABEL_COL] = int(r[LABEL_COL])
                out.append(noise)
                new_turn += 1

    return enrich_missing_numeric_features_force(out)


                                                                             
                      
                                                                             

def file_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def profile_neural(model, pack, device="cpu", warmup=50, n_samples=1000):
    X = torch.tensor(pack.X, dtype=torch.float32)
    L = torch.tensor(pack.lengths, dtype=torch.long)
    n = min(n_samples, len(X))
    idx = np.linspace(0, len(X)-1, n, dtype=int)

    proc = psutil.Process(os.getpid()) if psutil else None
    rss_before = proc.memory_info().rss if proc else None

    with torch.no_grad():
        for i in idx[:min(warmup, len(idx))]:
            _ = model(X[i:i+1].to(device), L[i:i+1].to(device))

    times = []
    with torch.no_grad():
        for i in idx:
            xb = X[i:i+1].to(device)
            lb = L[i:i+1].to(device)
            t0 = time.perf_counter()
            _ = model(xb, lb)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    rss_after = proc.memory_info().rss if proc else None
    return {
        "n_profiled": int(n),
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_median": float(np.median(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "rss_delta_mb": None if rss_before is None else float((rss_after-rss_before)/(1024*1024)),
    }


def profile_by_length(model, pack, device="cpu", max_each=200):
    result = {}
    for length in sorted(set(pack.lengths.tolist())):
        ids = np.where(pack.lengths == length)[0][:max_each]
        if len(ids) < 5:
            continue
        times = []
        X = torch.tensor(pack.X[ids], dtype=torch.float32)
        L = torch.tensor(pack.lengths[ids], dtype=torch.long)
        with torch.no_grad():
            for i in range(len(ids)):
                t0 = time.perf_counter()
                _ = model(X[i:i+1].to(device), L[i:i+1].to(device))
                times.append((time.perf_counter()-t0)*1000.0)
        result[str(int(length))] = {
            "n": int(len(ids)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times,95)),
        }
    return result


                                                                             
                     
                                                                             

def intervention_table(rows, y, probs, thresholds):
    sess = np.asarray([int(r[SESSION_COL]) for r in rows])
    records = []

    for t in thresholds:
        pred = (probs >= t).astype(int)
        benign_mask = (y == 0)
        false_alerts = int(((pred == 1) & benign_mask).sum())
        benign_total = int(benign_mask.sum())

        benign_session_ids = set(sess[benign_mask].tolist())
        affected = set(sess[(pred == 1) & benign_mask].tolist())

        records.append({
            "threshold": float(t),
            "precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)),
            "false_alerts": false_alerts,
            "false_alerts_per_1000_benign_prefixes":
                float(1000.0*false_alerts/max(1,benign_total)),
            "benign_sessions": int(len(benign_session_ids)),
            "benign_sessions_with_false_alert": int(len(affected)),
            "pct_benign_sessions_affected":
                float(100.0*len(affected)/max(1,len(benign_session_ids))),
            "overall_intervention_rate":
                float(100.0*pred.mean()),
        })
    return records


                                                                             
      
                                                                             

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--balanced", required=True)
    p.add_argument("--realistic", default=None)
    p.add_argument("--risk", default=None)
    p.add_argument("--gru", required=True)
    p.add_argument("--bilstm", required=True)
    p.add_argument("--transformer", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tfidf_features", type=int, default=1000)
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--max_len", type=int, default=14)
    p.add_argument("--outdir", default="data/reviewer_remaining_validation")
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    rows = enrich_missing_numeric_features(load_json(args.balanced))
    rows = sorted(rows, key=lambda r:(int(r[SESSION_COL]), int(r[TURN_COL])))
    train_ids, val_ids, test_ids = make_session_splits(rows, args.seed)
    train_rows, val_rows, test_rows = split_rows(rows, train_ids, val_ids, test_ids)

                                      
    lr_vec, lr_scaler, lr_clf, lr_t = fit_cumulative_lr(train_rows, val_rows)
    nvec, nscaler, input_dim = neural_components(train_rows, args)

    gru = load_model("gru", args.gru, input_dim, args)
    bilstm = load_model("bilstm", args.bilstm, input_dim, args)
    transformer = load_model("transformer", args.transformer, input_dim, args)

                              
    pva_gru, yva_gru, _ = neural_predict(val_rows, nvec, nscaler, gru, args)
    gru_t = best_threshold(yva_gru, pva_gru)

                                                                        
                   
                                                                        
    robustness = []
    conditions = ["original", "synonym", "code_mixed", "entity_text_obfuscation", "entity_extraction_failure", "delayed"]

    for cond in conditions:
        rr = transform_rows(test_rows, cond)

        p_lr = cumulative_lr_predict(rr, lr_vec, lr_scaler, lr_clf)
        y_lr = labels(rr)
        m_lr = metrics(y_lr, p_lr, lr_t)
        robustness.append({"condition":cond, "model":"Cumulative LR + Numeric", **m_lr})

        p_gru, y_gru, _ = neural_predict(rr, nvec, nscaler, gru, args)
        m_gru = metrics(y_gru, p_gru, gru_t)
        robustness.append({"condition":cond, "model":"GRU", **m_gru})

    robustness_path = os.path.join(args.outdir, "robustness_results_v2.csv")
    pd.DataFrame(robustness).to_csv(robustness_path, index=False)

    print("\nRobustness results saved.")
    print("Saved:", robustness_path)
    print("\nConditions:")
    for c in conditions:
        print(" -", c)
    return

                                                                        
                                         
                                                                        
    baseline_rows = []
    ytest = labels(test_rows)

               
    p_prompt, t_prompt = prompt_lr_baseline(train_rows, val_rows, test_rows)
    baseline_rows.append({"model":"Prompt LR", **metrics(ytest,p_prompt,t_prompt)})

               
    sva, ste = regex_score(val_rows), regex_score(test_rows)
    t = tune_score_threshold(labels(val_rows), sva)
    pred = (ste > t).astype(int)
    baseline_rows.append({
        "model":"Keyword/Regex DLP",
        "accuracy":float(accuracy_score(ytest,pred)),
        "precision":float(precision_score(ytest,pred,zero_division=0)),
        "recall":float(recall_score(ytest,pred,zero_division=0)),
        "f1":float(f1_score(ytest,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(ytest,score_to_probs(ste))),
        "threshold":float(t), "n":int(len(ytest))
    })

                
    sva, ste = entity_score(val_rows), entity_score(test_rows)
    t = tune_score_threshold(labels(val_rows), sva)
    pred = (ste > t).astype(int)
    baseline_rows.append({
        "model":"Entity-count DLP",
        "accuracy":float(accuracy_score(ytest,pred)),
        "precision":float(precision_score(ytest,pred,zero_division=0)),
        "recall":float(recall_score(ytest,pred,zero_division=0)),
        "f1":float(f1_score(ytest,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(ytest,score_to_probs(ste))),
        "threshold":float(t), "n":int(len(ytest))
    })

                        
    p_ret, t_ret = retrieval_baseline(train_rows,val_rows,test_rows,k=7)
    baseline_rows.append({"model":"Cumulative Retrieval (kNN)", **metrics(ytest,p_ret,t_ret)})

                            
    p_lr = cumulative_lr_predict(test_rows, lr_vec, lr_scaler, lr_clf)
    baseline_rows.append({"model":"Cumulative LR + Numeric", **metrics(ytest,p_lr,lr_t)})

         
    p_gru, y_gru, _ = neural_predict(test_rows,nvec,nscaler,gru,args)
    baseline_rows.append({"model":"GRU", **metrics(y_gru,p_gru,gru_t)})

    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(args.outdir, "enterprise_baseline_results.csv"), index=False
    )

                                                                        
                             
                                                                        
    test_feat = make_turn_features(test_rows,nvec,nscaler)
    test_pack = make_sequence_pack(test_rows,test_feat,args.max_len)

    deployment = {}
    for kind, model, path in [
        ("GRU",gru,args.gru),
        ("BiLSTM",bilstm,args.bilstm),
        ("Transformer",transformer,args.transformer),
    ]:
        deployment[kind] = {
            "checkpoint_size_mb": file_mb(path),
            **profile_neural(model,test_pack,device="cpu"),
            "latency_by_prefix_length": profile_by_length(model,test_pack,device="cpu")
        }

    save_json(deployment, os.path.join(args.outdir, "deployment_profile.json"))

                                                                        
                                             
                                                                        
    thresholds = sorted(set([0.50,0.60,0.70,0.80,0.90,round(gru_t,4)]))
    intervention = intervention_table(test_rows,y_gru,p_gru,thresholds)
    pd.DataFrame(intervention).to_csv(
        os.path.join(args.outdir, "intervention_burden.csv"), index=False
    )

                                                                        
                                                         
                                                                        
    cross = []
    for name, path in [("Realistic",args.realistic),("Risk-Enriched",args.risk)]:
        if not path:
            continue
        r = enrich_missing_numeric_features(load_json(path))
        r = sorted(r,key=lambda x:(int(x[SESSION_COL]),int(x[TURN_COL])))
        p, y, _ = neural_predict(r,nvec,nscaler,gru,args)
        cross.append({"profile":name, **metrics(y,p,gru_t)})
    if cross:
        pd.DataFrame(cross).to_csv(
            os.path.join(args.outdir,"cross_profile_checkpoint_results.csv"), index=False
        )

    manifest = {
        "seed":args.seed,
        "numeric_features":NUMERIC_FEATURES,
        "tfidf_features":args.tfidf_features,
        "sequence_input_dim":input_dim,
        "hidden_dim":args.hidden_dim,
        "max_len":args.max_len,
        "outputs":[
            "robustness_results.csv",
            "enterprise_baseline_results.csv",
            "deployment_profile.json",
            "intervention_burden.csv",
            "cross_profile_checkpoint_results.csv (if profiles supplied)"
        ],
        "notes":[
            "Robustness perturbations are deterministic controlled stress tests, not natural-language human evaluation.",
            "Code-mixed perturbation is a controlled partial lexical code-mix, not a full multilingual benchmark.",
            "Cumulative Retrieval (kNN) is a reproducible retrieval-based multi-turn comparator, not an external LLM API.",
            "Intervention analysis quantifies false-positive burden but is not a user study."
        ]
    }
    save_json(manifest,os.path.join(args.outdir,"manifest.json"))

    print("\nValidation results saved.")
    print("Outputs in:", args.outdir)
    for fn in manifest["outputs"]:
        print(" -",fn)


if __name__ == "__main__":
    main()
