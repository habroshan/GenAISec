


   

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for GRU/BiLSTM/Transformer experiments. Install with: pip install torch"
    ) from exc


                                                                             
               
                                                                             

TEXT_COL = "prompt_text"
LABEL_COL = "attack_ready_label"
SESSION_COL = "session_id"
TURN_COL = "turn_id"

                                   
 
            
                                                    
                                                                          
                                                 
                                                                     
                                                                             
                                                                            
                                                                           
                            
 
                                                                           
                                                                           
NUMERIC_FEATURES = [
    "turn_id",
    "new_entity_count",
    "new_entity_type_count",
    "cumulative_entity_count",
    "cumulative_entity_type_count",
]


                                                                             
           
                                                                             

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_entities(row: dict) -> List[dict]:
    ents = row.get("referenced_entities", [])
    return ents if isinstance(ents, list) else []


def enrich_missing_numeric_features(rows: List[dict]) -> List[dict]:
                                                                                     
    sessions: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        sessions[int(r[SESSION_COL])].append(r)

    enriched = []
    for sid, srows in sessions.items():
        srows = sorted(srows, key=lambda x: int(x[TURN_COL]))
        seen_values = set()
        seen_types = set()
        seen_contexts = set()

        for r in srows:
            ents = get_entities(r)
            current_values = {(e.get("type"), e.get("value")) for e in ents}
            current_types = {e.get("type") for e in ents if e.get("type")}

                                                                       
            type_to_context = {
                "employee": "identity",
                "manager": "authority",
                "role": "authority",
                "department": "authority",
                "project": "business",
                "vendor": "vendor",
                "system": "technical",
                "invoice": "financial",
                "amount": "financial",
                "date": "timing",
            }
            current_contexts = {type_to_context.get(t) for t in current_types if type_to_context.get(t)}

            new_values = current_values - seen_values
            new_types = current_types - seen_types
            new_contexts = current_contexts - seen_contexts

            seen_values.update(current_values)
            seen_types.update(current_types)
            seen_contexts.update(current_contexts)

            r = dict(r)
            r.setdefault("new_entity_count", len(new_values))
            r.setdefault("new_entity_type_count", len(new_types))
            r.setdefault("new_context_count", len(new_contexts))
            r.setdefault("cumulative_entity_count", len(seen_values))
            r.setdefault("cumulative_entity_type_count", len(seen_types))
            r.setdefault("cumulative_context_count", len(seen_contexts))
            r.setdefault("session_length", len(srows))
            r.setdefault("direct_sensitive", 0)
            r.setdefault("contextual_sensitive", 0)
            r.setdefault("cumulative_candidate", 0)
            enriched.append(r)

    return sorted(enriched, key=lambda x: (int(x[SESSION_COL]), int(x[TURN_COL])))


def make_session_splits(rows: List[dict], seed: int) -> Tuple[set, set, set]:
                                                                                                  
    session_labels = {}
    for r in rows:
        sid = int(r[SESSION_COL])
        session_labels[sid] = max(session_labels.get(sid, 0), int(r[LABEL_COL]))

    session_ids = np.array(sorted(session_labels.keys()))
    y = np.array([session_labels[sid] for sid in session_ids])

    train_ids, temp_ids, y_train, y_temp = train_test_split(
        session_ids,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    return set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())


def split_rows(rows: List[dict], train_ids: set, val_ids: set, test_ids: set):
    train, val, test = [], [], []
    for r in rows:
        sid = int(r[SESSION_COL])
        if sid in train_ids:
            train.append(r)
        elif sid in val_ids:
            val.append(r)
        elif sid in test_ids:
            test.append(r)
    return train, val, test


def build_cumulative_text(rows: List[dict]) -> List[str]:
    sessions = defaultdict(list)
    for i, r in enumerate(rows):
        sessions[int(r[SESSION_COL])].append((i, r))

    output = [""] * len(rows)
    for _, items in sessions.items():
        items = sorted(items, key=lambda ir: int(ir[1][TURN_COL]))
        texts = []
        for idx, r in items:
            texts.append(str(r.get(TEXT_COL, "")))
            output[idx] = "\n".join(texts)
    return output


def numeric_matrix(rows: List[dict]) -> np.ndarray:
    mat = []
    for r in rows:
        mat.append([float(r.get(k, 0) or 0) for k in NUMERIC_FEATURES])
    return np.asarray(mat, dtype=np.float32)


def labels(rows: List[dict]) -> np.ndarray:
    return np.asarray([int(r[LABEL_COL]) for r in rows], dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }
    return metrics


                                                                             
                     
                                                                             

def train_logistic_baseline(
    name: str,
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    cumulative: bool,
    use_numeric: bool,
    output_dir: str,
    max_features: int = 20000,
) -> dict:
    train_text = build_cumulative_text(train_rows) if cumulative else [r.get(TEXT_COL, "") for r in train_rows]
    val_text = build_cumulative_text(val_rows) if cumulative else [r.get(TEXT_COL, "") for r in val_rows]
    test_text = build_cumulative_text(test_rows) if cumulative else [r.get(TEXT_COL, "") for r in test_rows]

    y_train = labels(train_rows)
    y_val = labels(val_rows)
    y_test = labels(test_rows)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_text = vectorizer.fit_transform(train_text)
    X_val_text = vectorizer.transform(val_text)
    X_test_text = vectorizer.transform(test_text)

    if use_numeric:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(numeric_matrix(train_rows))
        X_val_num = scaler.transform(numeric_matrix(val_rows))
        X_test_num = scaler.transform(numeric_matrix(test_rows))
        X_train = hstack([X_train_text, X_train_num])
        X_val = hstack([X_val_text, X_val_num])
        X_test = hstack([X_test_text, X_test_num])
    else:
        X_train, X_val, X_test = X_train_text, X_val_text, X_test_text

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

                                                      
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])

    test_metrics = compute_metrics(y_test, test_prob, threshold=best_t)
    test_metrics.update({
        "model": name,
        "cumulative_input": cumulative,
        "use_numeric_features": use_numeric,
        "num_test_samples": int(len(y_test)),
    })

    save_json(test_metrics, os.path.join(output_dir, f"{name}_metrics.json"))
    save_json(
        [
            {"y_true": int(y), "y_prob": float(p), "y_pred": int(p >= best_t)}
            for y, p in zip(y_test, test_prob)
        ],
        os.path.join(output_dir, f"{name}_predictions.json"),
    )
    print(f"{name}: F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
    return test_metrics


                                                                             
                              
                                                                             

@dataclass
class SequencePack:
    X: np.ndarray
    lengths: np.ndarray
    y: np.ndarray


class PrefixSequenceDataset(Dataset):
    def __init__(self, pack: SequencePack):
        self.X = torch.tensor(pack.X, dtype=torch.float32)
        self.lengths = torch.tensor(pack.lengths, dtype=torch.long)
        self.y = torch.tensor(pack.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]


def build_turn_feature_transformers(train_rows: List[dict], max_features: int):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    scaler = StandardScaler()
    vectorizer.fit([r.get(TEXT_COL, "") for r in train_rows])
    scaler.fit(numeric_matrix(train_rows))
    return vectorizer, scaler


def make_turn_features(rows: List[dict], vectorizer, scaler) -> np.ndarray:
    X_text = vectorizer.transform([r.get(TEXT_COL, "") for r in rows]).toarray().astype(np.float32)
    X_num = scaler.transform(numeric_matrix(rows)).astype(np.float32)
    return np.concatenate([X_text, X_num], axis=1)


def make_sequence_pack(rows: List[dict], turn_features: np.ndarray, max_len: int) -> SequencePack:
    sessions = defaultdict(list)
    for i, r in enumerate(rows):
        sessions[int(r[SESSION_COL])].append((i, r))

    X_list, len_list, y_list = [], [], []
    feat_dim = turn_features.shape[1]

    for _, items in sessions.items():
        items = sorted(items, key=lambda ir: int(ir[1][TURN_COL]))
        prefix_feats = []
        for idx, r in items:
            prefix_feats.append(turn_features[idx])
            seq = np.asarray(prefix_feats[-max_len:], dtype=np.float32)
            length = seq.shape[0]
            padded = np.zeros((max_len, feat_dim), dtype=np.float32)
            padded[:length, :] = seq
            X_list.append(padded)
            len_list.append(length)
            y_list.append(int(r[LABEL_COL]))

    return SequencePack(
        X=np.asarray(X_list, dtype=np.float32),
        lengths=np.asarray(len_list, dtype=np.int64),
        y=np.asarray(y_list, dtype=np.int64),
    )


class RNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rnn_type: str, bidirectional: bool, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        z = self.proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        if self.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]
        return self.classifier(last).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, max_len: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, lengths):
        z = self.pos(self.proj(x))
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        enc = self.encoder(z, src_key_padding_mask=mask)
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, enc.size(-1))
        last = enc.gather(1, idx).squeeze(1)
        return self.classifier(last).squeeze(1)


def train_torch_model(
    name: str,
    model: nn.Module,
    train_pack: SequencePack,
    val_pack: SequencePack,
    test_pack: SequencePack,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> dict:
    train_loader = DataLoader(PrefixSequenceDataset(train_pack), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PrefixSequenceDataset(val_pack), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PrefixSequenceDataset(test_pack), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    y_train = train_pack.y
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = -1.0
    best_state = None
    patience = 3
    stale = 0
    history = []

    def predict(loader):
        model.eval()
        probs, ys = [], []
        total_loss = 0.0
        with torch.no_grad():
            for xb, lb, yb in loader:
                xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
                logits = model(xb, lb)
                loss = criterion(logits, yb)
                total_loss += float(loss.item()) * len(yb)
                probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
                ys.extend(yb.detach().cpu().numpy().astype(int).tolist())
        return np.asarray(ys), np.asarray(probs), total_loss / max(1, len(ys))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += float(loss.item()) * len(yb)

        y_val, p_val, val_loss = predict(val_loader)
        val_auc = roc_auc_score(y_val, p_val) if len(set(y_val.tolist())) > 1 else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_pack.y), "val_loss": val_loss, "val_auc": val_auc})
        print(f"{name} | epoch {epoch:02d} | val_auc={val_auc:.4f} | val_loss={val_loss:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val, p_val, _ = predict(val_loader)
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])

    y_test, p_test, test_loss = predict(test_loader)
    metrics = compute_metrics(y_test, p_test, threshold=best_t)
    metrics.update({
        "model": name,
        "test_loss": float(test_loss),
        "best_val_auc": float(best_val_auc),
        "num_test_samples": int(len(y_test)),
        "history": history,
    })

    save_json(metrics, os.path.join(output_dir, f"{name}_metrics.json"))
    save_json(
        [{"y_true": int(y), "y_prob": float(p), "y_pred": int(p >= best_t)} for y, p in zip(y_test, p_test)],
        os.path.join(output_dir, f"{name}_predictions.json"),
    )
    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}.pt"))
    print(f"{name}: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
    return metrics


                                                                             
           
                                                                             

def generate_comparison_outputs(metrics: List[dict], fig_dir: str, table_dir: str, output_dir: str) -> None:
    ensure_dir(fig_dir)
    ensure_dir(table_dir)

                                       
    order = [
        "prompt_lr_text",
        "cumulative_lr_text",
        "cumulative_lr_text_numeric",
        "gru_sequence",
        "bilstm_sequence",
        "transformer_sequence",
    ]
    metric_map = {m["model"]: m for m in metrics}
    metrics = [metric_map[k] for k in order if k in metric_map]

    labels_short = [
        "Prompt LR",
        "Cumulative LR",
        "Cum. LR + Num.",
        "GRU",
        "BiLSTM",
        "Transformer",
    ][: len(metrics)]

    x = np.arange(len(metrics))
    width = 0.35
    f1_vals = [m["f1"] for m in metrics]
    auc_vals = [m["roc_auc"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_vals, width, label="F1-score")
    ax.bar(x + width / 2, auc_vals, width, label="ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3 Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, "phase3_model_comparison_leakage_safe.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    table_path = os.path.join(table_dir, "phase3_model_comparison_leakage_safe_latex.txt")
    display_names = {
        "prompt_lr_text": "Prompt LR",
        "cumulative_lr_text": "Cumulative LR",
        "cumulative_lr_text_numeric": "Cum. LR + Num.",
        "gru_sequence": "GRU",
        "bilstm_sequence": "BiLSTM",
        "transformer_sequence": "Transformer",
    }
    lines = []
    lines.append(r"\begin{table}[!tbp]")
    lines.append(r"\centering")
    lines.append(r"\centerline{Table X. Phase 3 model comparison}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{ROC-AUC} \\ \hline")
    for m in metrics:
        lines.append(
            f"{display_names.get(m['model'], m['model'])} & {m['precision']:.2f} & {m['recall']:.2f} & {m['f1']:.2f} & {m['roc_auc']:.3f} \\\\ \\hline"
        )
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{tab:phase3_model_comparison}")
    lines.append(r"\end{table}")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    save_json(metrics, os.path.join(output_dir, "phase3_model_comparison_leakage_safe_summary.json"))
    print(f"Saved figure: {fig_path}")
    print(f"Saved table: {table_path}")


                                                                             
      
                                                                             

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="balanced", help="Dataset profile under data/profiles/")
    parser.add_argument("--input", default=None, help="Optional explicit labelled dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tfidf_features", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--max_len", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = args.input or f"data/profiles/{args.profile}/prompt_sessions_labeled.json"
    output_dir = "data/phase3_model_comparison_leakage_safe"
    fig_dir = "figures"
    table_dir = "tables"
    ensure_dir(output_dir)

    print(f"Loading: {input_path}")
    rows = load_json(input_path)
    rows = enrich_missing_numeric_features(rows)
    rows = sorted(rows, key=lambda r: (int(r[SESSION_COL]), int(r[TURN_COL])))

    if LABEL_COL not in rows[0]:
        raise KeyError(f"Missing {LABEL_COL}. Run label_leakage.py / main_phase1b.py first.")

    train_ids, val_ids, test_ids = make_session_splits(rows, seed=args.seed)
    train_rows, val_rows, test_rows = split_rows(rows, train_ids, val_ids, test_ids)

    split_summary = {
        "profile": args.profile,
        "input_path": input_path,
        "train_sessions": len(train_ids),
        "val_sessions": len(val_ids),
        "test_sessions": len(test_ids),
        "train_prefixes": len(train_rows),
        "val_prefixes": len(val_rows),
        "test_prefixes": len(test_rows),
        "train_positive_rate": float(labels(train_rows).mean()),
        "val_positive_rate": float(labels(val_rows).mean()),
        "test_positive_rate": float(labels(test_rows).mean()),
        "numeric_features": NUMERIC_FEATURES,
    }
    save_json(split_summary, os.path.join(output_dir, "split_summary.json"))
    print(json.dumps(split_summary, indent=2))

    all_metrics = []

                         
    all_metrics.append(train_logistic_baseline(
        name="prompt_lr_text",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        cumulative=False,
        use_numeric=False,
        output_dir=output_dir,
    ))
    all_metrics.append(train_logistic_baseline(
        name="cumulative_lr_text",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        cumulative=True,
        use_numeric=False,
        output_dir=output_dir,
    ))
    all_metrics.append(train_logistic_baseline(
        name="cumulative_lr_text_numeric",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        cumulative=True,
        use_numeric=True,
        output_dir=output_dir,
    ))

                                                                
    vectorizer, scaler = build_turn_feature_transformers(train_rows, max_features=args.tfidf_features)
    train_turn = make_turn_features(train_rows, vectorizer, scaler)
    val_turn = make_turn_features(val_rows, vectorizer, scaler)
    test_turn = make_turn_features(test_rows, vectorizer, scaler)

    train_pack = make_sequence_pack(train_rows, train_turn, args.max_len)
    val_pack = make_sequence_pack(val_rows, val_turn, args.max_len)
    test_pack = make_sequence_pack(test_rows, test_turn, args.max_len)
    input_dim = train_pack.X.shape[-1]
    print(f"Sequence input_dim={input_dim}, max_len={args.max_len}, device={args.device}")

    all_metrics.append(train_torch_model(
        name="gru_sequence",
        model=RNNClassifier(input_dim, args.hidden_dim, rnn_type="gru", bidirectional=False, dropout=0.25),
        train_pack=train_pack,
        val_pack=val_pack,
        test_pack=test_pack,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    ))
    all_metrics.append(train_torch_model(
        name="bilstm_sequence",
        model=RNNClassifier(input_dim, args.hidden_dim, rnn_type="lstm", bidirectional=True, dropout=0.25),
        train_pack=train_pack,
        val_pack=val_pack,
        test_pack=test_pack,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    ))
    all_metrics.append(train_torch_model(
        name="transformer_sequence",
        model=TransformerClassifier(input_dim, d_model=128, nhead=4, num_layers=2, max_len=args.max_len, dropout=0.25),
        train_pack=train_pack,
        val_pack=val_pack,
        test_pack=test_pack,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    ))

    generate_comparison_outputs(all_metrics, fig_dir=fig_dir, table_dir=table_dir, output_dir=output_dir)
    print("\nModel comparison saved.")


if __name__ == "__main__":
    main()
