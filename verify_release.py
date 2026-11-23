import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "realistic_mixture": (5000, 27556, 0.099),
    "balanced": (5000, 30783, 0.236),
    "risk_enriched": (5000, 34452, 0.5378),
}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    for profile, (sessions, prompts, rate) in EXPECTED.items():
        stats = load(f"data/profiles/{profile}/dataset_statistics.json")
        assert stats["num_sessions"] == sessions
        assert stats["num_prompts"] == prompts
        assert abs(stats["attack_ready_session_rate"] - rate) < 1e-9
        records = load(f"data/profiles/{profile}/prompt_sessions_labeled.json")
        assert len(records) == prompts
        assert len({row["session_id"] for row in records}) == sessions
    summary = load("data/multiseed_leakage_safe/multiseed_statistics.json")
    expected_f1 = {
        "prompt_lr_text": 0.515,
        "cumulative_lr_text": 0.905,
        "cumulative_lr_text_numeric": 0.936,
        "gru_sequence": 0.935,
        "bilstm_sequence": 0.933,
        "transformer_sequence": 0.912,
    }
    for row in summary["summary"]:
        assert abs(row["f1_mean"] - expected_f1[row["model"]]) < 0.0006
    validation = load("data/validation_reference/expert_validation_results.json")
    assert validation["n_cases"] == 150
    assert validation["n_experts"] == 3
    assert round(validation["framework_vs_expert_majority"]["f1"], 3) == 0.813
    print("Data and reported metrics checked.")


if __name__ == "__main__":
    main()
