"""Shared utilities: ChromaDB access, JSON parsing, metrics, CSV logging."""

import json
import re
import os
import csv
import math
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "../data/dpdp_db"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection("dpdp_act", embedding_function=ef)


def retrieve_relevant_sections(text, n_results=3):
    collection = get_collection()
    results = collection.query(query_texts=[text], n_results=n_results)
    return "\n\n---\n\n".join(results["documents"][0])


def parse_json(raw, fallback):
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    return fallback


def mean(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def std_dev(values):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return 0.0
    m = mean(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / (len(valid) - 1))


def compute_metrics(true_labels, pred_labels):
    TP = sum(1 for t, p in zip(true_labels, pred_labels) if t == "violating" and p == "violating")
    FP = sum(1 for t, p in zip(true_labels, pred_labels) if t == "compliant"  and p == "violating")
    FN = sum(1 for t, p in zip(true_labels, pred_labels) if t == "violating" and p == "compliant")
    TN = sum(1 for t, p in zip(true_labels, pred_labels) if t == "compliant"  and p == "compliant")
    P  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    Ac = (TP + TN) / len(true_labels) if len(true_labels) > 0 else 0.0
    FPR= FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "P": P, "R": R, "F1": F1, "Acc": Ac, "FPR": FPR}


def print_confusion_matrix(metrics, label=""):
    print(f"\n{'─'*52}")
    print(f"CONFUSION MATRIX {label}")
    print(f"{'─'*52}")
    print(f"                    Pred:Violating  Pred:Compliant")
    print(f"  True: Violating   {metrics['TP']:<16}{metrics['FN']}")
    print(f"  True: Compliant   {metrics['FP']:<16}{metrics['TN']}")
    print(f"{'─'*52}")
    print(f"  Precision: {metrics['P']:.3f}  Recall: {metrics['R']:.3f}")
    print(f"  F1:        {metrics['F1']:.3f}  Accuracy: {metrics['Acc']:.3f}")
    print(f"  FP Rate:   {metrics['FPR']:.3f}")
    print(f"{'─'*52}")


def log_run_to_csv(state, temp, run, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "id", "category", "temperature", "run",
                "input_text", "true_label",
                "original_violation", "original_section",
                "adversarial_text", "adversarial_mode",
                "adversarial_violation", "adversarial_section",
                "violation_caught", "evasion_successful",
                "false_positive_triggered", "correctly_identified_as_compliant",
                "robustness_score", "judge_reasoning", "final_prediction"
            ])
        js  = state["judge_score"]
        cr  = state["compliance_result"]
        acr = state["adversarial_compliance_result"]
        mode = "hide_violation" if state["true_label"] == "violating" else "sound_suspicious"
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state["id"], state.get("category", ""), temp, run + 1,
            state["input_text"], state["true_label"],
            cr.get("violation"), cr.get("dpdp_section"),
            state["adversarial_text"], mode,
            acr.get("violation"), acr.get("dpdp_section"),
            js.get("violation_caught"), js.get("evasion_successful"),
            js.get("false_positive_triggered"), js.get("correctly_identified_as_compliant"),
            js.get("robustness_score"), js.get("judge_reasoning"),
            state.get("final_prediction")
        ])


def log_summary_to_csv(summary_rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "temperature", "model",
            "mean_precision", "std_precision",
            "mean_recall", "std_recall",
            "mean_f1", "std_f1",
            "mean_accuracy", "std_accuracy",
            "mean_fp_rate", "std_fp_rate",
            "runs"
        ])
        for row in summary_rows:
            writer.writerow(row)
    print(f"Summary saved to {filename}")
