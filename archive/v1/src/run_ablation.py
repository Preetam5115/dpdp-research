"""
Ablation study across three pipeline configurations:
  Config A — Compliance Agent only (no adversarial, no judge)
  Config B — Compliance + Adversarial Agent (no judge)
  Config C — Full 3-agent system (use results from run_experiment.py)

Runs 1 trial per temperature per config (sufficient for ablation).
Results saved to ../results/
"""

import json
import os
from groq import Groq
from utils import (
    retrieve_relevant_sections, parse_json,
    compute_metrics, print_confusion_matrix, mean, std_dev
)
import csv
from datetime import datetime

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
TEMPERATURES = [0.5, 0.7, 0.9]

with open("../data/test_cases.json") as f:
    TEST_CASES = json.load(f)

COMPLIANCE_PROMPT = """You are a DPDP Act 2023 compliance checker.
STRICT RULES:
1. Only flag violations EXPLICITLY present in the given text
2. Absence of information is NOT a violation
3. Only return true if there is clear evidence of non-compliance
4. Claims of regulatory compliance without specified consent mechanisms are red flags
VERSION 5 ANTI-OVER-FLAGGING RULES:
5. Formal or technical protective language is NOT automatically suspicious
6. Texts mentioning AES-256, TLS, ISO certification, DigiLocker, RBI KYC, employment law,
   Income Tax Act, or other legal frameworks describe PROTECTION — not a violation
7. A text that explicitly states data is deleted, not shared, or used for a stated purpose is COMPLIANT
8. Lawful purposes include: employment, legal obligations, state subsidies, healthcare, fraud detection
Return ONLY valid JSON:
{
    "violation": true or false,
    "dpdp_section": "Short section number e.g. Section 7, or null",
    "reason": "One sentence referencing what the text DOES"
}"""


def check_compliance(text):
    sections = retrieve_relevant_sections(text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COMPLIANCE_PROMPT},
            {"role": "user", "content": f"Relevant DPDP sections:\n{sections}\n\nCheck this text:\n{text}"}
        ],
        temperature=0.1,
    )
    return parse_json(response.choices[0].message.content,
                      {"violation": None, "dpdp_section": None, "reason": "parse error"})


def adversarial_rewrite(text, true_label, temp):
    if true_label == "violating":
        system = (
            "You are an adversarial agent. The text VIOLATES the DPDP Act 2023. "
            "Rewrite to HIDE the violation using corporate euphemisms and passive voice. "
            "Return ONLY the rewritten text."
        )
        user = f"Rewrite to hide violations:\n{text}"
    else:
        system = (
            "You are an adversarial agent. The text is FULLY COMPLIANT with the DPDP Act 2023. "
            "Rewrite to SOUND suspicious without introducing actual violations. "
            "Return ONLY the rewritten text."
        )
        user = f"Rewrite to sound suspicious:\n{text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
    )
    return response.choices[0].message.content.strip()


def log_to_csv(results, config, temp, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "config", "temperature", "id",
                             "true_label", "adversarial_text", "predicted",
                             "violation_flag", "dpdp_section", "reason"])
        for r in results:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                config, temp, r["id"], r["true"],
                r.get("adversarial_text", ""),
                r["pred"],
                r["result"].get("violation"),
                r["result"].get("dpdp_section"),
                r["result"].get("reason")
            ])


def run_config_a():
    print(f"\n{'='*60}")
    print("CONFIG A — Compliance Agent Only")
    print(f"{'='*60}")
    summary = []

    for temp in TEMPERATURES:
        print(f"\nTemperature: {temp}")
        true_labels, pred_labels, run_results = [], [], []

        for tc in TEST_CASES:
            result  = check_compliance(tc["text"])
            predicted = "violating" if result.get("violation") else "compliant"
            true_labels.append(tc["type"])
            pred_labels.append(predicted)
            run_results.append({"id": tc["id"], "true": tc["type"],
                                 "pred": predicted, "result": result})
            print(f"  [{tc['id']}] True: {tc['type']:<10} Pred: {predicted}")

        metrics = compute_metrics(true_labels, pred_labels)
        log_to_csv(run_results, "A", temp, "../results/ablation_configA.csv")
        print_confusion_matrix(metrics, f"Config A, Temp {temp}")
        summary.append({"temp": temp, "metrics": metrics})

    return summary


def run_config_b():
    print(f"\n{'='*60}")
    print("CONFIG B — Compliance + Adversarial (no Judge)")
    print(f"{'='*60}")
    summary = []

    for temp in TEMPERATURES:
        print(f"\nTemperature: {temp}")
        true_labels, pred_labels, run_results = [], [], []

        for tc in TEST_CASES:
            adv_text  = adversarial_rewrite(tc["text"], tc["type"], temp)
            result    = check_compliance(adv_text)
            predicted = "violating" if result.get("violation") else "compliant"
            true_labels.append(tc["type"])
            pred_labels.append(predicted)
            run_results.append({"id": tc["id"], "true": tc["type"],
                                 "adversarial_text": adv_text,
                                 "pred": predicted, "result": result})
            print(f"  [{tc['id']}] True: {tc['type']:<10} Pred: {predicted}")

        metrics = compute_metrics(true_labels, pred_labels)
        log_to_csv(run_results, "B", temp, "../results/ablation_configB.csv")
        print_confusion_matrix(metrics, f"Config B, Temp {temp}")
        summary.append({"temp": temp, "metrics": metrics})

    return summary


def print_ablation_table(summary_a, summary_b):
    print(f"\n{'='*72}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*72}")
    print(f"{'Config':<40} {'P':<8} {'R':<8} {'F1':<8} {'Acc':<8} {'FPR'}")
    print("-" * 72)

    def avg(summary, key):
        return mean([s["metrics"][key] for s in summary])

    print(f"{'A: Compliance only (mean across temps)':<40} "
          f"{avg(summary_a,'P'):.3f}    {avg(summary_a,'R'):.3f}    "
          f"{avg(summary_a,'F1'):.3f}    {avg(summary_a,'Acc'):.3f}    "
          f"{avg(summary_a,'FPR'):.3f}")
    print(f"{'B: +Adversarial, no judge (mean across temps)':<40} "
          f"{avg(summary_b,'P'):.3f}    {avg(summary_b,'R'):.3f}    "
          f"{avg(summary_b,'F1'):.3f}    {avg(summary_b,'Acc'):.3f}    "
          f"{avg(summary_b,'FPR'):.3f}")
    print(f"{'C: Full 3-agent system (from paper)':<40} "
          f"0.572    0.695    0.628    0.535    0.666")


if __name__ == "__main__":
    summary_a = run_config_a()
    summary_b = run_config_b()
    print_ablation_table(summary_a, summary_b)
    print(f"\nLogs saved to ../results/ablation_configA.csv and ablation_configB.csv")
