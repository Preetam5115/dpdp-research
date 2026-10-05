"""
Main experiment: bidirectional adversarial stress-testing across temperature sweep.
Runs NUM_RUNS independent trials per temperature on all 200 test cases.
Results saved to ../results/
"""

import json
import sys
import os
from agents import build_app
from utils import (
    compute_metrics, print_confusion_matrix,
    log_run_to_csv, log_summary_to_csv, mean, std_dev
)

NUM_RUNS   = 3
TEMPERATURES = [0.5, 0.7, 0.9]
MODEL_NAME = "llama-3.1-8b-instant"

with open("../data/test_cases.json") as f:
    TEST_CASES = json.load(f)


def run(temperatures=TEMPERATURES, num_runs=NUM_RUNS):
    app = build_app()
    total_calls = len(temperatures) * num_runs * len(TEST_CASES)

    print(f"\n{'='*60}")
    print("FULL EXPERIMENT — BIDIRECTIONAL ADVERSARIAL TESTING")
    print(f"{'='*60}")
    print(f"Model:           {MODEL_NAME}")
    print(f"Total cases:     {len(TEST_CASES)}")
    print(f"Temperatures:    {temperatures}")
    print(f"Runs per temp:   {num_runs}")
    print(f"Total API calls: {total_calls}")
    print(f"{'='*60}")

    summary_rows = []

    for temp in temperatures:
        print(f"\n{'#'*60}")
        print(f"TEMPERATURE: {temp}")
        print(f"{'#'*60}")

        run_metrics_list = []
        results_file = f"../results/results_temp{str(temp).replace('.','')}.csv"

        for run in range(num_runs):
            print(f"\n--- Run {run+1}/{num_runs} ---")
            true_labels, pred_labels = [], []

            for tc in TEST_CASES:
                print(f"  [{tc['id']}] {tc['type'].upper()} | {tc['text'][:55]}...")

                result = app.invoke({
                    "id": tc["id"],
                    "input_text": tc["text"],
                    "true_label": tc["type"],
                    "adversarial_temp": temp,
                    "compliance_result": {},
                    "adversarial_text": "",
                    "adversarial_compliance_result": {},
                    "judge_score": {},
                    "final_prediction": "",
                })
                result["category"] = tc.get("category", "")
                log_run_to_csv(result, temp, run, results_file)

                true_labels.append(tc["type"])
                pred_labels.append(result.get("final_prediction", "unknown"))

                score = result["judge_score"].get("robustness_score", "?")
                pred  = result.get("final_prediction", "?")
                print(f"    → Predicted: {pred} | True: {tc['type']} | Score: {score}/10")

            metrics = compute_metrics(true_labels, pred_labels)
            run_metrics_list.append(metrics)
            print(f"\n  Run {run+1} | P={metrics['P']:.3f} R={metrics['R']:.3f} "
                  f"F1={metrics['F1']:.3f} Acc={metrics['Acc']:.3f}")

        precisions = [m["P"]   for m in run_metrics_list]
        recalls    = [m["R"]   for m in run_metrics_list]
        f1s        = [m["F1"]  for m in run_metrics_list]
        accuracies = [m["Acc"] for m in run_metrics_list]
        fp_rates   = [m["FPR"] for m in run_metrics_list]

        print(f"\n{'='*60}")
        print(f"TEMPERATURE {temp} — {num_runs} RUNS SUMMARY")
        print(f"{'='*60}")
        print(f"Precision:  {mean(precisions):.3f} ± {std_dev(precisions):.3f}")
        print(f"Recall:     {mean(recalls):.3f} ± {std_dev(recalls):.3f}")
        print(f"F1-Score:   {mean(f1s):.3f} ± {std_dev(f1s):.3f}")
        print(f"Accuracy:   {mean(accuracies):.3f} ± {std_dev(accuracies):.3f}")
        print(f"FP Rate:    {mean(fp_rates):.3f} ± {std_dev(fp_rates):.3f}")
        print_confusion_matrix(run_metrics_list[0], f"(Run 1, Temp {temp})")

        summary_rows.append([
            temp, MODEL_NAME,
            f"{mean(precisions):.3f}", f"{std_dev(precisions):.3f}",
            f"{mean(recalls):.3f}",    f"{std_dev(recalls):.3f}",
            f"{mean(f1s):.3f}",        f"{std_dev(f1s):.3f}",
            f"{mean(accuracies):.3f}", f"{std_dev(accuracies):.3f}",
            f"{mean(fp_rates):.3f}",   f"{std_dev(fp_rates):.3f}",
            num_runs
        ])

    print(f"\n{'='*60}")
    print("FINAL RESULTS ACROSS ALL TEMPERATURES")
    print(f"{'='*60}")
    print(f"{'Temp':<8} {'Precision':<20} {'Recall':<20} {'F1':<20} {'FPR'}")
    print("-" * 76)
    for row in summary_rows:
        print(f"{row[0]:<8} {row[2]+'±'+row[3]:<20} {row[4]+'±'+row[5]:<20} "
              f"{row[6]+'±'+row[7]:<20} {row[10]+'±'+row[11]}")

    log_summary_to_csv(summary_rows, "../results/summary.csv")
    print(f"\nDetailed logs: ../results/results_temp*.csv")
    print(f"Summary:       ../results/summary.csv")


if __name__ == "__main__":
    run()
