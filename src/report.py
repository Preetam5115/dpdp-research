import csv
import json
import os
import random
import statistics

from data import section_numbers
from prompts import MAX_ATTACK_ATTEMPTS

METRICS = ["precision", "recall", "f1", "accuracy", "fpr"]


def confusion(pairs):
    tp = fp = fn = tn = err = 0
    for y, p in pairs:
        if p is None:
            err += 1; continue
        if y == "violating":
            tp += p; fn += not p
        else:
            fp += p; tn += not p
    return tp, fp, fn, tn, err


def scores(tp, fp, fn, tn):
    d = lambda a, b: a / b if b else float("nan")
    p, r = d(tp, tp + fp), d(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": d(2 * p * r, p + r) if p == p and r == r else float("nan"),
            "accuracy": d(tp + tn, tp + fp + fn + tn), "fpr": d(fp, fp + tn)}


def msd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def report(records, cases, out, model):
    ids = {c["id"] for c in cases}
    n = len(cases)
    recs = [r for r in records if r["case_id"] in ids]
    groups = {}
    for r in recs:
        for cfg in (["A"] if r["kind"] == "A" else ["B", "C"]):
            key = (cfg, None if cfg == "A" else r["temperature"])
            groups.setdefault(key, {}).setdefault(r["run"], []).append((r, r[f"pred_{cfg}"], r.get(cfg)))

    L = [f"Model: {model} | cases: {n} | Judge: label-blind | only COMPLETE runs ({n} cases) are aggregated", ""]
    L.append(f"{'Cfg':<4}{'Temp':<6}{'Runs':<5}" + "".join(f"{m:<17}" for m in ["Precision", "Recall", "F1", "Accuracy", "FPR"])
             + "SectionAcc   TP/FP/FN/TN[/parse-errors][/refused attacks, excluded] per run")
    summary = []
    for (cfg, t), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        complete = {k: v for k, v in runs.items() if len(v) == n}
        per = []
        for k in sorted(complete):
            refused = sum(bool(r.get("attack_refused")) for r, _, _ in complete[k])
            rows = [x for x in complete[k] if not x[0].get("attack_refused")]
            tp, fp, fn, tn, err = confusion([(r["label"], p) for r, p, _ in rows])
            hits = [bool(section_numbers(v.get("dpdp_section")) & section_numbers(r["gold_section"]))
                    for r, p, v in rows if r["label"] == "violating" and p and v]
            per.append({"run": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "parse_errors": err, "attack_refused": refused,
                        "section_acc": (sum(hits) / len(hits)) if hits else float("nan"), **scores(tp, fp, fn, tn)})
        agg = {m: msd([p[m] for p in per]) for m in METRICS + ["section_acc"]}
        f = lambda ms: f"{ms[0]:.3f}±{ms[1]:.3f}" if ms[0] == ms[0] else "n/a"
        L.append(f"{cfg:<4}{str(t if t is not None else '-'):<6}{len(per):<5}" + "".join(f"{f(agg[m]):<17}" for m in METRICS)
                 + f"{f(agg['section_acc']):<13}" + "; ".join(
                     f"{p['tp']}/{p['fp']}/{p['fn']}/{p['tn']}" + (f"/{p['parse_errors']}err" if p["parse_errors"] else "")
                     + (f"/{p['attack_refused']}refused" if p["attack_refused"] else "")
                     for p in per))
        inc = {k: len(v) for k, v in runs.items() if len(v) != n}
        if inc:
            L.append(f"          incomplete runs not aggregated: {inc}")
        summary.append({"config": cfg, "temperature": t, "runs": per,
                        "mean": {m: agg[m][0] for m in agg}, "sd": {m: agg[m][1] for m in agg}})

    for cfg in ("B", "C"):
        allruns = [p for s in summary if s["config"] == cfg for p in s["runs"]]
        if allruns:
            L.append(f"\nConfig {cfg} pooled over all temperatures/runs: " +
                     ", ".join(f"{m}={statistics.mean([p[m] for p in allruns]):.3f}" for m in METRICS))

    bc = [r for r in recs if r["kind"] == "BC"]
    if bc:
        ref = [r for r in bc if r.get("attack_refused")]
        L.append(f"\nAttacks refused by the Adversarial Agent after {MAX_ATTACK_ATTEMPTS} attempts: {len(ref)}/{len(bc)} "
                 f"(violating: {sum(r['label'] == 'violating' for r in ref)}, compliant: {sum(r['label'] == 'compliant' for r in ref)}); "
                 f"rewrites needing a retry: {sum((r.get('adv_refusals') or 0) > 0 for r in bc)}")
        L.append(f"Adversarial rewrites: {len(bc)} | instruction-echo (possible label leak) flagged: "
                 f"{sum(r['adv_leak_flag'] for r in bc)} | preambles stripped: {sum(r['adv_preamble_stripped'] for r in bc)}")
        flips = {"fixed_FN": 0, "fixed_FP": 0, "broke_TP": 0, "broke_TN": 0}
        for r in bc:
            b, c, y = r["pred_B"], r["pred_C"], r["label"] == "violating"
            if b is None or c is None or b == c:
                continue
            flips["fixed_FN" if (y and c) else "broke_TP" if y else "fixed_FP" if not c else "broke_TN"] += 1
        L.append(f"Judge overturned the checker: {flips}")

    L.append("\nDetection rate of violating cases by category (all complete runs pooled):")
    for (cfg, t), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        cat = {}
        for v in runs.values():
            if len(v) == n:
                for r, p, _ in v:
                    if r["label"] == "violating" and p is not None:
                        d = cat.setdefault(r["category"], [0, 0]); d[0] += p; d[1] += 1
        L.append(f"  {cfg} t={t}: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(cat.items())))

    text = "\n".join(L)
    print("\n" + text)
    open(os.path.join(out, "summary.txt"), "w", encoding="utf-8").write(text + "\n")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    with open(os.path.join(out, "per_case_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "temperature", "run", "case_id", "category", "true_label", "gold_section",
                    "pred_A", "pred_B_no_judge", "pred_C_blind_judge", "checker_section", "judge_section",
                    "judge_confidence", "adv_leak_flag", "rewrite", "checker_reason", "judge_reasoning"])
        lab = lambda p: "PARSE_ERROR" if p is None else ("violating" if p else "compliant")
        for r in sorted(recs, key=lambda r: (r["kind"], r["temperature"] or 0, r["run"], r["case_id"])):
            if r["kind"] == "A":
                w.writerow(["A", "", r["run"], r["case_id"], r["category"], r["label"], r["gold_section"],
                            lab(r["pred_A"]), "", "", (r["A"] or {}).get("dpdp_section"), "", "", "", "",
                            (r["A"] or {}).get("reason"), ""])
            else:
                refusedrow = r.get("attack_refused")
                w.writerow(["BC", r["temperature"], r["run"], r["case_id"], r["category"], r["label"], r["gold_section"],
                            "", "ATTACK_REFUSED" if refusedrow else lab(r["pred_B"]),
                            "ATTACK_REFUSED" if refusedrow else lab(r["pred_C"]), (r["B"] or {}).get("dpdp_section"),
                            (r["C"] or {}).get("dpdp_section"), r["judge_confidence"], r["adv_leak_flag"], r["rewrite"],
                            (r["B"] or {}).get("reason"), (r["C"] or {}).get("judge_reasoning")])

    text_by_id = {c["id"]: c["text"] for c in cases}
    sample = [r for r in recs if r["kind"] == "BC" and not r.get("attack_refused")]
    random.Random(42).shuffle(sample)
    with open(os.path.join(out, "audit_sample.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "temperature", "run", "true_label", "original_text", "rewrite",
                    "annotator1_label_preserved(Y/N)", "annotator2_label_preserved(Y/N)", "notes"])
        for r in sample[:50]:
            w.writerow([r["case_id"], r["temperature"], r["run"], r["label"], text_by_id[r["case_id"]], r["rewrite"], "", "", ""])
    print(f"\nWrote summary.txt, summary.json, per_case_results.csv, audit_sample.csv to {out}")
