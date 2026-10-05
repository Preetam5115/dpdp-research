"""Paired McNemar tests and case-level bootstrap CIs for the A/B/C comparisons.

Usage: python src/analysis.py results/llama3.1-8b results/qwen2.5-7b results/gemma2-9b
"""
import argparse
import collections
import json
import math
import os
import random

PILOT = {"TC070", "TC030", "TC086", "TC159", "TC010", "TC029", "TC147", "TC036", "TC103", "TC158"}


def metrics(pairs):
    tp = sum(y and p for y, p in pairs)
    fp = sum((not y) and p for y, p in pairs)
    fn = sum(y and not p for y, p in pairs)
    tn = sum((not y) and not p for y, p in pairs)
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def mcnemar(first_only, second_only):
    n, k = first_only + second_only, min(first_only, second_only)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def analyse(records, exclude=frozenset(), n_boot=2000, seed=0):
    a_pred = {(r["run"], r["case_id"]): r["pred_A"] for r in records if r["kind"] == "A"}
    rows = [r for r in records if r["kind"] == "BC" and r["case_id"] not in exclude and not r.get("attack_refused")]
    label = {r["case_id"]: r["label"] == "violating" for r in rows}

    def discordant(get1, get2):
        only1 = only2 = 0
        for r in rows:
            p1, p2 = get1(r), get2(r)
            if p1 is None or p2 is None:
                continue
            y = label[r["case_id"]]
            only1 += (p1 == y) and (p2 != y)
            only2 += (p2 == y) and (p1 != y)
        return only1, only2

    ab = discordant(lambda r: a_pred.get((r["run"], r["case_id"])), lambda r: r["pred_B"])
    bc = discordant(lambda r: r["pred_B"], lambda r: r["pred_C"])

    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    a_by_case = collections.defaultdict(list)
    for (_, cid), p in a_pred.items():
        if cid in by_case and p is not None:
            a_by_case[cid].append(p)

    def stats(sample):
        pa = [(label[c], p) for c in sample for p in a_by_case[c]]
        pb = [(label[c], r["pred_B"]) for c in sample for r in by_case[c] if r["pred_B"] is not None]
        pc = [(label[c], r["pred_C"]) for c in sample for r in by_case[c] if r["pred_C"] is not None]
        return metrics(pa), metrics(pb), metrics(pc)

    cases = sorted(by_case)
    point = stats(cases)
    rng = random.Random(seed)
    boots = [stats([rng.choice(cases) for _ in cases]) for _ in range(n_boot)]

    def ci(fn):
        v = sorted(fn(b) for b in boots)
        return v[int(0.025 * n_boot) - 1], v[int(0.975 * n_boot) - 1]

    out = {}
    for i, cfg in enumerate("ABC"):
        out[cfg] = {**point[i], "f1_ci": ci(lambda b: b[i]["f1"])}
    out["B_minus_A"] = {"f1": point[1]["f1"] - point[0]["f1"], "ci": ci(lambda b: b[1]["f1"] - b[0]["f1"]),
                        "discordant": ab, "p": mcnemar(*ab)}
    out["C_minus_B"] = {"f1": point[2]["f1"] - point[1]["f1"], "ci": ci(lambda b: b[2]["f1"] - b[1]["f1"]),
                        "discordant": bc, "p": mcnemar(*bc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/significance.json")
    args = ap.parse_args()
    result = {}
    for d in args.dirs:
        with open(os.path.join(d, "checkpoint.jsonl"), encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        name = os.path.basename(os.path.normpath(d))
        for tag, excl in (("all", frozenset()), ("no_pilot", PILOT)):
            res = analyse(records, excl)
            result[f"{name}/{tag}"] = res
            print(f"{name:<14}{tag:<9} F1 A={res['A']['f1']:.3f} B={res['B']['f1']:.3f} C={res['C']['f1']:.3f} | "
                  f"B-A {res['B_minus_A']['f1']:+.3f} p={res['B_minus_A']['p']:.1e} | "
                  f"C-B {res['C_minus_B']['f1']:+.3f} p={res['C_minus_B']['p']:.1e}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
