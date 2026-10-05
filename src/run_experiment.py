"""Bidirectional adversarial stress test of the DPDP compliance pipeline.

Configurations
  A  Compliance Agent on the original text
  B  Compliance Agent on the adversarial rewrite
  C  Judge Agent on the same rewrite, given the Compliance Agent's verdict

Example
  python src/run_experiment.py --model llama3.1:8b --attacker-model qwen2.5:7b \\
      --temps 0.5 0.7 0.9 --runs 3 --workers 4 --out results/llama3.1-8b
"""
import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from data import load_cases
from knowledge_base import KnowledgeBase
from llm import LLM
from pipeline import build_graph, check_label_blind, compliance_check
from prompts import COMPLIANCE_TEMP, JUDGE_TEMP, TOP_K
from report import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = {"ollama": "llama3.1:8b", "groq": "llama-3.1-8b-instant"}


def seed_for(case_id, run, extra=""):
    return int(hashlib.md5(f"{case_id}|{run}|{extra}".encode()).hexdigest()[:8], 16) % 2_000_000_000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "test_cases.json"))
    ap.add_argument("--pdfs", nargs="*", default=None, help="knowledge-base PDFs (default: all PDFs in data/)")
    ap.add_argument("--db-dir", default=os.path.join(ROOT, "data", "dpdp_db"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "run"))
    ap.add_argument("--backend", default="ollama", choices=["ollama", "groq"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--attacker-model", default=None, help="model for the Adversarial Agent (default: --model)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--exclude-ids", nargs="*", default=[], help="case ids to leave out (e.g. pilot cases)")
    ap.add_argument("--sample", type=int, default=None, help="quick test on N cases, half violating / half compliant")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    model = args.model or DEFAULT_MODEL[args.backend]
    attacker_model = args.attacker_model or model

    check_label_blind()
    cases = load_cases(args.cases)[: args.limit]
    if args.exclude_ids:
        cases = [c for c in cases if c["id"] not in set(args.exclude_ids)]
        print(f"[data] excluded {len(args.exclude_ids)} cases -> {len(cases)} cases", flush=True)
    if args.sample:
        rng = random.Random(7)
        vio = [c for c in cases if c["label"] == "violating"]
        com = [c for c in cases if c["label"] == "compliant"]
        cases = rng.sample(vio, (args.sample + 1) // 2) + rng.sample(com, args.sample // 2)
        print(f"[data] sample: {[c['id'] for c in cases]}", flush=True)

    data_dir = os.path.join(ROOT, "data")
    pdfs = args.pdfs or sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf"))
    kb = KnowledgeBase(pdfs, args.db_dir)

    llm = LLM(args.backend, model, args.base_url)
    attacker = llm if attacker_model == model else LLM(args.backend, attacker_model, args.base_url)
    print(f"[models] system under test: {model} | attacker: {attacker_model}", flush=True)
    graph = build_graph(llm, kb, attacker)

    os.makedirs(args.out, exist_ok=True)
    ck_path = os.path.join(args.out, "checkpoint.jsonl")
    done = {}
    if os.path.exists(ck_path):
        with open(ck_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[(r["kind"], r["temperature"], r["run"], r["case_id"])] = r
        print(f"[resume] {len(done)} records already done", flush=True)

    with open(os.path.join(args.out, "run_metadata.json"), "w") as f:
        json.dump({"model": model, "attacker_model": attacker_model, "backend": args.backend, "temps": args.temps,
                   "runs": args.runs, "n_cases": len(cases), "kb_sources": kb.sources,
                   "compliance_temp": COMPLIANCE_TEMP, "judge_temp": JUDGE_TEMP, "top_k": TOP_K,
                   "judge": "label-blind", "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    def run_case(kind, t, run, c):
        seed = seed_for(c["id"], run, t)
        rec = {"kind": kind, "temperature": t, "run": run, "case_id": c["id"], "label": c["label"],
               "category": c["category"], "gold_section": c["gold_section"], "seed": seed,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        if kind == "A":
            v, raw, n = compliance_check(llm, kb, c["text"], seed)
            rec.update(pred_A=None if v is None else v["violation"], A=v, A_attempts=n,
                       A_raw=raw if v is None else None)
            return rec
        s = graph.invoke({"h_label": c["label"], "h_original_text": c["text"], "h_temperature": t, "h_seed": seed})
        cv, jv = s.get("checker_verdict"), s.get("judge_verdict")
        rec.update(attack_refused=s["attack_refused"], adv_attempts=s["adv_attempts"],
                   adv_refusals=s["adv_refusals"], rewrite=s["audit_text"], adv_leak_flag=s["adv_leak_flag"],
                   adv_preamble_stripped=s["adv_preamble_stripped"],
                   B=cv, pred_B=None if cv is None else cv["violation"],
                   C=jv, pred_C=None if jv is None else jv["violation"],
                   judge_confidence=(jv or {}).get("confidence"), judge_prompt_hash=s["judge_prompt_hash"])
        return rec

    # Config A does not depend on the attacker temperature, so it runs once per run.
    plan = [("A", None, r) for r in range(1, args.runs + 1)]
    plan += [("BC", t, r) for t in args.temps for r in range(1, args.runs + 1)]
    lock = threading.Lock()
    with open(ck_path, "a", encoding="utf-8") as ck:
        for kind, t, run in plan:
            todo = [c for c in cases if (kind, t, run, c["id"]) not in done]
            name = "Config A (original text)" if kind == "A" else "Configs B+C"
            print(f"\n=== {name} | temp {t} | run {run} | {len(cases) - len(todo)}/{len(cases)} done ===", flush=True)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(run_case, kind, t, run, c) for c in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    rec = fut.result()
                    with lock:
                        ck.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        ck.flush()
                        done[(kind, t, run, rec["case_id"])] = rec
                    if i % 10 == 0 or i == len(todo):
                        rate = (time.time() - t0) / i
                        print(f"  {i}/{len(todo)}  {rate:.1f}s/case  ~{rate * (len(todo) - i) / 60:.0f} min left",
                              flush=True)

    report(list(done.values()), cases, args.out, f"{model} (attacker: {attacker_model})")


if __name__ == "__main__":
    main()
