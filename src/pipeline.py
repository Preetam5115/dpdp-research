import hashlib
import inspect
import json
from typing import Optional, TypedDict

from llm import ask
from prompts import (ADV_EVASION_SYSTEM, ADV_EVASION_USER, ADV_SMEAR_SYSTEM, ADV_SMEAR_USER, COMPLIANCE_PROMPT,
                     COMPLIANCE_TEMP, JUDGE_SYSTEM, JUDGE_TEMP, JUDGE_USER, LEAK_PATTERNS, MAX_ATTACK_ATTEMPTS,
                     PREAMBLE, REFUSAL)


# System under test: these functions only ever receive the text being audited
# (and, for the judge, the checker's verdict), never the label or the original text.

def compliance_check(llm, kb, text, seed):
    user = f"Relevant DPDP sections:\n{kb.retrieve(text)}\n\nCheck this text:\n{text}"
    return ask(llm, COMPLIANCE_PROMPT, user, COMPLIANCE_TEMP, seed)


def verdict_text(v):
    if v is None:
        return "unavailable (checker output could not be parsed)"
    return json.dumps({"violation": v["violation"], "dpdp_section": v.get("dpdp_section"),
                       "reason": v.get("reason")}, ensure_ascii=False)


def build_judge_user(sections, text, checker_verdict):
    return JUDGE_USER.format(sections=sections, text=text, verdict=verdict_text(checker_verdict))


def judge(llm, kb, text, checker_verdict, seed):
    user = build_judge_user(kb.retrieve(text), text, checker_verdict)
    return ask(llm, JUDGE_SYSTEM, user, JUDGE_TEMP, seed), hashlib.sha1(user.encode()).hexdigest()[:12]


# Attacker: the only component that reads the ground-truth label.

def adversarial_rewrite(llm, text, label, temperature, seed):
    """Returns (rewrite or None if every attempt was refused, leak_flag, preamble_stripped, attempts, refusals)."""
    if label == "violating":
        system, user = ADV_EVASION_SYSTEM, ADV_EVASION_USER.format(text=text)
    else:
        system, user = ADV_SMEAR_SYSTEM, ADV_SMEAR_USER.format(text=text)
    refusals = 0
    for attempt in range(MAX_ATTACK_ATTEMPTS):
        raw = llm.chat(system, user, temperature, seed + 31337 * attempt).strip()
        cleaned = PREAMBLE.sub("", raw).strip().strip('"').strip()
        if not cleaned or REFUSAL.search(cleaned[:300]):
            refusals += 1
            continue
        return cleaned, bool(LEAK_PATTERNS.search(cleaned)), raw != cleaned, attempt + 1, refusals
    return None, False, False, MAX_ATTACK_ATTEMPTS, refusals


class State(TypedDict, total=False):
    # read by the attacker node only
    h_label: str
    h_original_text: str
    h_temperature: float
    h_seed: int
    # system under test
    audit_text: str
    adv_leak_flag: bool
    adv_preamble_stripped: bool
    adv_attempts: int
    adv_refusals: int
    attack_refused: bool
    checker_verdict: Optional[dict]
    checker_attempts: int
    judge_verdict: Optional[dict]
    judge_attempts: int
    judge_prompt_hash: str


def build_graph(llm, kb, attacker=None):
    from langgraph.graph import END, StateGraph

    attacker = attacker or llm

    def n_adversarial(s):
        text, leak, pre, att, ref = adversarial_rewrite(attacker, s["h_original_text"], s["h_label"],
                                                         s["h_temperature"], s["h_seed"])
        return {"audit_text": text or "", "adv_leak_flag": leak, "adv_preamble_stripped": pre,
                "adv_attempts": att, "adv_refusals": ref, "attack_refused": text is None,
                "checker_verdict": None, "judge_verdict": None, "judge_prompt_hash": ""}

    def n_checker(s):
        v, _, n = compliance_check(llm, kb, s["audit_text"], s["h_seed"] + 101)
        return {"checker_verdict": v, "checker_attempts": n}

    def n_judge(s):
        (v, _, n), h = judge(llm, kb, s["audit_text"], s["checker_verdict"], s["h_seed"] + 202)
        return {"judge_verdict": v, "judge_attempts": n, "judge_prompt_hash": h}

    g = StateGraph(State)
    g.add_node("adversarial", n_adversarial)
    g.add_node("compliance_check", n_checker)
    g.add_node("judge", n_judge)
    g.set_entry_point("adversarial")
    g.add_conditional_edges("adversarial", lambda s: "end" if s["attack_refused"] else "check",
                            {"end": END, "check": "compliance_check"})
    g.add_edge("compliance_check", "judge")
    g.add_edge("judge", END)
    return g.compile()


def check_label_blind():
    src = inspect.getsource(judge) + inspect.getsource(build_judge_user) + inspect.getsource(compliance_check)
    for name in ("h_label", "h_original_text", "type", "category", "expected", "true_label"):
        assert name not in src, f"system component references '{name}'"
    for word in ("violating text", "compliant text", "true_label", "ground truth", "smear", "evasion",
                 "adversarial", "rewrite"):
        assert word not in (JUDGE_SYSTEM + JUDGE_USER).lower(), f"judge prompt contains label cue: {word}"
    print("[self-test] judge is label-blind: OK", flush=True)
