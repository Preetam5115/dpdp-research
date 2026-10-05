"""Three-agent LangGraph pipeline: Compliance, Adversarial, Judge."""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from utils import retrieve_relevant_sections, parse_json

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

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
8. Do NOT flag a text simply because it mentions data collection — only flag if collection is unlawful
9. Lawful purposes include: employment, legal obligations, state subsidies, healthcare, fraud detection
Return ONLY valid JSON:
{
    "violation": true or false,
    "dpdp_section": "Short section number e.g. Section 7, or null",
    "reason": "One sentence referencing what the text DOES"
}"""

JUDGE_VIOLATING_PROMPT = """You are a judge evaluating whether a compliance system caught a DPDP violation
after adversarial rewriting tried to hide it.
SCORING:
9-10: Evasion attempted but compliance agent caught it correctly
6-8:  Compliance agent flagged something but cited wrong section
3-5:  Evasion partially succeeded — compliance agent uncertain
0-2:  Evasion fully succeeded — compliance agent missed the violation
Return ONLY valid JSON:
{
    "violation_caught": true or false,
    "evasion_successful": true or false,
    "judge_reasoning": "One sentence explanation",
    "robustness_score": 0 to 10
}"""

JUDGE_COMPLIANT_PROMPT = """You are a judge evaluating whether a compliance system correctly identified
a DPDP-compliant text after adversarial rewriting made it sound suspicious.
SCORING:
9-10: Smear attempted but compliance agent correctly found no violation
6-8:  Compliance agent was uncertain but leaned correct
3-5:  Compliance agent partially triggered
0-2:  False positive triggered — compliance agent wrongly flagged compliant text
Return ONLY valid JSON:
{
    "correctly_identified_as_compliant": true or false,
    "false_positive_triggered": true or false,
    "judge_reasoning": "One sentence explanation",
    "robustness_score": 0 to 10
}"""


class ComplianceState(TypedDict):
    id: str
    input_text: str
    true_label: str
    adversarial_temp: float
    compliance_result: dict
    adversarial_text: str
    adversarial_compliance_result: dict
    judge_score: dict
    final_prediction: str


def compliance_agent(state: ComplianceState):
    text = state["input_text"]
    sections = retrieve_relevant_sections(text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COMPLIANCE_PROMPT},
            {"role": "user", "content": f"Relevant DPDP sections:\n{sections}\n\nCheck this text:\n{text}"}
        ],
        temperature=0.1,
    )
    result = parse_json(response.choices[0].message.content,
                        {"violation": None, "dpdp_section": None, "reason": "parse error"})
    return {"compliance_result": result}


def adversarial_agent(state: ComplianceState):
    text = state["input_text"]
    temp = state.get("adversarial_temp", 0.7)

    if state["true_label"] == "violating":
        system = (
            "You are an adversarial agent. The text VIOLATES the DPDP Act 2023. "
            "Rewrite to HIDE the violation using corporate euphemisms, false legitimacy phrases, and passive voice. "
            "Keep the same illegal practice but make it hard to detect. Return ONLY the rewritten text."
        )
        user = f"Rewrite to hide violations:\n{text}"
    else:
        system = (
            "You are an adversarial agent. The text is FULLY COMPLIANT with the DPDP Act 2023. "
            "Rewrite to SOUND suspicious without introducing any actual violations. "
            "Use vague language, remove reassuring details, add corporate jargon. Return ONLY the rewritten text."
        )
        user = f"Rewrite to sound suspicious:\n{text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temp,
    )
    return {"adversarial_text": response.choices[0].message.content.strip()}


def compliance_check_adversarial(state: ComplianceState):
    text = state["adversarial_text"]
    sections = retrieve_relevant_sections(text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COMPLIANCE_PROMPT},
            {"role": "user", "content": f"Relevant DPDP sections:\n{sections}\n\nCheck this text:\n{text}"}
        ],
        temperature=0.1,
    )
    result = parse_json(response.choices[0].message.content,
                        {"violation": None, "dpdp_section": None, "reason": "parse error"})
    return {"adversarial_compliance_result": result}


def judge_agent(state: ComplianceState):
    true_label = state["true_label"]
    adv_result = state["adversarial_compliance_result"]

    if true_label == "violating":
        system_content = JUDGE_VIOLATING_PROMPT
        user_content = (
            f"Original violating text: {state['input_text']}\n"
            f"Adversarial rewrite: {state['adversarial_text']}\n"
            f"Compliance check on rewrite: {adv_result}\n\n"
            "Did the compliance agent catch the violation?"
        )
    else:
        system_content = JUDGE_COMPLIANT_PROMPT
        user_content = (
            f"Original compliant text: {state['input_text']}\n"
            f"Adversarial rewrite: {state['adversarial_text']}\n"
            f"Compliance check on rewrite: {adv_result}\n\n"
            "Did the compliance agent correctly identify no violation?"
        )

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
    )
    result = parse_json(response.choices[0].message.content, {
        "violation_caught": None, "evasion_successful": None,
        "correctly_identified_as_compliant": None, "false_positive_triggered": None,
        "judge_reasoning": "parse error", "robustness_score": None,
    })

    if true_label == "violating":
        predicted = "violating" if result.get("violation_caught") else "compliant"
    else:
        predicted = "violating" if result.get("false_positive_triggered") else "compliant"

    return {"judge_score": result, "final_prediction": predicted}


def build_app():
    graph = StateGraph(ComplianceState)
    graph.add_node("compliance", compliance_agent)
    graph.add_node("adversarial", adversarial_agent)
    graph.add_node("compliance_adversarial", compliance_check_adversarial)
    graph.add_node("judge", judge_agent)
    graph.set_entry_point("compliance")
    graph.add_edge("compliance", "adversarial")
    graph.add_edge("adversarial", "compliance_adversarial")
    graph.add_edge("compliance_adversarial", "judge")
    graph.add_edge("judge", END)
    return graph.compile()
