import json
import re
import sys

CATEGORY_MAP = {"children data": "Children's Data", "children's data": "Children's Data",
                "rights of data principal": "Data Principal Rights", "data principal rights": "Data Principal Rights",
                "sensitive personal data": "Sensitive Data", "data sharing": "Data Sharing / Disclosure",
                "profiling": "Profiling / Behavioral Advertising"}


def load_cases(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for c in data:
        label = str(c.get("type", c.get("label", ""))).lower()
        label = {"violation": "violating", "violating": "violating", "compliant": "compliant"}.get(label)
        if label is None or not c.get("text"):
            sys.exit(f"bad case {c.get('id')}: needs 'text' and 'type' violating/compliant")
        cat = (c.get("category") or "").strip()
        cases.append({"id": c["id"], "text": c["text"], "label": label, "gold_section": c.get("dpdp_section"),
                      "category": CATEGORY_MAP.get(cat.lower(), cat)})
    assert len({c["id"] for c in cases}) == len(cases), "duplicate ids"
    n_vio = sum(c["label"] == "violating" for c in cases)
    print(f"[data] {len(cases)} cases: {n_vio} violating, {len(cases) - n_vio} compliant", flush=True)
    return cases


def section_numbers(s):
    if not s:
        return set()
    return {(m.group(1)[0].lower(), m.group(2)) for m in re.finditer(r"(section|rule)\s*(\d+)", str(s), re.I)}
