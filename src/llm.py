import json
import os
import re
import sys
import time

import requests


class LLM:
    def __init__(self, backend, model, base_url=None, max_tokens=400, timeout=300):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        if backend == "ollama":
            self.url = (base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
        elif backend == "groq":
            self.url = (base_url or "https://api.groq.com/openai/v1").rstrip("/") + "/chat/completions"
            self.key = os.environ.get("GROQ_API_KEY") or sys.exit("Set GROQ_API_KEY")
        else:
            sys.exit(f"unknown backend {backend}")

    def chat(self, system, user, temperature, seed):
        for attempt in range(8):
            try:
                return self._chat(system, user, temperature, seed)
            except Exception as e:
                wait = min(120, 5 * 2 ** attempt)
                print(f"    [llm] {str(e)[:120]} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("LLM call failed repeatedly (progress is saved, re-run to resume)")

    def _chat(self, system, user, temperature, seed):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "ollama":
            r = requests.post(self.url, timeout=self.timeout, json={
                "model": self.model, "stream": False, "messages": msgs,
                "options": {"temperature": temperature, "seed": seed, "num_predict": self.max_tokens}})
            r.raise_for_status()
            return r.json()["message"]["content"]
        r = requests.post(self.url, timeout=self.timeout, headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "messages": msgs, "temperature": temperature,
                                "max_tokens": self.max_tokens})
        if r.status_code == 429:
            raise RuntimeError("rate limited (429)")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def parse_verdict(raw):
    """Return the first JSON object with a boolean 'violation' field, or None."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    for cand in [raw] + re.findall(r"\{.*\}", raw, re.S) + re.findall(r"\{.*?\}", raw, re.S):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "violation" in obj:
            v = obj["violation"]
            if isinstance(v, str):
                v = {"true": True, "yes": True, "false": False, "no": False}.get(v.strip().lower())
            if isinstance(v, bool):
                obj["violation"] = v
                return obj
    return None


def ask(llm, system, user, temperature, seed, tries=3):
    raw = ""
    for i in range(tries):
        raw = llm.chat(system, user, temperature, seed + 7919 * i)
        obj = parse_verdict(raw)
        if obj is not None:
            return obj, raw, i + 1
    return None, raw, tries
