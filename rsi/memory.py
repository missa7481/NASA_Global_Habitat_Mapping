#!/usr/bin/env python3
"""Structured research memory for the RSI loop.

ledger.jsonl   one JSON object per experiment (append-only; corrections are new entries with 'supersedes')
RESEARCH_MEMORY.md  rendered view, regenerated from the ledger:  python rsi/memory.py render
Fields follow the RSI protocol: question, hypothesis, data, representation, fusion, architecture, weighting,
validation, experiment, results, error_patterns, uncertainty, ecological, causal, improved, failed,
open_questions, next, status (baseline-candidate | rejected | inconclusive | infrastructure), level (Data / Representation /
Fusion / Spatial-Model / Validation / Scientific RSI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MEM = Path(__file__).resolve().parent / "memory"
LEDGER = MEM / "ledger.jsonl"
FIELDS = ["question", "hypothesis", "data", "representation", "fusion", "architecture", "weighting", "validation",
          "experiment", "results", "error_patterns", "uncertainty", "ecological", "causal", "improved", "failed",
          "open_questions", "next"]


def load():
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]


def add(entry: dict):
    MEM.mkdir(exist_ok=True)
    ids = {e["id"] for e in load()}
    if entry["id"] in ids:
        raise ValueError(f"duplicate id {entry['id']}; add a new entry with 'supersedes'")
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def render():
    es = load()
    out = ["# RSI research memory", "", "Generated from `ledger.jsonl`; do not edit by hand.", ""]
    out += ["| id | iteration | level | title | status |", "|---|---|---|---|---|"]
    out += [f"| {e['id']} | {e['iteration']} | {e['level']} | {e['title']} | {e['status']} |" for e in es]
    for e in es:
        out += ["", f"## {e['id']} - {e['title']}", "",
                f"*Iteration {e['iteration']} - {e['level']} - status: **{e['status']}***"
                + (f" - supersedes {e['supersedes']}" if e.get("supersedes") else ""), ""]
        for k in FIELDS:
            v = e.get(k)
            if v:
                out.append(f"- **{k.replace('_', ' ')}:** {v}")
    (MEM / "RESEARCH_MEMORY.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["render"]:
        render()
