"""Spike for #16: does the local model call the source tools reliably?

Asks the configured model (LOCAL_AI_MODEL, default qwen3.5:9b) to plan the
sources for 12 research questions across disciplines, exactly as Discovery
does, and counts per prompt: a valid plan (every call a known tool with a
usable query), and a sensible one (no PubMed for the engineering and
computing questions, PubMed for the biomedical ones). Paste the printed
summary into issue #16 before merging.

    python scripts/spike_tool_calling.py [--model qwen3.5:9b] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

# (question, field, PubMed expected?)
PROMPTS = [
    ("Fatigue life of welded steel joints under variable amplitude loading", "Mechanical", False),
    ("Self-healing concrete using bacterial additives for crack repair", "Civil", False),
    ("Grid-forming inverter control for weak power grids", "Electrical", False),
    ("Low-power wake-up receivers for IoT sensor nodes", "Electronics", False),
    ("Kubernetes autoscaling policies for latency-sensitive microservices", "Cloud / DevOps", False),
    ("Retrieval-augmented generation for reducing LLM hallucination", "Computer Science", False),
    ("Effects of intermittent fasting on cognitive performance in adults", "Nutrition", True),
    ("Gut microbiome changes after antibiotic treatment in infants", "Medicine", True),
    ("CRISPR base editing efficiency in human hematopoietic stem cells", "Genetics", True),
    ("Urban green space access and residents' mental health", "Public health", True),
    ("Remote work and employee productivity after 2020", "Management", None),
    ("Topological insulators for low-dissipation electronics", "Physics", None),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b"))
    parser.add_argument("--json", type=Path, help="also save every prompt's raw calls here")
    args = parser.parse_args()

    from discovery import source_plan
    from shared.utilities import llm_provider

    keys = ["s2", "openalex", "pubmed", "arxiv"]
    rows = []
    for question, field, wants_pubmed in PROMPTS:
        started = time.monotonic()
        try:
            calls = await llm_provider.complete_tools(
                system=source_plan.PLAN_SYSTEM,
                user=f"Research question: {question}\nField: {field}",
                tools=source_plan.tools(keys), model=args.model)
            error = None
        except llm_provider.LLMProviderError as exc:
            calls, error = None, str(exc)
        plan = source_plan._valid(calls or [], keys)
        sensible = plan is not None and (wants_pubmed is None or ("pubmed" in plan) == wants_pubmed)
        rows.append({"question": question, "field": field, "calls": calls, "error": error,
                     "valid": plan is not None, "sensible": sensible,
                     "sources": sorted(plan) if plan else [],
                     "seconds": round(time.monotonic() - started, 1)})
        print(f"{'OK ' if sensible else 'VAL' if plan else 'NO '} {field:<16} "
              f"{','.join(sorted(plan)) if plan else (error or 'no valid calls')}")

    valid = sum(r["valid"] for r in rows)
    sensible = sum(r["sensible"] for r in rows)
    print(f"\nModel {args.model}: {valid}/{len(rows)} valid tool plans, "
          f"{sensible}/{len(rows)} sensible (PubMed only where it belongs).")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
