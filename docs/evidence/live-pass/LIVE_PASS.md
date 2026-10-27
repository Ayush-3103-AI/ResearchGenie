# Live pass

Written by `scripts/live_pass.py` on 2026-10-27 16:12 UTC. Provider: ollama qwen3.5:9b, commit ad6ec33. Every number below is measured, failures included.

| Check | Ticket | Result | Measured |
|---|---|---|---|
| Full test suite | — | PASS | 553 passed, 1 skipped, 1 warning in 537.49s (0:08:57) |
| Tool-calling spike | #16 | PASS | 12/12 valid tool plans, 11/12 sensible; model step stays on |
| Doctor recovers a run after Ollama is stopped | #13 | FAIL | failed before Writing: The analysis exceeded the output limit. Try analyzing fewer papers. →  →  |
| Dev Agent APA 7, red → green, no hand edits | #26 | FAIL | gemini-2.5-flash (Gemini): None in None round(s), red first: False; suite after patch:  |
| Fresh full runs vs the 3,691-word baseline (D-039) | #27 | PASS | mean 2768 words over the fresh runs |
| Library reuse on a related run | #19 | PASS | Discovery 239.5 s → 259.8 s; memory_hit {'raw_type': 'memory_hit', 'count': 10, 'reused_excerpts': 1, 'seeded_papers': 9} |
| Trend Advisor on the real model | #27 | PASS | CS_AI_ML: 8 topics, 12 gap statements, handed off; ECE: 8 topics, 12 gap statements; MECHANICAL: 8 topics, 10 gap statements |
| Container, local profile | #23 | not run |  |

## Runs

| Run | draft_status | Words | QA overall | Verified / invalid / unverifiable | Discovery s |
|---|---|---|---|---|---|
| fasting | complete | 3357 | 3.5 | 7 / 0 / 0 | 239.5 |
| remote-work | complete | 2709 | 4.25 | 6 / 0 / 2 | 141.6 |
| green-space | partial | 2238 | 4.25 | 6 / 0 / 0 | 242.1 |
| related | complete | 3648 | 4.25 | 7 / 0 / 0 | 259.8 |

Raw output for each step is beside this file (`<step>.json`, `<step>/`).
