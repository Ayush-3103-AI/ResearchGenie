# Demo runbook: final review

This is the 12-minute run of show for the professor and other stakeholders. It runs on the GPU laptop (Ollama + `qwen3.5:9b`). The hosted URL ([hosted-demo.md](hosted-demo.md)) is the backup, and the recorded video is the last resort.

## The day before

1. `git pull` on the GPU laptop, then `pip install -e .` in the venv. Run `python -m pytest -q` and note the count.
2. `python scripts/live_pass.py --push` if it has not run yet. It records every live check in `docs/evidence/live-pass/LIVE_PASS.md` (about 2–3 hours).
3. Warm the cache: run the demo question once (`researchgenie run -q "<demo question>" --length short --json`). On the day, Discovery then reads from the response cache and the library, so a rate limit cannot stall the demo.
4. Record the fallback video of this whole runbook (6 minutes or less).
5. Open these files in tabs:
   - a golden run: `docs/evidence/live-pass/runs/fasting/final/` (draft, PDF, evidence table, decision brief)
   - `docs/evidence/live-pass/LIVE_PASS.md`
   - `docs/lab-worklet-final.html`
   - the Dev Agent audit: `docs/evidence/live-pass/devagent/audit/`
6. Claude Desktop has `researchgenie-mcp` configured ([mcp-server.md](mcp-server.md)). Check it once with "list my ResearchGenie runs".

## On the day: 15 minutes before

```
ollama list                                  # qwen3.5:9b is there
uvicorn orchestrator.api:app --port 8000     # web UI at http://localhost:8000
researchgenie status <warm run id> --json    # the CLI answers
```

Turn off sleep, notifications and Windows Update restarts. Have the laptop on its charger.

## Run of show (12 min)

| Min | Show | Say (one line) | Weeks |
|---|---|---|---|
| 0–1 | The architecture slide | "Four agents in a pipeline, plus a trend advisor, a self-healing Doctor and a Dev Agent that improves the product. All of it runs on a 9B model on this laptop." | 14 |
| 1–3 | Web: **Find rising topics** for CS/AI/ML → the shortlist with growth numbers and gap flags → **Pick this topic** | "The ranking comes from real OpenAlex counts, not the model's opinion. A gap counts only when several papers raise it." | 13, 8 |
| 3–5 | Type a vague question → clarify (**Use suggestion**) → live stage tracker and log → **Approve and continue** at the paper approval gate | "It asks only when the question is vague. A human approves the corpus before any writing." | 1, 2, 6, 12 |
| 5–6 | In a terminal, end the `ollama` process → the run fails → **Diagnose & fix** → **Apply fix** → it resumes | "The Doctor quotes the evidence, names the cause from a rule table, asks before it acts, and resumes from the failed stage. It does not start over." | 10, 12 |
| 6–8 | The golden run: decision brief → evidence table → validation report → QA report → PDF | "Every claim traces to a paper and a DOI. Every reference is verified, invalid or unverifiable, and never repaired. QA is deterministic. The brief gives exactly one next action." | 9, 12, 7 |
| 8–9 | Claude Desktop: "search my ResearchGenie library for fasting", "verify DOI 10.…" | "A read-only MCP server: any MCP client can use our library and DOI checker." | 5, 4 |
| 9–10 | The chat clip: a question sent from a chat app → OpenClaw skill → the finish message with the brief | "One skill folder works in both Claude Code and OpenClaw. It only calls fixed CLI commands." | 3, 6 |
| 10–11 | The Dev Agent audit: the red test, the green test, the merged APA commit | "The agent wrote a failing test first, then patched until it passed, with the old tests as guards. A human applied the diff." | 11 |
| 11–12 | `LIVE_PASS.md` and the final sheet | "Every number here is measured, including the ones that are not good yet." | 14 |

The run started at minute 3 keeps going in the background (about 15 minutes). Open its folder during the Q&A.

## If something fails on the day

| Problem | Do this |
|---|---|
| Ollama will not start | Switch to the hosted URL (Gemini) for the live parts. Open the golden run for the outputs. |
| Discovery stalls on a rate limit | Use the warmed demo question. Its search answers come from the cache. |
| The Doctor fix does not resume | Show `doctor_report.json` and the recording in `docs/evidence/live-pass/doctor/`. |
| Claude Desktop cannot start the MCP server | Use the full path to `researchgenie-mcp.exe` in its config, or run `npx @modelcontextprotocol/inspector researchgenie-mcp`. |
| Everything fails | Play the fallback video, then take questions with the golden run open. |

## Likely questions

- **Is the agent really choosing its plan?** The order of the stages is fixed in code, on purpose, so that every run can be audited. The model chooses the queries, the sources (by tool calling, checked against the field), the gaps and the prose. The clarify, re-plan, revise and Doctor steps are the points where it adapts.
- **Why a 9B local model?** No cost and no data leaves the laptop. It is also a hard test: if the safeguards work on a 9B model, they work on a bigger one. Gemini is one setting away.
- **How do you know the citations are real?** Each DOI is looked up in Crossref and OpenAlex. A reference that cannot be confirmed is marked; it is never silently fixed.
- **What doesn't work yet?** Read the FAIL rows in `LIVE_PASS.md` aloud. Do not hide them.
