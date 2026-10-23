# ResearchGenie

[![tests](https://github.com/Ayush-3103-AI/ResearchGenie/actions/workflows/test.yml/badge.svg)](https://github.com/Ayush-3103-AI/ResearchGenie/actions/workflows/test.yml)

A local-first AI research assistant. Give it a research question — it discovers relevant literature, identifies genuine research gaps, drafts an evidence-grounded paper, verifies every citation against real scholarly databases, and independently audits its own quality. Everything runs on your own machine by default.

## 📦 Install

```
pip install researchgenie
```

## 🚀 Run

```
researchgenie
```

That's it. The first run walks you through a one-time setup; every run after that goes straight to research.

## ✨ What happens the first time

1. **Choose your AI engine**
   - 🟢 **Ollama** *(recommended)* — runs entirely on your machine, your research data never leaves your device, no API key needed.
   - 🔵 **Gemini** — a cloud alternative for machines without the hardware to comfortably run a local model. Requires your own [Gemini API key](https://aistudio.google.com/apikey).
2. If you chose Ollama, ResearchGenie checks whether it's installed and the model is downloaded — and does both automatically where it can be done safely on your operating system.
3. **Optional: scholarly access** — a free OpenAlex key, a contact email and a free NCBI key make literature searches faster and more reliable. Each is checked live and can be skipped. There is no Semantic Scholar key; it stays keyless.
4. Your choices are saved locally. You won't be asked again.

Run `researchgenie config` any time to change your setup.

## 🔬 Doing research

Answer a few short questions — your research topic, how many papers to include (6–9), your preferred format (IEEE, Springer or APA), and your target length (Short/Standard/Detailed) — confirm the summary, and watch live progress as each stage of the pipeline runs:

- 🔍 **Research Discovery** — searches multiple scholarly databases, filters for genuine topical relevance, and selects the strongest papers.
- ✍️ **Research Writing** — builds an evidence-grounded outline and draft, citing only the literature it actually found.
- 🔗 **Citation Verification** — checks every reference against real scholarly databases (Crossref, OpenAlex).
- 🛡️ **Quality Assessment** — an independent, fully deterministic audit of citation integrity and claim-to-evidence alignment.

After Discovery picks the papers, the run pauses and shows them as a numbered table. Press Enter to approve, or type `drop 2,5` to swap papers for the next-best candidates. If the search finds too few on-topic papers, Discovery tries once more with new queries.

A run that fails part-way does not have to start over: `researchgenie resume <run folder or run id>` picks up from the failed stage (`--from STAGE` to choose one). Earlier output is kept as `final_v1/`, `final_v2/`, ….

Not sure why a run failed? `researchgenie doctor <run id>` reads the run's event log and checks your system (Ollama, the model, keys, LaTeX). It explains the cause in plain language, quotes the evidence, and proposes one fix: start Ollama, download the model, add a free key, wait for a provider's reset, or resume from a stage. It applies the fix only after you say yes (`--yes` in headless mode) and saves `doctor_report.json` in the run folder. A failed run in the web UI has the same thing as a **Diagnose & fix** button.

Searches, DOI lookups and extracted full text are cached in `~/.researchgenie/cache.sqlite` (searches for 7 days, lookups for 30, full text for good), so a repeat or similar run is faster and spends almost no quota. Add `--no-cache` to any command to fetch everything fresh; `researchgenie cache clear` empties the cache.

## 🤖 Headless use (scripts and agents)

The same pipeline runs without prompts, with JSON output and exit codes a script can act on:

```
researchgenie run -q "How does sleep affect memory consolidation?" --format APA --length short --json
researchgenie run -q "..." --detach --json      # prints {"run_id": ...} and returns at once
researchgenie status <run id> --json            # stage, state, last event
researchgenie result <run id> --json            # final files, scores, verification counts, draft status
researchgenie resume <run id> --json            # headless resume (--from STAGE to choose)
researchgenie doctor <run id> --yes --json      # diagnose, apply the fix, resume
researchgenie library search "<query>" --json   # past runs and papers in your library
```

`run` takes every research-request field as a flag (`researchgenie run --help` lists them). `--clarify` is `skip` (default) or `auto`; approval is always automatic headless. Progress goes to stderr, the answer to stdout.

| Exit code | Meaning |
|---|---|
| 0 | Success (for `result`: the run completed) |
| 1 | Failed, no more specific cause |
| 2 | Bad arguments |
| 3 | The model is unavailable (Ollama not running, model missing, Gemini key problem) |
| 4 | Every literature source was exhausted or failed |
| 5 | No relevant paper was found |
| 6 | No run with that id or folder |
| 7 | `result` asked of a run that is still going |

Other tools can read your runs, search your library and check DOIs through a read-only MCP server, `researchgenie-mcp`. Setup for Claude Code, Claude Desktop and OpenClaw is in [docs/mcp-server.md](docs/mcp-server.md).

## 🧭 Not sure what to research yet?

```
researchgenie-advise
```

Pick your field — Computer Science/AI/ML, ECE, EEE, Mechanical, Civil, Cloud/DevOps, or type your own — and the Trend & Gap Advisor produces a ranked shortlist of rising topics, each backed by evidence you can check yourself:

- **Publication growth** computed from live OpenAlex counts (how many papers in the last three complete years versus the three before them) — never guessed by the AI.
- **Recurring research gaps** mined in parallel from what published papers themselves say is still unsolved, flagged only when several independent papers say it.
- **Hot-but-crowded vs under-explored**, so you can decide whether to compete in a busy area or fill a quiet one.

Ask follow-up questions about any topic, say "go with #2" when you've decided, and your pick auto-fills the research question for the full pipeline — which you can run immediately or come back to later. If a data source or the local model is unreachable, the advisor says so rather than showing you a confident empty result.

## 🖥️ Web demo (for evaluators)

A browser front end over the same pipeline, showing every agent stage live. With Ollama running and `qwen3.5:9b` pulled:

```
git clone https://github.com/Ayush-3103-AI/ResearchGenie.git
cd ResearchGenie
python -m venv .venv
.venv\Scriptsctivate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e .
uvicorn orchestrator.api:app --port 8000
```

Open http://localhost:8000. The header shows whether the model is ready. Then either:

1. **Start from a field**: the Trend Advisor ranks rising topics from live OpenAlex data and mines each one's papers for stated gaps in parallel. Open a topic's evidence, ask the advisor questions, pick one, edit the drafted research question, and run the pipeline on it.
2. **Start from a question**: type a question and run Discovery → Writing → Verification → Quality directly.

A stage tracker and a live event log follow the run; results show quality scores, citation-verification counts, the draft, reports, and downloads (`draft.md`, `paper.tex`, `paper.pdf` when LaTeX is installed). Reloading the page reconnects to the run. A full run takes roughly 15 minutes on an RTX 4060, much longer on CPU. The API docs are at http://localhost:8000/docs.

## 🐳 Run with Docker

The same web UI and API in a container, with tectonic included for PDFs. Pick one profile from the repo root:

**Local (Ollama in a container, nothing leaves your machine):**

```
docker compose --profile local up --build
```

This starts the app, an Ollama server, and a one-shot `ollama-pull` service that downloads the model (`qwen3.5:9b`, about 6.6 GB) into the `ollama-models` volume the first time. Open http://localhost:8000: the header turns ready once the pull finishes (`docker compose logs -f ollama-pull` shows progress). To pull by hand instead: `docker compose exec ollama ollama pull qwen3.5:9b`. Set `LOCAL_AI_MODEL` to use a different model.

With an NVIDIA GPU (NVIDIA Container Toolkit on Linux, or Docker Desktop with WSL2 on Windows), add the GPU override; without it Ollama runs on CPU, which is much slower:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile local up --build
```

**Cloud (Gemini, no GPU needed):**

```
GEMINI_API_KEY=your-key docker compose --profile cloud up --build
```

On Windows PowerShell, run `$env:GEMINI_API_KEY="your-key"` first. You can also put `GEMINI_API_KEY=...` in a `.env` file next to `docker-compose.yml` (it is git-ignored and excluded from the image). The key is read at run time only and is never built into the image.

| Variable | Profile | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | cloud | — | Your Gemini API key (required) |
| `GEMINI_MODEL` | cloud | `gemini-2.5-flash` | Gemini model |
| `LOCAL_AI_MODEL` | local | `qwen3.5:9b` | Ollama model to pull and use |
| `CONTACT_EMAIL` | both | — | Optional; Crossref/OpenAlex polite pool and the email NCBI asks for (`researchgenie config` sets it) |
| `OPENALEX_API_KEY` | both | — | Optional free key, about 100 → 1,000 searches a day (`researchgenie config` sets it) |
| `NCBI_API_KEY` | both | — | Optional free key, PubMed at 10 instead of 3 requests a second (`researchgenie config` sets it) |
| `RESEARCHGENIE_PORT` | both | `8000` | Host port for the web UI |

Run folders land in `./outputs` on your machine. On Linux the container writes as uid 1000, so that folder must be writable by uid 1000. The first PDF compile downloads tectonic's TeX bundle (a few minutes, needs internet); it is cached in the `researchgenie-home` volume after that. Stop with `docker compose --profile local down` (or `cloud`); add `-v` to also delete the model and cache volumes.

## 📁 What you get

Every run saves a complete, timestamped project folder containing:

- The research draft (Markdown)
- LaTeX source, and a compiled PDF when a LaTeX toolchain is available on your machine
- The reference list
- The quality and citation-validation reports
- Full run metadata

ResearchGenie always tells you the exact folder path when a run finishes.

## 🛡️ Safety

[docs/GOVERNANCE.md](docs/GOVERNANCE.md) maps each risk (invented citation, off-topic paper, unsupported claim, key leak, exhausted provider, bad output) to the control that handles it, where it is in the code, the test that proves it, and how to roll back.

## 🔒 Privacy

With the default Ollama engine, your research question, retrieved literature, and generated draft never leave your machine — there is no cloud AI call anywhere in the pipeline. Choosing the optional Gemini engine sends prompts to Google's API, same as using any cloud AI service; your API key is stored locally and never logged, committed, or embedded in generated output.

## 📊 Project status

For the final review: [docs/lab-worklet-final.html](docs/lab-worklet-final.html) maps each of the 14 lab weeks to its code, tests and live evidence. [docs/demo-runbook.md](docs/demo-runbook.md) is the 12-minute demo, and [docs/review-deck.html](docs/review-deck.html) is the deck. `python scripts/live_pass.py` runs every check that needs a real model and writes [docs/evidence/live-pass/LIVE_PASS.md](docs/evidence/live-pass/LIVE_PASS.md). For a hosted demo, see [docs/hosted-demo.md](docs/hosted-demo.md).

See [TILL_NOW.md](TILL_NOW.md) for current progress and [DECISIONS.md](DECISIONS.md) for technical decisions and their reasoning. [analytics_and_results.md](analytics_and_results.md) documents the reliability work behind the pipeline, with real measured results.

## 🛠️ For developers

This repository *is* the ResearchGenie source — `pip install researchgenie` installs it directly from here. See `DECISIONS.md` for the internal architecture (a four-service pipeline behind one orchestrator, plus an optional Trend & Gap Advisor stage) and `pyproject.toml` for the packaging setup.

`researchgenie-dev "<ticket text>"` (or `--spec ticket.md`) runs the Dev Agent on a small, well-specified ticket: it plans, writes a failing test, then patches until the test passes (at most 3 rounds), in a throwaway git worktree. It never commits, pushes or merges. Review `patch.diff` in the audit folder it prints, then `git apply` it yourself.
