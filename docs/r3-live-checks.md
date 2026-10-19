# R3: the three live checks left

R3 (#6, #10, #13, #16, #20, #22, #26) is built and tested on the branch `r3/integration`. Three acceptance criteria need a real model, and the machine R3 was built on had none. Run them on a machine with Ollama and `qwen3.5:9b` (or a Gemini key), record each result where it says, and then the branch can be merged.

## Setup

```
git fetch genie
git checkout r3/integration
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e . pytest
python -m pytest -q               # expect all pass; the live tests run too when Ollama is up
ollama list                       # qwen3.5:9b should be there; if not: ollama pull qwen3.5:9b
```

## 1. Tool-calling spike (#16)

The ticket says this result must be recorded before merging.

```
python scripts/spike_tool_calling.py --json spike.json
```

It asks the model to choose databases for 12 research questions and prints one line per question, then a summary such as `9/12 valid tool plans, 8/12 sensible`. Paste the printed output into a comment on issue #16.

If the numbers are low, nothing breaks: every invalid or missing call falls back to the default sources for the domain. Say in the comment whether you think the model step should stay on.

## 2. Doctor live demo (#13)

1. Start a run and let it get past Discovery:
   ```
   researchgenie run -q "Effects of intermittent fasting on cognitive performance" --length short --json
   ```
2. While it is in Writing (watch the `[writing]` lines on stderr), stop Ollama: quit it from the tray, or end the `ollama` process.
3. The run fails. Note its `run_id` from the JSON output.
4. Diagnose it and confirm the fix:
   ```
   researchgenie doctor <run_id>
   ```
   It should say Ollama is not running, quote the evidence, and offer "Start Ollama, then resume". Answer `y`. It starts Ollama, resumes from the failed stage, and the run completes.
5. Record it: attach a screen recording or the terminal output, plus the run folder's `doctor_report.json`, to issue #13. The report's `outcome` should read `Fix applied; the resumed run completed.`

The web UI has the same flow. A failed run shows **Diagnose & fix**, then **Apply fix**. Start the server with `uvicorn orchestrator.api:app`.

## 3. Dev Agent APA run (#26)

```
researchgenie-dev --spec docs/tickets/26-apa-7.md
```

The Dev Agent writes a failing test, then patches until it passes (at most 3 rounds), in a throwaway worktree. When it finishes:

1. Open the audit folder it prints. Check that the saved run shows the test red first, then green.
2. Review `patch.diff`. Don't edit it by hand: the ticket asks for no hand edits.
3. Apply it with `git apply <audit folder>/patch.diff`, run `python -m pytest -q`, and commit it.
4. Fill in the **Result** line at the end of `docs/tickets/26-apa-7.md`: which model produced the patch, and whether it went red → green with no hand edits.

If the local model can't finish in 3 rounds, run it again with Gemini (`researchgenie config`, choose Gemini) and record that in the ticket file.

## Then

Push the commits to `r3/integration` and open the PR to `main`. Issues #6, #10, #20 and #22 need nothing more. Close #13, #16 and #26 once their results are posted.
