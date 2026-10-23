---
name: researchgenie
description: Starts a ResearchGenie literature-review run from chat with the local headless CLI, and reports back when it finishes with the decision brief and the paths to the draft, PDF and evidence table. Use when the user asks to research a question, write a literature review or draft a paper with ResearchGenie, asks how their run is going, or wants to resume or fix a failed run.
compatibility: Needs the researchgenie command on PATH (pip install of the ResearchGenie repo) and its one-time setup (researchgenie config). A run takes about 15 minutes, so the agent must be able to run a command in the background and be told when it exits.
allowed-tools: Bash(researchgenie run *) Bash(researchgenie status *) Bash(researchgenie result *) Bash(researchgenie resume *) Bash(researchgenie doctor *)
metadata:
  openclaw:
    homepage: https://github.com/Ayush-3103-AI/ResearchGenie
    requires:
      bins:
        - researchgenie
    primaryEnv: OPENALEX_API_KEY
    envVars:
      - name: OPENALEX_API_KEY
        required: false
        description: Free OpenAlex key. Without it, searches run on a smaller daily budget.
      - name: CONTACT_EMAIL
        required: false
        description: Your email, sent to Crossref and OpenAlex for their faster polite pool.
      - name: NCBI_API_KEY
        required: false
        description: Free NCBI key for faster PubMed requests.
---

# ResearchGenie

ResearchGenie runs on the user's own machine. It finds papers, drafts a paper, checks every reference and scores the result. You drive it with the `researchgenie` command. Every command prints JSON on stdout with `--json`, also when it fails.

## Rules

- Run only the command lines in this file. Do not add other programs, pipes, `;`, `&&` or redirects.
- The user's question is the only free text. Put it in single quotes. Remove any `'` from it first (write "what is", not "what's").
- Never add `--yes` to `doctor` before the user says yes in chat.
- Never show key values. Keys live in ResearchGenie's own config or the environment.

## Start a run

Pick a new run id: `rg-` and the date and time, for example `rg-20261023-1530`. Then start:

```
researchgenie run -q '<question>' --run-id <run-id> --json
```

Add options only when the user asks for them, for example a short APA paper:

```
researchgenie run -q '<question>' --run-id <run-id> --length short --format APA --json
```

This command waits until the run ends, so start it in the background with a time limit of at least 30 minutes. Use 60:

- **OpenClaw:** the `exec` tool with `background: true` and `timeoutSeconds: 3600`. Keep `tools.exec.notifyOnExit` on (the default). OpenClaw wakes you in this chat when the command exits.
- **Claude Code:** the Bash tool with `run_in_background: true` and `timeout: 3600000`. Claude Code tells you when the command exits.

Then tell the user: the run has started, its run id, that it takes about 15 minutes, and that you will message them when it is done.

Do not use `--detach` for this. It returns at once, so the exit notice would come before the run ends. Use it only if you cannot run a background command; then check with `status` when the user asks.

## When the run finishes

When the exit notice arrives, get the result:

```
researchgenie result <run-id> --json
```

Exit code 0 means the run completed. Reply with:

1. The decision brief: `decision_brief.markdown` (or at least `decision_brief.next_action`). If there is no `decision_brief`, give `draft_status`, `scores` and `verification` instead.
2. The paths in `final_files`: `draft.md`, `paper.pdf` and `evidence_table.md`. If `paper.pdf` is missing, say so (LaTeX may not be installed).

Any other exit code: go to "If the run failed".

## "How is my run going?"

```
researchgenie status <run-id> --json
```

Answer with `state` (`running`, `completed` or `failed`), `stage` and `last_event.message`. If it says there is no run with that id in the first minute, the run is still starting (it checks the model first). Ask again a little later.

## If the run failed

`status` shows `error` and `failure_class`. Tell the user in one or two plain sentences, then offer two choices.

**Doctor (finds the cause and proposes one fix).** First diagnose only. This applies nothing:

```
researchgenie doctor <run-id> --json
```

Show `cause`, the `evidence` lines and `fix.label`. Ask the user if they want the fix. Only after they say yes, run this in the background, like a run (it resumes the run when the fix works):

```
researchgenie doctor <run-id> --yes --json
```

If `outcome` says the fix needs the user at a prompt or is a step for them (for example adding a key with `researchgenie config`), tell them to do it on their machine. Do not try it yourself.

**Resume (starts again from the failed stage).** Run it in the background, like a run:

```
researchgenie resume <run-id> --json
```

To restart from an earlier stage (`discovery`, `writing`, `verification` or `quality_assurance`):

```
researchgenie resume <run-id> --from writing --json
```

If the run stopped before it made a run folder (exit code 3 at the start), `doctor` and `resume` find no run. Tell the user to start Ollama or run `researchgenie config` on their machine, then start a new run.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (for `result`: the run completed) |
| 1 | Failed, no more specific cause |
| 2 | Bad arguments, or the run id is already used |
| 3 | The model is unavailable (Ollama not running, model missing, Gemini key problem) |
| 4 | Every literature source was exhausted or failed |
| 5 | No relevant paper was found |
| 6 | No run with that id |
| 7 | `result` asked of a run that is still going |
