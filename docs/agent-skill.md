# ResearchGenie agent skill

One skill folder, [`skills/researchgenie/`](../skills/researchgenie/SKILL.md), lets an agent start a run from chat and report back when it finishes. It works in Claude Code and in OpenClaw, so a student can ask from WhatsApp, Telegram, Slack or Discord. It follows the [Agent Skills spec](https://agentskills.io/specification) and passes the reference validator:

```
uvx --from "git+https://github.com/agentskills/agentskills#subdirectory=skills-ref" skills-ref validate skills/researchgenie
```

`tests/test_agent_skill.py` checks the frontmatter and parses every command line in the skill with the headless CLI's own parser, so the skill can't drift from the CLI.

## How it works

1. The agent picks a run id (`rg-20261023-1530`) and starts `researchgenie run -q '<question>' --run-id <run-id> --json` as a background command with a 60-minute limit. The command waits until the run ends.
2. When the command exits, the runtime wakes the agent (OpenClaw's `notifyOnExit`, Claude Code's background-task notice). The agent runs `researchgenie result <run-id> --json` and replies with the decision brief and the paths to `draft.md`, `paper.pdf` and `evidence_table.md`.
3. "How is my run going?" gets `researchgenie status <run-id> --json`.
4. On a failure the agent offers `doctor` (diagnose first, apply the fix only after the user says yes) or `resume`.

Every command is a fixed `researchgenie` command line. The question is the only free text and goes in single quotes. There is no inline code, no install script and no other binary.

## Before you install

Install ResearchGenie and run its setup once, on the machine where the agent runs:

```
pip install -e .
researchgenie config
```

`researchgenie` must be on the agent's PATH. Keys are optional. ResearchGenie reads them from its own config; `OPENALEX_API_KEY`, `CONTACT_EMAIL` and `NCBI_API_KEY` in the environment win over the saved values.

## Install in Claude Code

As a personal skill (all your projects): copy the folder.

```
cp -r skills/researchgenie ~/.claude/skills/researchgenie
```

On Windows: `Copy-Item -Recurse skills\researchgenie $HOME\.claude\skills\researchgenie`. Claude Code picks up the skill without a restart.

As a plugin, from this repo's marketplace (`.claude-plugin/marketplace.json`):

```
claude plugin marketplace add Ayush-3103-AI/ResearchGenie
claude plugin install researchgenie@researchgenie
```

The plugin's skill is `/researchgenie:researchgenie`. `claude plugin details researchgenie` should list `Skills (1) researchgenie`.

The skill's `allowed-tools` pre-approves the five `researchgenie` commands while the skill runs. To approve them always, add `Bash(researchgenie run *)`, `Bash(researchgenie status *)`, `Bash(researchgenie result *)`, `Bash(researchgenie resume *)` and `Bash(researchgenie doctor *)` to `permissions.allow` in your settings.

## Install in OpenClaw

Install the folder into the active workspace's `skills/` folder:

```
openclaw skills install ./skills/researchgenie
openclaw skills check
```

Or copy `skills/researchgenie` to `<workspace>/skills/researchgenie` yourself. Copy, don't symlink: OpenClaw skips skill folders whose real path is outside the skills root. A new session picks the skill up.

`openclaw skills check` lists it as ready once `researchgenie` is on PATH (the skill's only hard requirement). The three keys are declared optional, so a missing key never hides the skill. To give OpenClaw the OpenAlex key, set `skills.entries.researchgenie.apiKey` (it maps to `OPENALEX_API_KEY`).

Notes:

- Run the agent on the host, not in a sandbox. The run needs the local Ollama and writes to ResearchGenie's `outputs/` folder.
- Keep `tools.exec.notifyOnExit` on (the default). Without it the agent is not woken when a run ends.
- In exec allowlist mode, allow the `researchgenie` binary. The skill never chains commands, so nothing else is needed.

## Publish (left to a human)

**ClawHub.** Install the CLI, log in, publish the folder:

```
npm i -g clawhub
clawhub login
clawhub skill publish ./skills/researchgenie --slug researchgenie --name "ResearchGenie" --version 0.1.0 --changelog "First release" --tags latest
```

ClawHub publishes every skill under MIT-0 (no attribution needed). The repo is MIT; publishing the skill folder means agreeing to MIT-0 for it. Users then install with `openclaw skills install @<publisher>/researchgenie`.

**Claude Code plugin marketplace.** The marketplace is this repo. Merge `.claude-plugin/marketplace.json` to the default branch and push; users add it with the two commands above. Before pushing a change, run `claude plugin validate .`. Bump `version` in `marketplace.json` for each release, or users stay on the old one. To list it in Anthropic's public directory, follow [Submit to Anthropic's directory](https://code.claude.com/docs/en/plugins/publish).

## Demo script (for the recording)

Setup: ResearchGenie installed and configured, Ollama running, the skill installed in OpenClaw, OpenClaw connected to a chat app (Telegram works well on screen). Show the phone or chat window and a terminal side by side.

1. In the chat, send: *"Research how sleep affects memory consolidation. Make it a short APA paper."*
2. The agent answers: run started, the run id, about 15 minutes, it will message when done. In the terminal, show `researchgenie status <run-id> --json` reporting `running`.
3. A few minutes later, send: *"How is my run going?"* The agent answers with the stage and the last event.
4. Wait for the finish message (cut the wait in editing). It contains the decision brief (strong points, weak points, references to check, one next action) and the paths to `draft.md`, `paper.pdf` and `evidence_table.md`. Open `paper.pdf` on screen.
5. Failure path: start a second run, then stop Ollama (`ollama` tray: Quit) while it is in Discovery or Writing. The finish message says the run failed and offers doctor or resume.
6. Send: *"Run the doctor."* The agent shows the cause ("Ollama is not running"), the evidence and the proposed fix, and asks first. Reply *"yes"*. The agent runs the doctor with `--yes`, Ollama starts, the run resumes, and a second finish message arrives with the brief and paths.
7. Optional: on a failed run, send *"Resume it."* and show the resumed run finishing.
8. Repeat steps 1 and 3 in Claude Code to show the same skill working there.
