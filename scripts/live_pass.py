"""Every check that needs a real model, in one command (#13, #16, #19, #26, #27).

Run it on the machine with Ollama and qwen3.5:9b (the GPU laptop), from the
project venv, at the repo root:

    python scripts/live_pass.py                 # all steps, about 2-3 hours
    python scripts/live_pass.py --only spike doctor
    python scripts/live_pass.py --push          # then commit the evidence and push

Each step saves what it saw under docs/evidence/live-pass/ (<step>.json plus
copied artifacts), then docs/evidence/live-pass/LIVE_PASS.md sums them up.
A step that already passed is skipped on the next start (--redo to run it
again), so the pass can be stopped and restarted. Nothing here edits the
product code except the Dev Agent step, which applies the agent's own patch
after its tests went red then green, with no hand edits (the #26 rule).

Steps, in order:
  preflight  the model answers; record provider, model and git commit
  tests      the full pytest count
  spike      tool-calling reliability on 12 questions (#16)
  doctor     a run is started, Ollama is stopped during Writing, then
             `researchgenie doctor --yes` starts it and resumes the run (#13)
  devagent   the Dev Agent's APA 7 ticket, red -> green; the patch is applied
             and the suite re-run (#26). Retries on Gemini if GEMINI_API_KEY is set.
  runs       3 fresh full runs: words and draft_status against the 3,691-word
             baseline (D-039), and a related second run to time the library (#19)
  advisor    the Trend Advisor on the real model: gap statements for 3 fields,
             one hand-off into the full pipeline (#27, weeks 8 and 13)
  docker     opt-in (--only docker): compose's local profile reports model_ready (#23)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "live-pass"
OUTPUTS = ROOT / "outputs"
PY = sys.executable
BASELINE_WORDS = 3691  # TESTING_PROTOCOL.md: 4 fresh runs before D-039
OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")

# (label, question). "related" repeats the first topic so the library has
# something to reuse; its Discovery time is compared with the first run's.
RUN_QUESTIONS = [
    ("fasting", "What are the effects of intermittent fasting on cognitive performance?"),
    ("remote-work", "What is the impact of remote work on employee productivity?"),
    ("green-space", "How does urban green space access affect residents' mental health?"),
    ("related", "How does intermittent fasting affect memory and attention in adults?"),
]
DOCTOR_QUESTION = "Effects of caffeine on short-term memory in adults"
ADVISOR_DOMAINS = ["CS_AI_ML", "ECE", "MECHANICAL"]  # the first one also hands off to the pipeline


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sh(args: list[str], *, timeout: float | None = None, env: dict | None = None,
       out: Path | None = None) -> subprocess.CompletedProcess:
    """Runs one fixed command line; stdout+stderr are kept, and copied to `out`."""
    log("$ " + " ".join(args))
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env={**os.environ, **(env or {})})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
                       encoding="utf-8")
    return proc


def cli(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return sh([PY, "-m", "researchgenie.cli", *args], **kwargs)


def json_out(proc: subprocess.CompletedProcess) -> dict:
    """The JSON object a --json command printed (progress lines go to stderr)."""
    text = proc.stdout
    start = text.find("{")
    try:
        return json.loads(text[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except OSError:
        return False


def ollama_models() -> list[str]:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as resp:
        return [m["name"] for m in json.load(resp).get("models", [])]


def stop_ollama() -> bool:
    """Kills the Ollama app and server, the way a crash or a closed tray app would."""
    if sys.platform == "win32":
        for image in ("ollama app.exe", "ollama.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "Ollama"], capture_output=True)  # the macOS app
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    for _ in range(20):
        if not ollama_up():
            return True
        time.sleep(1)
    return False


def run_folder(run_id: str) -> Path | None:
    return next(OUTPUTS.glob(f"*_{run_id}"), None)


def events_of(folder: Path) -> list[dict]:
    path = folder / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keep(folder: Path, dest: Path, names=("events.jsonl", "metadata.json", "doctor_report.json")) -> None:
    """Copies a run's small audit files and final/ into the evidence folder."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (folder / name).is_file():
            shutil.copy2(folder / name, dest / name)
    if (folder / "final").is_dir():
        shutil.copytree(folder / "final", dest / "final", dirs_exist_ok=True)


# --------------------------------------------------------------------- steps

def step_preflight() -> dict:
    provider = os.getenv("AI_PROVIDER", "ollama")
    commit = sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    if provider == "gemini":
        ok = bool(os.getenv("GEMINI_API_KEY"))
        return {"ok": ok, "provider": "gemini", "commit": commit,
                "note": "" if ok else "AI_PROVIDER=gemini but GEMINI_API_KEY is not set"}
    if not ollama_up():
        return {"ok": False, "provider": "ollama", "commit": commit,
                "note": f"Ollama is not answering at {OLLAMA}. Start it and run again."}
    models = ollama_models()
    ok = any(m == MODEL or m.startswith(MODEL + ":") for m in models)
    return {"ok": ok, "provider": "ollama", "model": MODEL, "models": models, "commit": commit,
            "note": "" if ok else f"{MODEL} is not pulled: run `ollama pull {MODEL}`."}


def step_tests() -> dict:
    proc = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout=3600,
              out=EVIDENCE / "tests" / "pytest.txt")
    summary = next((l for l in reversed(proc.stdout.splitlines()) if " passed" in l or " failed" in l), "")
    return {"ok": proc.returncode == 0, "summary": summary.strip("= ")}


def step_spike() -> dict:
    proc = sh([PY, "scripts/spike_tool_calling.py", "--json", str(EVIDENCE / "spike" / "spike.json")],
              timeout=3600, out=EVIDENCE / "spike" / "spike.txt")
    match = re.search(r"(\d+)/(\d+) valid tool plans, (\d+)/\d+ sensible", proc.stdout)
    if not match:
        return {"ok": False, "note": "no summary line; see spike/spike.txt"}
    valid, total, sensible = map(int, match.groups())
    return {"ok": proc.returncode == 0, "valid": valid, "sensible": sensible, "total": total,
            "keep_model_step": valid >= 8, "summary": match.group(0)}


def step_doctor() -> dict:
    started = json_out(cli("run", "-q", DOCTOR_QUESTION, "--length", "short", "--detach", "--json"))
    run_id = started.get("run_id")
    if not run_id:
        return {"ok": False, "note": "the run did not start"}
    log(f"run {run_id} started; waiting for Writing to begin")
    deadline = time.time() + 45 * 60
    while time.time() < deadline:
        status = json_out(cli("status", run_id, "--json"))
        if status.get("state") == "failed":
            return {"ok": False, "run_id": run_id, "note": "failed before Writing: " + str(status.get("error"))}
        if status.get("stage") == "writing":
            break
        time.sleep(10)
    else:
        return {"ok": False, "run_id": run_id, "note": "never reached Writing in 45 min"}
    # No grace period: on a fast GPU a short run's Writing can finish in well under a minute.
    log("stopping Ollama as Writing starts")
    if not stop_ollama():
        log("Ollama still answers (a service manager restarted it?). Stop it by hand now; waiting 2 min.")
        for _ in range(120):
            if not ollama_up():
                break
            time.sleep(1)
        else:
            return {"ok": False, "run_id": run_id, "note": "could not stop Ollama"}
    # Writing isolates each card and section, so the run may end "failed" or
    # "completed" with a broken draft. The Doctor has a rule for both.
    deadline = time.time() + 20 * 60
    while time.time() < deadline and json_out(cli("status", run_id, "--json")).get("state") == "running":
        time.sleep(10)
    broken = json_out(cli("result", run_id, "--json"))
    folder = run_folder(run_id)
    if folder:
        keep(folder, EVIDENCE / "doctor" / "before")
    if broken.get("state") == "completed" and broken.get("draft_status") == "complete":
        return {"ok": False, "run_id": run_id, "note": "Writing finished before Ollama was stopped; "
                "nothing to diagnose. Run again with a longer question."}
    log(f"run ended {broken.get('state')} (draft_status {broken.get('draft_status')}); asking the Doctor")
    proc = cli("doctor", run_id, "--yes", "--json", timeout=90 * 60, out=EVIDENCE / "doctor" / "doctor.txt")
    report = json_out(proc)
    if folder:
        keep(folder, EVIDENCE / "doctor" / "after")
    final = json_out(cli("result", run_id, "--json"))
    return {"ok": final.get("state") == "completed" and "completed" in str(report.get("outcome", "")),
            "run_id": run_id, "before": {"state": broken.get("state"), "draft_status": broken.get("draft_status")},
            "rule": report.get("rule"), "cause": report.get("cause"),
            "fix": (report.get("fix") or {}).get("label"), "outcome": report.get("outcome"),
            "after": {"state": final.get("state"), "draft_status": final.get("draft_status")}}


def _dev_agent_once(env: dict | None) -> dict:
    proc = sh([PY, "-m", "dev_agent.agent", "--spec", "docs/tickets/26-apa-7.md"], timeout=3 * 3600,
              env=env, out=EVIDENCE / "devagent" / f"dev-agent-{(env or {}).get('AI_PROVIDER', 'ollama')}.txt")
    match = re.search(r"Audit trail and patch: (.+)", proc.stdout)
    audit = Path(match.group(1).strip()) if match else None
    entries = []
    if audit and (audit / "audit.jsonl").is_file():
        entries = [json.loads(l) for l in (audit / "audit.jsonl").read_text(encoding="utf-8").splitlines() if l]
    tests = [e for e in entries if e.get("step") == "tests"]
    finish = next((e for e in reversed(entries) if e.get("step") == "finish"), {})
    return {"green": proc.returncode == 0, "audit": str(audit) if audit else None,
            "status": finish.get("status"), "rounds": finish.get("rounds"),
            "red_first": bool(tests) and tests[0].get("passed") is False}


def step_devagent() -> dict:
    model = f"{MODEL} (Ollama)"
    result = _dev_agent_once(None)
    if not result["green"] and os.getenv("GEMINI_API_KEY"):
        log("the local model did not finish in 3 rounds; retrying on Gemini, as the ticket allows")
        model = f"{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')} (Gemini)"
        result = _dev_agent_once({"AI_PROVIDER": "gemini"})
    result["model"] = model
    if result["audit"]:
        shutil.copytree(result["audit"], EVIDENCE / "devagent" / "audit", dirs_exist_ok=True)
    if not result["green"]:
        return {"ok": False, **result}
    patch = Path(result["audit"]) / "patch.diff"
    if sh(["git", "apply", "--check", str(patch)]).returncode != 0:
        return {"ok": False, **result, "note": "patch.diff does not apply to this checkout"}
    sh(["git", "apply", str(patch)])
    suite = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout=3600,
               out=EVIDENCE / "devagent" / "pytest-after-patch.txt")
    result["suite_after_patch"] = next((l for l in reversed(suite.stdout.splitlines()) if "passed" in l), "").strip("= ")
    ticket = ROOT / "docs" / "tickets" / "26-apa-7.md"
    line = (f"**Result:** {datetime.now():%Y-%m-%d}, {model}: {result['status']} in {result['rounds']} "
            f"round(s), {'red → green' if result['red_first'] else 'NOT red first'}, no hand edits. "
            f"The patch was applied with `git apply`; full suite after it: {result['suite_after_patch']}. "
            f"Audit copy: `docs/evidence/live-pass/devagent/audit/`.")
    text = ticket.read_text(encoding="utf-8")
    ticket.write_text(re.sub(r"\*\*Result:\*\*.*", lambda _: line, text, count=1), encoding="utf-8")
    return {"ok": suite.returncode == 0 and result["red_first"], **result}


def step_runs() -> dict:
    rows = []
    for label, question in RUN_QUESTIONS:
        proc = cli("run", "-q", question, "--json", timeout=3 * 3600, out=EVIDENCE / "runs" / f"{label}.txt")
        report = json_out(proc)
        folder = run_folder(report.get("run_id", "")) if report.get("run_id") else None
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if folder and (folder / "metadata.json").is_file() else {}
        memory = next((e.get("details", {}) for e in reversed(events_of(folder)) if e.get("status") == "memory_hit"), {}) if folder else {}
        if folder:
            keep(folder, EVIDENCE / "runs" / label)
        rows.append({"label": label, "question": question, "run_id": report.get("run_id"),
                     "state": report.get("state"), "draft_status": report.get("draft_status"),
                     "words": report.get("actual_words"), "target_words": report.get("target_words"),
                     "scores": report.get("scores"), "verification": report.get("verification"),
                     "discovery_seconds": meta.get("timings", {}).get("discovery_seconds"),
                     "memory_hit": memory})
    fresh = [r for r in rows if r["label"] != "related" and r["words"]]
    mean = round(sum(r["words"] for r in fresh) / len(fresh)) if fresh else None
    first, related = rows[0], rows[-1]
    return {"ok": all(r["state"] == "completed" for r in rows), "runs": rows,
            "mean_words_fresh": mean, "baseline_words": BASELINE_WORDS,
            "library": {"first_discovery_seconds": first["discovery_seconds"],
                        "related_discovery_seconds": related["discovery_seconds"],
                        "related_memory_hit": related["memory_hit"]}}


def step_advisor() -> dict:
    rows = []
    for position, domain in enumerate(ADVISOR_DOMAINS):
        out = EVIDENCE / "advisor" / f"{domain}.txt"
        if position == 0:  # the smoke test also hands the top pick to the pipeline
            proc = sh([PY, "scripts/smoke_test_trend_advisor.py", domain], timeout=4 * 3600, out=out)
        else:  # the advisor alone, in a child process like the smoke test
            proc = sh([PY, "scripts/live_pass.py", "--advisor-only", domain], timeout=3600, out=out)
        text = out.read_text(encoding="utf-8")
        saved = re.search(r"Saved to: (.+)", text)
        rows.append({"domain": domain, "ok": proc.returncode == 0,
                     "gap_flags": len(re.findall(r"^\s+gap: ", text, re.M)),
                     "topics": len(re.findall(r"^\d+\. ", text, re.M)),
                     "handed_off": "run directory:" in text,
                     "folder": saved.group(1).strip() if saved else None})
    return {"ok": all(r["ok"] for r in rows) and any(r["gap_flags"] for r in rows), "domains": rows}


def advisor_only(domain: str) -> None:
    """The shortlist for one field, printed like the smoke test (no hand-off)."""
    from uuid import uuid4
    sys.path.insert(0, str(ROOT))
    from rich.console import Console
    from researchgenie.advisor_cli import _run_advisor
    from shared.contracts.trend_contract import TrendAdvisorRequest
    found = _run_advisor(Console(legacy_windows=False), TrendAdvisorRequest(domain=domain), uuid4().hex)
    if found is None:
        raise SystemExit("FAILED: no shortlist")
    result, folder = found
    print(f"Saved to: {folder}")
    for position, c in enumerate(result.shortlist, start=1):
        print(f"{position}. {c.topic}  [{c.signal_label}]  growth {c.growth_metric}x")
        for evidence in c.gap_evidence:
            print(f"   gap: {evidence}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")


def step_docker() -> dict:
    up = sh(["docker", "compose", "--profile", "local", "up", "-d", "--build"], timeout=3600,
            out=EVIDENCE / "docker" / "compose-up.txt")
    if up.returncode != 0:
        return {"ok": False, "note": "compose up failed; see docker/compose-up.txt"}
    system = {}
    for _ in range(180):  # the model pull inside the container can take a while
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/system", timeout=5) as resp:
                system = json.load(resp)
            if system.get("model_ready"):
                break
        except OSError:
            pass
        time.sleep(20)
    (EVIDENCE / "docker" / "system.json").write_text(json.dumps(system, indent=2), encoding="utf-8")
    return {"ok": bool(system.get("model_ready")), "system": system}


STEPS = {"preflight": step_preflight, "tests": step_tests, "spike": step_spike,
         "doctor": step_doctor, "devagent": step_devagent, "runs": step_runs,
         "advisor": step_advisor, "docker": step_docker}
DEFAULT = ["preflight", "tests", "spike", "doctor", "devagent", "runs", "advisor"]


# ------------------------------------------------------------------- summary

def summary() -> Path:
    def load(name: str) -> dict:
        path = EVIDENCE / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    mark = lambda r: "not run" if not r else ("PASS" if r.get("ok") else "FAIL")  # noqa: E731
    def row(check: str, ticket: str, result: dict, measured: str) -> str:
        return f"| {check} | {ticket} | {mark(result)} | {measured if result else ''} |"

    pre, tests, spike, doc, dev, runs, adv, dock = map(load, STEPS)
    library = runs.get("library") or {}
    lines = [
        "# Live pass",
        "",
        f"Written by `scripts/live_pass.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
        f"Provider: {pre.get('provider', '?')} {pre.get('model', '')}, commit {pre.get('commit', '?')}. "
        "Every number below is measured, failures included.",
        "",
        "| Check | Ticket | Result | Measured |",
        "|---|---|---|---|",
        row("Full test suite", "—", tests, tests.get("summary", "")),
        row("Tool-calling spike", "#16", spike, spike.get("summary", spike.get("note", ""))
            + ("; model step stays on" if spike.get("keep_model_step") else "")),
        row("Doctor recovers a run after Ollama is stopped", "#13", doc,
            f"{doc.get('cause', doc.get('note', ''))} → {doc.get('fix', '')} → {doc.get('outcome', '')}"),
        row("Dev Agent APA 7, red → green, no hand edits", "#26", dev,
            f"{dev.get('model', '')}: {dev.get('status', '')} in {dev.get('rounds', '?')} round(s), "
            f"red first: {dev.get('red_first')}; suite after patch: {dev.get('suite_after_patch', '')}"),
        row(f"Fresh full runs vs the {BASELINE_WORDS:,}-word baseline (D-039)", "#27", runs,
            f"mean {runs.get('mean_words_fresh')} words over the fresh runs"),
        row("Library reuse on a related run", "#19", runs,
            f"Discovery {library.get('first_discovery_seconds')} s → "
            f"{library.get('related_discovery_seconds')} s; memory_hit {library.get('related_memory_hit')}"),
        row("Trend Advisor on the real model", "#27", adv,
            "; ".join(f"{d['domain']}: {d['topics']} topics, {d['gap_flags']} gap statements"
                      f"{', handed off' if d['handed_off'] else ''}" for d in adv.get("domains", []))),
        row("Container, local profile", "#23", dock,
            f"model_ready: {(dock.get('system') or {}).get('model_ready')}"),
        "",
        "## Runs",
        "",
        "| Run | draft_status | Words | QA overall | Verified / invalid / unverifiable | Discovery s |",
        "|---|---|---|---|---|---|",
    ]
    for r in runs.get("runs", []):
        v = r.get("verification") or {}
        lines.append(f"| {r['label']} | {r.get('draft_status')} | {r.get('words')} | "
                     f"{(r.get('scores') or {}).get('overall')} | {v.get('verified')} / {v.get('invalid')} / "
                     f"{v.get('unverifiable')} | {r.get('discovery_seconds')} |")
    lines += ["", "Raw output for each step is beside this file (`<step>.json`, `<step>/`)."]
    path = EVIDENCE / "LIVE_PASS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _mark_sheet({name: load(name) for name in STEPS})
    return path


def _mark_sheet(results: dict[str, dict]) -> None:
    """Sets each `data-live` chip in the final lab sheet from what was measured."""
    sheet = ROOT / "docs" / "lab-worklet-final.html"
    if not sheet.is_file():
        return

    def chip(match: re.Match) -> str:
        result = results.get(match.group(1))
        if not result:
            return match.group(0)
        css, text = ("ok", "Live: shown") if result.get("ok") else ("no", "Live: failed")
        return f'<span class="chip {css}" data-live="{match.group(1)}">{text}</span>'

    text = sheet.read_text(encoding="utf-8")
    sheet.write_text(re.sub(r'<span class="chip \w+" data-live="(\w+)">[^<]*</span>', chip, text),
                     encoding="utf-8")


def push() -> None:
    sh(["git", "add", "--", "docs", "services", "tests", "shared"])
    staged = sh(["git", "diff", "--cached", "--stat"]).stdout
    if not staged.strip():
        log("nothing to commit")
        return
    message = ("Record the live pass: spike, Doctor, Dev Agent, fresh runs, advisor\n\n"
               "Evidence in docs/evidence/live-pass/ (written by scripts/live_pass.py).\n")
    sh(["git", "commit", "-m", message])
    proc = sh(["git", "push"])
    log(proc.stdout + proc.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=list(STEPS))
    parser.add_argument("--redo", action="store_true", help="run steps that already passed again")
    parser.add_argument("--push", action="store_true", help="commit docs/evidence and the APA patch, then push")
    parser.add_argument("--advisor-only", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.advisor_only:
        advisor_only(args.advisor_only)
        return
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in args.only or DEFAULT:
        path = EVIDENCE / f"{name}.json"
        if not args.redo and path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("ok"):
            log(f"{name}: passed before, skipped (--redo to repeat)")
            continue
        log(f"=== {name} ===")
        started = time.time()
        try:
            result = STEPS[name]()
        except Exception as error:  # one broken step must not lose the others' evidence
            result = {"ok": False, "note": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.time() - started)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"{name}: {'PASS' if result.get('ok') else 'FAIL'} {result.get('note', '')}")
        if name == "preflight" and not result["ok"]:
            break
        if name == "doctor" and not ollama_up():
            log("Ollama is still down after the Doctor step; starting it for the next steps")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
    log(f"summary: {summary()}")
    if args.push:
        push()


if __name__ == "__main__":
    main()
