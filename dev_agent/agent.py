"""Dev Agent (#25): ticket -> plan -> failing test -> patch -> green, then a
diff for a human to approve.

A LangGraph state graph reads a ticket, plans the files to change and the
acceptance test, writes the test, runs it (it must fail), then patches and
runs the tests again, feeding pytest's output back each time, for at most
MAX_ROUNDS patch rounds. It works in a detached git worktree, writes only to
the paths in its plan (the patch may not touch its own test), runs pytest in
a subprocess with a timeout, calls the model only through
shared/utilities/llm_provider.py, and logs every step to an audit trail.

A patch is a list of search/replace edits, never a retyped file: a small
local model copying a whole file back inside a JSON string corrupts lines
it never meant to touch (a regex's backslashes, most often). Each round the
model also sees the diff of its own changes so far and any edit that didn't
apply, and the existing tests that use the changed files run beside the
acceptance test, so a broken unrelated line can't come back green.

It never commits, pushes or merges: every git call goes through `git()`,
which only allows the read-only and worktree commands listed below. The
result is a patch file the human applies with `git apply` after review.

    researchgenie-dev "Add subtract(a, b) to calc.py"
    researchgenie-dev --spec ticket.md [--repo PATH] [--timeout SECONDS]
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.utilities import llm_provider  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402

MAX_ROUNDS = 3
TEST_TIMEOUT = 300.0     # seconds per pytest run
OUTPUT_TAIL = 4000       # characters of pytest output fed back to the model
REPO_LISTING = 400       # files of the repo shown to the planner
MAX_TOKENS = 16000       # a patch round's answer; thinking is off so all of it is the answer
GUARD_TESTS = 10         # existing tests that use the changed files, run beside the new one
DIFF_TAIL = 6000         # characters of "your changes so far" shown each round
_PATH_CHARS = re.compile(r"[\w.\-/]+")  # what a repo path may contain; a garbled one is refused
# git subcommands the agent may run; anything else (commit, push, merge,
# rebase, reset, checkout, ...) is refused before a process starts.
GIT_ALLOWED = {"worktree": {"add", "remove", "prune"}, "add": {"--intent-to-add"},
               "diff": None, "ls-files": None, "rev-parse": None, "status": None}

Complete = Callable[..., Awaitable[llm_provider.Completion]]


class DevAgentError(RuntimeError):
    """A refused git command, or a plan the agent cannot act on."""


def git(repo: Path, *args: str) -> str:
    """Run an allow-listed git command in `repo` and return its stdout."""
    rest = list(args)
    while rest and rest[0] == "-c":  # `-c key=value` pairs before the subcommand
        rest = rest[2:]
    command = rest[0] if rest else ""
    allowed = GIT_ALLOWED.get(command, ())
    if command not in GIT_ALLOWED or (allowed and not set(rest[1:2]) & allowed):
        raise DevAgentError(f"The Dev Agent never runs `git {' '.join(args)}`.")
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if done.returncode:
        raise DevAgentError(f"git {command} failed: {done.stderr.strip()}")
    return done.stdout


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"},
                  "description": "Repo-relative paths the patch will create or change."},
        "test_file": {"type": "string",
                      "description": "Repo-relative path of the new acceptance test file."},
    },
    "required": ["summary", "files", "test_file"],
}
FILES_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
    "required": ["files"],
}
EDITS_SCHEMA = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Text copied exactly from the current file, "
                    "with enough lines to occur only once. Empty only to create a new file."},
            "new": {"type": "string", "description": "The text that replaces it."}},
        "required": ["path", "old", "new"]}}},
    "required": ["edits"],
}
SYSTEM = ("You are a careful software engineer working test-first on a small, well-specified "
          "ticket. Change as little as possible. Answer only with the requested JSON. "
          "Change existing files with search/replace edits: copy only the lines you change, plus "
          "enough context to be unique, exactly as they are in the file. Never retype a whole file.")


class State(TypedDict, total=False):
    ticket: str
    worktree: str
    plan: dict
    guards: list[str]   # existing tests that use the changed files and passed before the patch
    passed: bool
    output: str
    problems: list[str]  # last round's edits that were not applied, and why
    rounds: int
    status: str


def _safe_path(path: str) -> str | None:
    """The normalized repo-relative POSIX path, or None if it escapes the repo."""
    pure = PurePosixPath(path.strip().replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or ":" in pure.parts[0] \
            or not _PATH_CHARS.fullmatch(str(pure)):
        return None  # escapes the repo, or garbled (quotes, braces, spaces from broken JSON)
    return str(pure)


class _Run:
    """One agent run: its worktree, its audit trail, and the model."""

    def __init__(self, repo: Path, audit_dir: Path, complete: Complete, test_timeout: float):
        self.repo, self.audit_dir = repo, audit_dir
        self.complete, self.test_timeout = complete, test_timeout
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, **fields: Any) -> None:
        entry = {"step": step, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        with (self.audit_dir / "audit.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def ask(self, user: str, schema: dict) -> dict:
        comp = await self.complete(system=SYSTEM, user=user, schema=schema,
                                   max_tokens=MAX_TOKENS, think=False)
        if comp.stop != "end" or not comp.text.strip():
            raise DevAgentError(f"The model gave no usable answer (stop: {comp.stop}).")
        try:
            return json.loads(comp.text)
        except json.JSONDecodeError:
            raise DevAgentError("The model's answer was not valid JSON.") from None

    def read(self, worktree: Path, path: str) -> str:
        file = worktree / path
        return file.read_text(encoding="utf-8") if file.is_file() else "(new file)"

    def write(self, worktree: Path, allowed: set[str], files: list[dict], step: str) -> list[str]:
        """Writes each file whose path is in `allowed`; refuses (and logs) the rest."""
        written = []
        for item in files:
            path = _safe_path(item.get("path", ""))
            if path is None or path not in allowed:
                self.log("write_refused", path=item.get("path"), during=step,
                         allowed=sorted(allowed))
                continue
            target = worktree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.get("content", ""), encoding="utf-8", newline="")
            written.append(path)
        self.log("write", during=step, paths=written)
        return written

    def edit(self, worktree: Path, allowed: set[str], edits: list[dict], step: str) -> list[str]:
        """Applies each search/replace edit to a path in `allowed` whose `old`
        text occurs exactly once (an empty `old` creates a new file). Returns
        the problems with the edits that were not applied, for the next round.
        The file keeps its own line endings."""
        # ponytail: exact match only; add a whitespace-tolerant match if models keep missing on indentation.
        applied, problems = [], []
        for item in edits:
            path = _safe_path(str(item.get("path", "")))
            if path is None or path not in allowed:
                self.log("write_refused", path=item.get("path"), during=step, allowed=sorted(allowed))
                continue
            old = str(item.get("old", "")).replace("\r\n", "\n")
            new = str(item.get("new", "")).replace("\r\n", "\n")
            target = worktree / path
            if not old:
                if target.exists():
                    problems.append(f"{path}: `old` was empty but the file exists; copy the lines "
                                    "to replace into `old`.")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(new, encoding="utf-8", newline="")
                applied.append(path)
                continue
            raw = target.read_bytes().decode("utf-8") if target.is_file() else ""
            text = raw.replace("\r\n", "\n")
            found = text.count(old)
            if found != 1:
                problems.append(f"{path}: this `old` text occurs {found} time(s) in the file, not "
                                "once, so the edit was not applied. Copy it exactly from the "
                                f"current file, with more lines if needed:\n{old[:400]}")
                continue
            text = text.replace(old, new, 1)
            target.write_bytes((text.replace("\n", "\r\n") if "\r\n" in raw else text).encode("utf-8"))
            applied.append(path)
        self.log("edit", during=step, paths=applied, problems=problems)
        return problems

    def guard_tests(self, worktree: Path, plan: dict) -> list[str]:
        """Existing test files that mention a changed module by name."""
        stems = {PurePosixPath(f).stem for f in plan["files"] if f.endswith(".py")} - {"__init__"}
        found = []
        for path in git(worktree, "ls-files").splitlines():
            name = PurePosixPath(path).name
            if path == plan["test_file"] or not (name.startswith("test_") and name.endswith(".py")):
                continue
            text = (worktree / path).read_text(encoding="utf-8", errors="replace")
            if any(re.search(rf"\b{re.escape(stem)}\b", text) for stem in stems):
                found.append(path)
        return found[:GUARD_TESTS]

    def example_test(self, worktree: Path, test_file: str) -> str:
        """The setup and imports of an existing test beside the new one, so the
        model copies this repo's real import pattern (hyphenated service folders
        are put on sys.path, not imported as packages)."""
        folder = str(PurePosixPath(test_file).parent)
        siblings = sorted(p for p in git(worktree, "ls-files").splitlines()
                          if str(PurePosixPath(p).parent) == folder and p != test_file
                          and PurePosixPath(p).name.startswith("test_") and p.endswith(".py"))
        if not siblings:
            return ""
        head = []
        for line in (worktree / siblings[0]).read_text(encoding="utf-8").splitlines()[:60]:
            if line.startswith(("def ", "class ", "@")):
                break
            head.append(line)
        return f"--- {siblings[0]} (its imports and setup) ---\n" + "\n".join(head).strip()

    async def pytest(self, worktree: Path, *test_files: str) -> tuple[bool, str]:
        # ponytail: the acceptance test and its guards only; the human's review and CI run the full suite.
        # -B: no .pyc between rounds. A same-size patch written within the same second
        # as the last one would otherwise run the previous round's cached bytecode.
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files]
        try:
            done = await asyncio.to_thread(subprocess.run, command, cwd=worktree,
                                           capture_output=True, text=True,
                                           timeout=self.test_timeout)
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {self.test_timeout:.0f} s."
        return done.returncode == 0, (done.stdout + done.stderr)[-OUTPUT_TAIL:]


def _build_graph(run: _Run):
    async def plan(state: State) -> dict:
        worktree = Path(state["worktree"])
        listing = git(worktree, "ls-files").splitlines()[:REPO_LISTING]
        plan = await run.ask(f"Ticket:\n{state['ticket']}\n\nRepository files:\n"
                             + "\n".join(listing)
                             + "\n\nPlan the smallest change: the files to create or edit, "
                               "and one new pytest file that fails until the ticket is done.",
                             PLAN_SCHEMA)
        test_file = _safe_path(str(plan.get("test_file", "")))
        files = [_safe_path(str(f)) for f in plan.get("files", [])]
        files = list(dict.fromkeys(f for f in files if f != test_file))  # the test is written separately
        if not files or None in files or test_file is None:
            raise DevAgentError(f"The plan is unusable: {plan}")
        if (worktree / test_file).exists():
            # Writing it would replace an existing test file with the new acceptance test.
            raise DevAgentError(f"The plan's test file {test_file} already exists; the acceptance "
                                "test must be a new file.")
        plan = {**plan, "files": files, "test_file": test_file}
        run.log("plan", plan=plan)
        return {"plan": plan, "rounds": 0, "problems": []}

    async def write_test(state: State) -> dict:
        worktree, plan = Path(state["worktree"]), state["plan"]
        sources = "\n\n".join(f"--- {f} ---\n{run.read(worktree, f)}" for f in plan["files"])
        example = run.example_test(worktree, plan["test_file"])
        reply = await run.ask(f"Ticket:\n{state['ticket']}\n\nPlan: {plan.get('summary', '')}\n\n"
                              f"Current files:\n{sources}\n\n"
                              + (f"An existing test in the same folder; import the code the same "
                                 f"way:\n{example}\n\n" if example else "")
                              + f"Write the acceptance test {plan['test_file']}, as a complete new "
                              "file. It must fail now and pass once the ticket is done.", FILES_SCHEMA)
        run.write(worktree, {plan["test_file"]}, reply.get("files", []), "write_test")
        return {}

    async def red(state: State) -> dict:
        """The acceptance test must fail now. The guards are kept only if they
        pass now: an existing test that already fails can't vouch for the patch."""
        worktree = Path(state["worktree"])
        passed, output = await run.pytest(worktree, state["plan"]["test_file"])
        run.log("tests", round=0, passed=passed, output=output)
        guards = []
        for guard in run.guard_tests(worktree, state["plan"]):
            ok, _ = await run.pytest(worktree, guard)
            if ok:
                guards.append(guard)
            else:
                run.log("guard_dropped", test=guard)
        run.log("guards", tests=guards)
        return {"passed": passed, "output": output, "guards": guards}

    async def run_tests(state: State) -> dict:
        passed, output = await run.pytest(Path(state["worktree"]), state["plan"]["test_file"],
                                          *state.get("guards", []))
        run.log("tests", round=state["rounds"], passed=passed, output=output)
        return {"passed": passed, "output": output}

    async def patch(state: State) -> dict:
        worktree, plan = Path(state["worktree"]), state["plan"]
        n = state["rounds"] + 1
        sources = "\n\n".join(f"--- {f} ---\n{run.read(worktree, f)}" for f in plan["files"])
        diff = git(worktree, "diff", "--", *plan["files"])[-DIFF_TAIL:]
        guards = state.get("guards", [])
        problems = "\n".join(f"- {p}" for p in state.get("problems", []))
        reply = await run.ask(
            f"Patch round {n} of {MAX_ROUNDS}.\n\nTicket:\n{state['ticket']}\n\n"
            f"Acceptance test ({plan['test_file']}, do not change it):\n"
            f"{run.read(worktree, plan['test_file'])}\n\n"
            f"Files you may change (current content):\n{sources}\n\n"
            f"Your changes so far (git diff against the original):\n{diff or '(none yet)'}\n"
            "Undo any change in it that the ticket doesn't need.\n\n"
            + (f"Edits from the last round that were not applied:\n{problems}\n\n" if problems else "")
            + "Last pytest output (the acceptance test"
            + (f", and existing tests that use these files and must keep passing: {', '.join(guards)}"
               if guards else "") + f"):\n{state['output']}\n\n"
            "Return search/replace edits.", EDITS_SCHEMA)
        problems = run.edit(worktree, set(plan["files"]), reply.get("edits", []), f"patch_{n}")
        return {"rounds": n, "problems": problems}

    def after_red(state: State) -> str:
        return "finish" if state["passed"] else "patch"  # a test that passes first proves nothing

    def after_tests(state: State) -> str:
        return "finish" if state["passed"] or state["rounds"] >= MAX_ROUNDS else "patch"

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("write_test", write_test)
    graph.add_node("red", red)
    graph.add_node("patch", patch)
    graph.add_node("tests", run_tests)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "write_test")
    graph.add_edge("write_test", "red")
    graph.add_conditional_edges("red", after_red, {"patch": "patch", "finish": END})
    graph.add_edge("patch", "tests")
    graph.add_conditional_edges("tests", after_tests, {"patch": "patch", "finish": END})
    return graph.compile()


def _report(state: State, diff: str, audit_dir: Path) -> dict:
    rounds, passed = state.get("rounds", 0), state.get("passed", False)
    if passed and rounds:
        status = "green"
        guards = state.get("guards", [])
        summary = (f"The acceptance test is green after {rounds} patch round(s)"
                   + (f", and {len(guards)} existing test file(s) that use the changed files still "
                      "pass" if guards else "") + "; the rest of the suite was not run. Review "
                   "patch.diff, run the full tests, then `git apply` it.")
    elif passed:
        status = "not_red"
        summary = ("The new test passed before any patch, so it does not test the ticket. "
                   "Nothing was patched.")
    else:
        status = "failed"
        summary = (f"Stopped after {rounds} patch round(s): the test did not reach green. "
                   f"Last pytest output:\n{state.get('output', '')}")
    return {"status": status, "rounds": rounds, "summary": summary, "diff": diff,
            "audit_dir": str(audit_dir)}


async def run_dev_agent(ticket: str, *, repo: Path | str = ROOT, complete: Complete | None = None,
                        audit_root: Path | None = None,
                        test_timeout: float = TEST_TIMEOUT) -> dict:
    """Runs the agent on `ticket` in a throwaway worktree of `repo` and
    returns {status, rounds, summary, diff, audit_dir}. status is "green",
    "failed" or "not_red"; the diff is also saved as <audit_dir>/patch.diff."""
    repo = Path(git(Path(repo), "rev-parse", "--show-toplevel").strip())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    audit_dir = (audit_root or ROOT / "outputs" / "dev_agent") / f"{stamp}_{slugify(ticket[:60])}"
    run = _Run(repo, audit_dir, complete or llm_provider.complete_json, test_timeout)
    worktree = Path(tempfile.mkdtemp(prefix="dev-agent-")) / "worktree"
    git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    run.log("start", ticket=ticket, repo=str(repo), worktree=str(worktree))
    state: State = {"ticket": ticket, "worktree": str(worktree), "rounds": 0}
    try:
        try:
            state = await _build_graph(run).ainvoke(state)
        except DevAgentError as error:
            run.log("error", message=str(error))
            state = {**state, "passed": False, "output": str(error)}
        plan = state.get("plan") or {}
        paths = [p for p in [*plan.get("files", []), plan.get("test_file")] if p]
        if paths:
            git(worktree, "add", "--intent-to-add", "--", *paths)  # so new files show in the diff
        diff = git(worktree, "diff", "--", *paths) if paths else ""
    finally:
        git(repo, "worktree", "remove", "--force", str(worktree))
    report = _report(state, diff, audit_dir)
    (audit_dir / "patch.diff").write_text(diff, encoding="utf-8")
    (audit_dir / "report.md").write_text(f"# Dev Agent report\n\n{ticket}\n\n**{report['status']}**"
                                         f" — {report['summary']}\n", encoding="utf-8")
    run.log("finish", status=report["status"], rounds=report["rounds"])
    return report


def read_ticket(args: list[str]) -> str:
    """The ticket from `--spec FILE`, or the remaining arguments as text."""
    if "--spec" in args:
        return Path(args[args.index("--spec") + 1]).read_text(encoding="utf-8")
    return " ".join(args).strip()


def _option(args: list[str], name: str, default: str | None = None) -> tuple[str | None, list[str]]:
    if name not in args:
        return default, args
    i = args.index(name)
    return args[i + 1], args[:i] + args[i + 2:]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    repo, args = _option(args, "--repo", str(ROOT))
    timeout, args = _option(args, "--timeout", str(TEST_TIMEOUT))
    ticket = read_ticket(args)
    if not ticket:
        print(__doc__.split("\n\n")[-1])
        sys.exit(2)
    report = asyncio.run(run_dev_agent(ticket, repo=repo, test_timeout=float(timeout)))
    print(report["summary"])
    print(f"\nAudit trail and patch: {report['audit_dir']}")
    if report["diff"]:
        print("\n" + report["diff"])
    sys.exit(0 if report["status"] == "green" else 1)


if __name__ == "__main__":
    main()
