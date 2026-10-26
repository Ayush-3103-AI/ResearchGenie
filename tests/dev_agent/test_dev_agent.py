"""The Dev Agent (#25) on a throwaway fixture repo, with a scripted model.

Every test runs the real graph: a real git worktree, real file writes
through the path guard, and a real pytest subprocess. Patches are
search/replace edits; the existing tests that use a changed file guard it.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev_agent import agent  # noqa: E402
from shared.utilities.llm_provider import Completion  # noqa: E402

TICKET = "Add subtract(a, b) to calc.py; it returns a - b."
TEST = "from calc import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n"
BASE = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a + b\n"
RIGHT = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
GUARD = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _edits(*edits):
    return {"edits": [{"path": path, "old": old, "new": new} for path, old, new in edits]}


ADD_WRONG = _edits(("calc.py", BASE, WRONG))
ADD_RIGHT = _edits(("calc.py", BASE, RIGHT))
FIX_WRONG = _edits(("calc.py", "def subtract(a, b):\n    return a + b\n",
                    "def subtract(a, b):\n    return a - b\n"))


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "fixture-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text(BASE, encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_add.py").write_text(  # an existing test that uses calc.py
        "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parents[1]))\n"
        + GUARD, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "base")
    return repo


PLAN = {"summary": "add subtract", "files": ["calc.py"], "test_file": "tests/test_calc.py"}


def _model(*patches, plan=PLAN):
    """Answers the plan, then the test, then each patch round in turn (the
    last patch repeats). Records every prompt it was sent, and its options."""
    prompts, options = [], []
    replies = iter([plan, {"files": [{"path": "tests/test_calc.py", "content": TEST}]}])

    async def complete(*, system, user, schema, **kwargs):
        prompts.append(user)
        options.append(kwargs)
        reply = next(replies, None)
        if reply is None:
            patch_round = sum("Patch round" in p for p in prompts) - 1
            reply = patches[min(patch_round, len(patches) - 1)]
        return Completion(text=json.dumps(reply))

    complete.prompts, complete.options = prompts, options
    return complete


def _run(repo, tmp_path, model):
    return asyncio.run(agent.run_dev_agent(TICKET, repo=repo, complete=model,
                                           audit_root=tmp_path / "audit", test_timeout=120))


def _audit(report):
    return [json.loads(line) for line in
            (Path(report["audit_dir"]) / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_reaches_green_within_three_rounds_and_the_audit_shows_red_then_green(repo, tmp_path):
    model = _model(ADD_WRONG, FIX_WRONG)
    report = _run(repo, tmp_path, model)

    assert report["status"] == "green" and report["rounds"] == 2
    runs = [e["passed"] for e in _audit(report) if e["step"] == "tests"]
    assert runs == [False, False, True]  # red before the patch, red, then green
    assert "return a - b" in report["diff"] and "def test_subtract" in report["diff"]
    assert "AssertionError" in model.prompts[-1] or "assert" in model.prompts[-1]  # pytest output fed back
    assert (Path(report["audit_dir"]) / "patch.diff").read_text(encoding="utf-8") == report["diff"]
    # The real checkout is untouched; the work happened in a worktree.
    assert "subtract" not in (repo / "calc.py").read_text(encoding="utf-8")


def test_a_write_outside_the_planned_paths_is_refused(repo, tmp_path):
    model = _model(_edits(("calc.py", BASE, RIGHT), ("secrets.py", "", "x = 1\n"),
                          ("../escape.py", "", "x = 1\n"), ('calc.py"}, {"path', "", "x"),
                          ("tests/test_calc.py", TEST, "def test_x():\n    pass\n")))
    report = _run(repo, tmp_path, model)

    refused = [e["path"] for e in _audit(report) if e["step"] == "write_refused"]
    assert sorted(refused) == ["../escape.py", 'calc.py"}, {"path', "secrets.py", "tests/test_calc.py"]
    assert "secrets.py" not in report["diff"] and not (tmp_path / "escape.py").exists()
    assert report["status"] == "green"  # the planned write still went through


def test_stops_at_three_rounds_with_an_honest_failure_report(repo, tmp_path):
    report = _run(repo, tmp_path, _model(ADD_WRONG, _edits()))

    assert report["status"] == "failed" and report["rounds"] == 3
    assert [e["passed"] for e in _audit(report) if e["step"] == "tests"] == [False] * 4
    assert "did not reach green" in report["summary"] and "assert" in report["summary"]


def test_no_commit_push_or_merge_happens_in_any_path(repo, tmp_path):
    before = (_git(repo, "rev-list", "--all", "--count"), _git(repo, "branch", "--list"),
              _git(repo, "rev-parse", "HEAD"))
    _run(repo, tmp_path, _model(ADD_RIGHT))
    _run(repo, tmp_path, _model(ADD_WRONG, _edits()))
    after = (_git(repo, "rev-list", "--all", "--count"), _git(repo, "branch", "--list"),
             _git(repo, "rev-parse", "HEAD"))
    assert before == after
    assert _git(repo, "worktree", "list").count("\n") == 0  # worktrees cleaned up

    for forbidden in (["commit", "-m", "x"], ["push"], ["merge", "main"], ["-c", "a=b", "commit"]):
        with pytest.raises(agent.DevAgentError):
            agent.git(repo, *forbidden)


def test_the_entry_point_takes_ticket_text_or_a_spec_file(tmp_path):
    spec = tmp_path / "ticket.md"
    spec.write_text("# Add subtract\nIt returns a - b.\n", encoding="utf-8")
    assert agent.read_ticket(["--spec", str(spec)]) == spec.read_text(encoding="utf-8")
    assert agent.read_ticket(["Add", "subtract"]) == "Add subtract"


# ------------------------------------------------- edits, guards, feedback (#26 run)

def test_a_patch_is_edits_so_untouched_lines_are_never_retyped(repo, tmp_path):
    model = _model(ADD_RIGHT)
    report = _run(repo, tmp_path, model)
    assert report["status"] == "green"
    assert "Return search/replace edits" in model.prompts[-1]
    assert all(o.get("think") is False and o.get("max_tokens") == agent.MAX_TOKENS
               for o in model.options)
    # The line that already existed is untouched in the diff.
    assert "-    return a + b" not in report["diff"] and "+    return a - b" in report["diff"]


def test_an_edit_that_does_not_match_once_is_not_applied_and_is_fed_back(repo, tmp_path):
    missing = _edits(("calc.py", "def add(a,b):\n", "x"))  # not in the file as written
    model = _model(missing, ADD_RIGHT)
    report = _run(repo, tmp_path, model)

    assert report["status"] == "green" and report["rounds"] == 2
    [first] = [e for e in _audit(report) if e["step"] == "edit" and e["during"] == "patch_1"]
    assert first["paths"] == [] and "occurs 0 time(s)" in first["problems"][0]
    assert "were not applied" in model.prompts[-1] and "def add(a,b)" in model.prompts[-1]


def test_the_model_sees_its_own_changes_so_far(repo, tmp_path):
    model = _model(ADD_WRONG, FIX_WRONG)
    _run(repo, tmp_path, model)
    patch_prompts = [p for p in model.prompts if "Patch round" in p]
    assert "(none yet)" in patch_prompts[0]
    assert "+def subtract(a, b):" in patch_prompts[1] and "Undo any change" in patch_prompts[1]


def test_breaking_an_unrelated_line_is_caught_by_the_existing_tests(repo, tmp_path):
    """The failure the local model kept making: the acceptance test passes,
    but an untouched function is corrupted. The guard keeps it red."""
    corrupt = _edits(("calc.py", BASE, RIGHT.replace("return a + b", "return a * b")))
    model = _model(corrupt, _edits(("calc.py", "return a * b", "return a + b")))
    report = _run(repo, tmp_path, model)

    guards = [e["tests"] for e in _audit(report) if e["step"] == "guards"]
    assert guards == [["tests/test_add.py"]]
    runs = [(e["round"], e["passed"]) for e in _audit(report) if e["step"] == "tests"]
    assert runs == [(0, False), (1, False), (2, True)]  # round 1 was red only because of test_add
    patch_prompts = [p for p in model.prompts if "Patch round" in p]
    assert "tests/test_add.py" in patch_prompts[1] and "test_add" in patch_prompts[1]
    assert report["status"] == "green" and "1 existing test file" in report["summary"]


def test_the_plan_dedupes_its_test_file_and_refuses_to_reuse_an_existing_one(repo, tmp_path):
    listed_twice = {**PLAN, "files": ["calc.py", "tests/test_calc.py"]}
    report = _run(repo, tmp_path, _model(ADD_RIGHT, plan=listed_twice))
    assert report["status"] == "green"
    [plan] = [e["plan"] for e in _audit(report) if e["step"] == "plan"]
    assert plan["files"] == ["calc.py"]

    reuse = {**PLAN, "test_file": "tests/test_add.py"}
    report = _run(repo, tmp_path, _model(ADD_RIGHT, plan=reuse))
    assert report["status"] == "failed" and "already exists" in report["summary"]
    assert "test_add" in (repo / "tests" / "test_add.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", ['calc.py", "x', "calc.py}", "my file.py", "../calc.py", "C:/x.py", ""])
def test_garbled_or_escaping_paths_are_refused(path):
    assert agent._safe_path(path) is None
    assert agent._safe_path("services/research-writing/writing/x.py") == "services/research-writing/writing/x.py"


def test_the_test_writer_is_shown_how_tests_here_import_code(repo, tmp_path):
    model = _model(ADD_RIGHT)
    _run(repo, tmp_path, model)
    test_prompt = model.prompts[1]
    assert "tests/test_add.py (its imports and setup)" in test_prompt
    assert "sys.path.insert" in test_prompt and "def test_add" not in test_prompt


def test_crlf_files_keep_their_line_endings(repo, tmp_path):
    _git(repo, "config", "core.autocrlf", "false")  # keep the CRLF in the repo and the worktree
    (repo / "calc.py").write_bytes(BASE.replace("\n", "\r\n").encode())
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qam", "crlf")
    report = _run(repo, tmp_path, _model(ADD_RIGHT))
    assert report["status"] == "green"
    assert "-def add" not in report["diff"] and "+def subtract" in report["diff"]
