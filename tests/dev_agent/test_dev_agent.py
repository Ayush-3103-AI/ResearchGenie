"""The Dev Agent (#25) on a throwaway fixture repo, with a scripted model.

Every test runs the real graph: a real git worktree, real file writes
through the path guard, and a real pytest subprocess.
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
WRONG = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a + b\n"
RIGHT = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "fixture-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "base")
    return repo


def _model(*patches):
    """Answers the plan, then the test, then each patch round in turn (the
    last patch repeats). Records every prompt it was sent."""
    prompts = []
    replies = iter([
        {"summary": "add subtract", "files": ["calc.py"], "test_file": "tests/test_calc.py"},
        {"files": [{"path": "tests/test_calc.py", "content": TEST}]},
    ])

    async def complete(*, system, user, schema, **kwargs):
        prompts.append(user)
        reply = next(replies, None)
        if reply is None:
            patch_round = sum("Patch round" in p for p in prompts) - 1
            reply = patches[min(patch_round, len(patches) - 1)]
        return Completion(text=json.dumps(reply))

    complete.prompts = prompts
    return complete


def _run(repo, tmp_path, model):
    return asyncio.run(agent.run_dev_agent(TICKET, repo=repo, complete=model,
                                           audit_root=tmp_path / "audit", test_timeout=120))


def _audit(report):
    return [json.loads(line) for line in
            (Path(report["audit_dir"]) / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_reaches_green_within_three_rounds_and_the_audit_shows_red_then_green(repo, tmp_path):
    model = _model({"files": [{"path": "calc.py", "content": WRONG}]},
                   {"files": [{"path": "calc.py", "content": RIGHT}]})
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
    model = _model({"files": [{"path": "calc.py", "content": RIGHT},
                              {"path": "secrets.py", "content": "x = 1\n"},
                              {"path": "../escape.py", "content": "x = 1\n"},
                              {"path": "tests/test_calc.py", "content": "def test_x():\n    pass\n"}]})
    report = _run(repo, tmp_path, model)

    refused = [e["path"] for e in _audit(report) if e["step"] == "write_refused"]
    assert sorted(refused) == ["../escape.py", "secrets.py", "tests/test_calc.py"]
    assert "secrets.py" not in report["diff"] and not (tmp_path / "escape.py").exists()
    assert report["status"] == "green"  # the planned write still went through


def test_stops_at_three_rounds_with_an_honest_failure_report(repo, tmp_path):
    report = _run(repo, tmp_path, _model({"files": [{"path": "calc.py", "content": WRONG}]}))

    assert report["status"] == "failed" and report["rounds"] == 3
    assert [e["passed"] for e in _audit(report) if e["step"] == "tests"] == [False] * 4
    assert "did not reach green" in report["summary"] and "assert" in report["summary"]


def test_no_commit_push_or_merge_happens_in_any_path(repo, tmp_path):
    before = (_git(repo, "rev-list", "--all", "--count"), _git(repo, "branch", "--list"),
              _git(repo, "rev-parse", "HEAD"))
    _run(repo, tmp_path, _model({"files": [{"path": "calc.py", "content": RIGHT}]}))
    _run(repo, tmp_path, _model({"files": [{"path": "calc.py", "content": WRONG}]}))
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
