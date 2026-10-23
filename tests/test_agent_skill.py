"""The agent skill (#21) follows the Agent Skills spec, and every command
line in it is one the headless CLI accepts, so the skill can't drift."""

import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from researchgenie import headless  # noqa: E402

SKILL_DIR = ROOT / "skills" / "researchgenie"
TEXT = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
PLACEHOLDERS = {"<question>": "What is X?", "<run-id>": "rg-20261023-1530"}


def test_frontmatter_follows_the_agent_skills_spec():
    _, front, body = TEXT.split("---\n", 2)
    meta = yaml.safe_load(front)
    assert set(meta) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert meta["name"] == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert 0 < len(meta["compatibility"]) <= 500
    assert meta["metadata"]["openclaw"]["requires"]["bins"] == ["researchgenie"]
    assert body.strip()


def test_every_command_line_parses_with_the_headless_cli():
    lines = [line for block in re.findall(r"```\n(.*?)```", TEXT, re.S) for line in block.splitlines()]
    assert len(lines) >= 7
    for line in lines:
        for placeholder, value in PLACEHOLDERS.items():
            line = line.replace(placeholder, value)
        program, *argv = shlex.split(line)
        assert program == "researchgenie", line
        assert headless.is_headless(argv), line
        headless.build_parser().parse_args(argv)  # SystemExit fails the test
