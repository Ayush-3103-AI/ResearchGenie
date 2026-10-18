import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.utilities import llm_provider  # noqa: E402

_REAL_COMPLETE_TOOLS = llm_provider.complete_tools


@pytest.fixture
def real_complete_tools():
    """The unpatched tool-calling completion, for the tests of it."""
    return _REAL_COMPLETE_TOOLS


@pytest.fixture(autouse=True)
def _no_tool_calls(monkeypatch):
    """Discovery asks the model which sources to search (#16). Unless a test
    fakes that call itself, behave like a provider without tool support, so
    no test reaches a real Ollama and the deterministic fallback applies."""
    async def no_tools(**kwargs):
        return None
    monkeypatch.setattr(llm_provider, "complete_tools", no_tools)


@pytest.fixture(autouse=True)
def _temporary_research_library(tmp_path, tmp_path_factory, monkeypatch):
    """Every run_pipeline() writes to the research library; point it at a
    throwaway file so no test ever touches the real ~/.researchgenie one."""
    monkeypatch.setenv("RESEARCHGENIE_LIBRARY", str(tmp_path / "library.sqlite"))
    # Same for the scholarly response cache (#6): no test reads another's
    # answers. Its own folder, so a test that checks tmp_path stays empty can.
    monkeypatch.setenv("RESEARCHGENIE_CACHE", str(tmp_path_factory.mktemp("cache") / "cache.sqlite"))
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE", raising=False)
