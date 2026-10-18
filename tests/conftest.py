import pytest


@pytest.fixture(autouse=True)
def _temporary_research_library(tmp_path, monkeypatch):
    """Every run_pipeline() writes to the research library; point it at a
    throwaway file so no test ever touches the real ~/.researchgenie one."""
    monkeypatch.setenv("RESEARCHGENIE_LIBRARY", str(tmp_path / "library.sqlite"))
