import pytest


@pytest.fixture(autouse=True)
def _temporary_research_library(tmp_path, tmp_path_factory, monkeypatch):
    """Every run_pipeline() writes to the research library; point it at a
    throwaway file so no test ever touches the real ~/.researchgenie one."""
    monkeypatch.setenv("RESEARCHGENIE_LIBRARY", str(tmp_path / "library.sqlite"))
    # Same for the scholarly response cache (#6): no test reads another's
    # answers. Its own folder, so a test that checks tmp_path stays empty can.
    monkeypatch.setenv("RESEARCHGENIE_CACHE", str(tmp_path_factory.mktemp("cache") / "cache.sqlite"))
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE", raising=False)
