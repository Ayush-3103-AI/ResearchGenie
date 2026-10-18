"""Tests for the local ResearchGenie config store."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie import config as rg_config  # noqa: E402


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rg_config, "CONFIG_DIR", tmp_path / ".researchgenie")
    monkeypatch.setattr(rg_config, "CONFIG_PATH", tmp_path / ".researchgenie" / "config.json")
    return rg_config


def test_load_returns_defaults_when_no_config_file_exists(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    config = mod.load()
    assert config.setup_complete is False
    assert config.provider == "ollama"


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    original = mod.Config(setup_complete=True, provider="gemini", gemini_api_key="secret-key-123")
    mod.save(original)
    loaded = mod.load()
    assert loaded.setup_complete is True
    assert loaded.provider == "gemini"
    assert loaded.gemini_api_key == "secret-key-123"


def test_load_ignores_unknown_fields_and_corrupt_json(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    mod.CONFIG_DIR.mkdir(parents=True)
    mod.CONFIG_PATH.write_text("not valid json{{{", encoding="utf-8")
    config = mod.load()
    assert config == mod.Config()  # falls back to defaults, does not crash


def test_masked_summary_never_exposes_the_full_api_key(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    config = mod.Config(gemini_api_key="AIzaSyD-verysecretvalue1234567890")
    summary = config.masked_summary()
    assert "verysecretvalue1234567890" not in summary["gemini_api_key"]
    assert summary["gemini_api_key"].startswith("AIza")


def test_apply_to_environment_sets_expected_variables(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = mod.Config(provider="gemini", gemini_api_key="k123", gemini_model="gemini-2.5-flash")
    mod.apply_to_environment(config)
    import os
    assert os.environ["AI_PROVIDER"] == "gemini"
    assert os.environ["GEMINI_API_KEY"] == "k123"
    assert os.environ["GEMINI_MODEL"] == "gemini-2.5-flash"


def test_is_setup_complete_reflects_saved_state(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    assert mod.is_setup_complete() is False
    mod.save(mod.Config(setup_complete=True))
    assert mod.is_setup_complete() is True


SCHOLARLY_ENVS = ("OPENALEX_API_KEY", "CONTACT_EMAIL", "NCBI_API_KEY")


def test_scholarly_identity_round_trips_and_is_masked(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    mod.save(mod.Config(openalex_api_key="oa-0123456789", contact_email="me@uni.edu",
                        ncbi_api_key="ncbi-0123456789"))
    loaded = mod.load()
    assert (loaded.openalex_api_key, loaded.contact_email, loaded.ncbi_api_key) == (
        "oa-0123456789", "me@uni.edu", "ncbi-0123456789")
    summary = str(loaded.masked_summary())
    for secret in ("oa-0123456789", "me@uni.edu", "ncbi-0123456789"):
        assert secret not in summary


def test_a_config_saved_before_the_scholarly_fields_still_loads(tmp_path, monkeypatch):
    mod = _isolated(tmp_path, monkeypatch)
    mod.CONFIG_DIR.mkdir(parents=True)
    mod.CONFIG_PATH.write_text('{"setup_complete": true, "provider": "ollama"}', encoding="utf-8")
    config = mod.load()
    assert config.setup_complete and config.openalex_api_key is None


def test_environment_variables_override_the_saved_scholarly_values(tmp_path, monkeypatch):
    import os
    mod = _isolated(tmp_path, monkeypatch)
    for name in SCHOLARLY_ENVS:
        monkeypatch.setenv(name, "")  # recorded, so teardown removes what apply_scholarly sets
        monkeypatch.delenv(name)
    monkeypatch.setenv("CONTACT_EMAIL", "from-env@uni.edu")
    mod.apply_to_environment(mod.Config(contact_email="saved@uni.edu", ncbi_api_key="saved-ncbi"))
    assert os.environ["CONTACT_EMAIL"] == "from-env@uni.edu"
    assert os.environ["NCBI_API_KEY"] == "saved-ncbi"
    assert "OPENALEX_API_KEY" not in os.environ
