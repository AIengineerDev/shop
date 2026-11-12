"""Settings from env and .env (spec §8)."""

from pathlib import Path

import pytest

from catalog_onboard.config import Settings, parse_dotenv

ENV_KEYS = ["LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT", "LLM_MAX_RETRIES", "LLM_MAX_TOKENS",
            "HIGH_CONF_THRESHOLD", "DISAGREE_QUARANTINE_BELOW", "CANDIDATE_K", "RULES_DIR",
            "ANTHROPIC_API_KEY", "ENV_FILE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults_follow_spec(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    s = Settings.from_env()
    assert s.llm_provider == "fake"  # no key, no explicit provider
    assert s.llm_model == "claude-sonnet-5-5"  # spec §8: latest Sonnet-class
    assert s.llm_effort == "low" and s.llm_max_retries == 2 and s.llm_max_tokens == 4096


def test_parse_dotenv():
    text = """
# comment
export LLM_MODEL=claude-sonnet-5-5
LLM_MAX_RETRIES = 4
QUOTED="a b # not a comment"
SINGLE='x'
TRAILING=value # comment
BROKEN_LINE
=novalue
"""
    assert parse_dotenv(text) == {
        "LLM_MODEL": "claude-sonnet-5-5", "LLM_MAX_RETRIES": "4", "QUOTED": "a b # not a comment",
        "SINGLE": "x", "TRAILING": "value",
    }


def test_dotenv_is_loaded_and_real_env_wins(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=fake\nLLM_MAX_RETRIES=5\nHIGH_CONF_THRESHOLD=0.9\nLLM_EFFORT=medium\n")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")  # already set: must not be overridden
    s = Settings.from_env(env_file=env)
    assert s.llm_max_retries == 1 and s.high_conf_threshold == 0.9 and s.llm_effort == "medium"


def test_env_file_variable_and_missing_file(tmp_path: Path, monkeypatch):
    env = tmp_path / "custom.env"
    env.write_text("CANDIDATE_K=3\n")
    monkeypatch.setenv("ENV_FILE", str(env))
    assert Settings.from_env().candidate_k == 3
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("CANDIDATE_K")
    assert Settings.from_env().candidate_k == 8


def test_provider_and_effort_validation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings.from_env()
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_EFFORT", "turbo")
    with pytest.raises(ValueError, match="LLM_EFFORT"):
        Settings.from_env()


def test_api_key_selects_anthropic_unless_provider_set(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings.from_env().llm_provider == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert Settings.from_env().llm_provider == "fake"
