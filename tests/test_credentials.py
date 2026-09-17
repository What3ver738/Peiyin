"""Credential resolution, storage permissions and model settings."""

import os
import stat

import pytest

from peiyin import config


def test_an_environment_variable_beats_a_stored_key(monkeypatch):
    """So a shared machine or a CI run never has to write a key to disk."""
    config.save_config(fish_key="stored-value")
    assert config.api_key("fish") == "stored-value"
    monkeypatch.setenv("FISH_API_KEY", "env-value")
    assert config.api_key("fish") == "env-value"
    assert config.key_source("fish") == "environment"


def test_an_unset_key_is_empty_not_an_error(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    config.save_config(fish_key=None)
    assert config.api_key("fish") == ""
    assert config.key_source("fish") == "not set"


def test_one_box_takes_either_an_openai_or_a_gemini_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.save_config(llm_key="sk-abc123")
    assert config.api_key("openai") == "sk-abc123"
    assert config.api_key("gemini") == ""
    assert config.llm_key() == "sk-abc123"

    config.save_config(llm_key="AIzaSomeGeminiKey")
    assert config.api_key("openai") == ""
    assert config.api_key("gemini") == "AIzaSomeGeminiKey"
    assert config.llm_key() == "AIzaSomeGeminiKey"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits do not describe Windows ACLs")
def test_the_config_file_is_not_world_readable():
    """It holds API keys."""
    config.save_config(fish_key="secret")
    mode = stat.S_IMODE(config.CONFIG_PATH.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_dotenv_is_read_without_clobbering_a_real_env_var(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        'FISH_API_KEY="from-dotenv"\n'
        "OPENAI_API_KEY=sk-from-dotenv\n"
        "MALFORMED LINE\n", encoding="utf-8")
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    found = config.load_dotenv(env)
    assert found == {"FISH_API_KEY": "from-dotenv",
                     "OPENAI_API_KEY": "sk-from-dotenv"}
    assert os.environ["FISH_API_KEY"] == "from-dotenv"
    assert os.environ["OPENAI_API_KEY"] == "already-set"   # not clobbered


def test_a_missing_dotenv_is_fine():
    assert config.load_dotenv("/nonexistent/.env") == {}


def test_models_fall_back_to_the_defaults():
    config.save_config(fish_model=None, llm_model=None)
    assert config.fish_model() == config.FISH_TTS_MODEL
    assert config.llm_model() == ""          # "" means: pick the best available


def test_models_can_be_overridden():
    config.save_config(fish_model="s1", llm_model="gpt-4.1")
    assert config.fish_model() == "s1"
    assert config.llm_model() == "gpt-4.1"
    config.save_config(fish_model=None, llm_model=None)


def test_a_named_llm_model_is_used_without_probing_the_account():
    """Probing costs a network round trip; a named model must skip it. The
    suite blocks the network, so this would fail loudly if it probed."""
    from peiyin.llm import pick_openai_model
    config.save_config(llm_model="gpt-4.1")
    assert pick_openai_model("sk-whatever") == "gpt-4.1"
    config.save_config(llm_model=None)
