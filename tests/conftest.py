import pytest

from financial_rag.config import Settings


@pytest.fixture(autouse=True)
def _isolate_settings_from_local_env(monkeypatch):
    """Prevent developer .env files from altering deterministic unit test defaults."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
