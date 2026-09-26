"""Tests for centralized TraceScholar configuration."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracescholar.config import Settings, get_settings


class SettingsTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_defaults_are_available_without_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.app_env, "development")
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.data_dir, Path("data"))
        self.assertIsNone(settings.database_url)
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.llm_base_url, "https://api.openai.com/v1")
        self.assertIsNone(settings.llm_api_key)
        self.assertEqual(settings.llm_structured_mode, "json_schema")
        self.assertEqual(settings.fulltext_screening_llm_timeout_seconds, 180)
        self.assertEqual(settings.openalex_base_url, "https://api.openalex.org")
        self.assertEqual(settings.crossref_base_url, "https://api.crossref.org")
        self.assertIsNone(settings.crossref_email)

    def test_environment_variables_override_defaults(self) -> None:
        environment = {
            "TRACESCHOLAR_APP_ENV": "test",
            "TRACESCHOLAR_LOG_LEVEL": "DEBUG",
            "TRACESCHOLAR_DATABASE_URL": "postgresql://user:password@localhost/db",
            "TRACESCHOLAR_OPENAI_API_KEY": "test-secret",
            "TRACESCHOLAR_OPENALEX_API_KEY": "openalex-test-secret",
            "TRACESCHOLAR_CROSSREF_EMAIL": "researcher@example.org",
            "TRACESCHOLAR_LLM_BASE_URL": "https://compatible.example/v1",
            "TRACESCHOLAR_LLM_API_KEY": "llm-test-secret",
            "TRACESCHOLAR_LLM_MODEL": "test-model",
            "TRACESCHOLAR_LLM_STRUCTURED_MODE": "json_object",
            "TRACESCHOLAR_FULLTEXT_SCREENING_LLM_TIMEOUT_SECONDS": "240",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.app_env, "test")
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(
            settings.database_url.get_secret_value(),
            "postgresql://user:password@localhost/db",
        )
        self.assertEqual(settings.openai_api_key.get_secret_value(), "test-secret")
        self.assertEqual(settings.openalex_api_key.get_secret_value(), "openalex-test-secret")
        self.assertEqual(settings.crossref_email, "researcher@example.org")
        self.assertEqual(settings.llm_base_url, "https://compatible.example/v1")
        self.assertEqual(settings.llm_api_key.get_secret_value(), "llm-test-secret")
        self.assertEqual(settings.llm_model, "test-model")
        self.assertEqual(settings.llm_structured_mode, "json_object")
        self.assertEqual(settings.fulltext_screening_llm_timeout_seconds, 240)
        self.assertNotIn("test-secret", repr(settings))
        self.assertNotIn("openalex-test-secret", repr(settings))
        self.assertNotIn("llm-test-secret", repr(settings))
        self.assertNotIn("password", repr(settings))

    def test_settings_can_be_loaded_from_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "TRACESCHOLAR_APP_ENV=test\n"
                "TRACESCHOLAR_DATA_DIR=fixtures\n"
                "TRACESCHOLAR_OPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = Settings(_env_file=env_file)

        self.assertEqual(settings.app_env, "test")
        self.assertEqual(settings.data_dir, Path("fixtures"))
        self.assertEqual(settings.openai_model, "test-model")

    def test_get_settings_returns_one_process_wide_instance(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            first = get_settings()
            second = get_settings()

        self.assertIs(first, second)
