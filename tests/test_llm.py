"""Typed LLM gateway tests without any external API calls."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tracescholar.config import Settings
from tracescholar.llm import LLMConfigurationError, LLMError, OpenAICompatibleLLM
from tracescholar.planning.schemas import ResearchPlanDraft


def _draft() -> dict:
    return {
        "normalized_question": "How does query rewriting affect RAG?",
        "sub_questions": ["Which outcomes change?"],
        "concepts": [{"term": "RAG", "synonyms": [], "abbreviations": []}],
        "exclusion_terms": [],
        "inclusion_criteria": ["RAG experiments"],
        "exclusion_criteria": ["Unrelated systems"],
        "constraints": [],
        "ambiguity_items": [],
        "search_tracks": [{"label": "Null results", "intent": "counter_evidence",
                           "query": "RAG query rewriting no improvement", "rationale": "Seek failures"}],
        "stop_conditions": ["No new evidence"],
    }


def _response(content: str | None, *, finish_reason: str = "stop", refusal: str | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal),
    )])


class OpenAICompatibleLLMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            _env_file=None,
            llm_base_url="https://compatible.example/v1",
            llm_api_key="test-secret",
            llm_model="third-party-model",
        )
        self.client = MagicMock()

    def test_strict_schema_request_returns_validated_model(self) -> None:
        import json

        self.client.chat.completions.create.return_value = _response(
            json.dumps(_draft(), ensure_ascii=False)
        )
        llm = OpenAICompatibleLLM(settings=self.settings, client=self.client)
        output = llm.generate(ResearchPlanDraft, system_prompt="Plan scope.", user_prompt="Question")
        self.assertIsInstance(output, ResearchPlanDraft)
        self.assertEqual(llm.model_name, "third-party-model")
        kwargs = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "third-party-model")
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        schema = kwargs["response_format"]["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("search_tracks", schema["required"])
        self.assertEqual(kwargs["messages"][1]["content"], "Question")

    def test_json_object_mode_still_validates_locally(self) -> None:
        self.client.chat.completions.create.return_value = _response("{}")
        settings = Settings(
            _env_file=None,
            llm_base_url="https://compatible.example/v1",
            llm_api_key="test-secret",
            llm_model="third-party-model",
            llm_structured_mode="json_object",
        )
        llm = OpenAICompatibleLLM(settings=settings, client=self.client)
        with self.assertRaisesRegex(LLMError, "structured validation"):
            llm.generate(ResearchPlanDraft, system_prompt="Plan", user_prompt="Question")
        self.assertEqual(
            self.client.chat.completions.create.call_args.kwargs["response_format"],
            {"type": "json_object"},
        )

    def test_non_json_output_is_rejected(self) -> None:
        self.client.chat.completions.create.return_value = _response("not JSON")
        llm = OpenAICompatibleLLM(settings=self.settings, client=self.client)
        with self.assertRaisesRegex(LLMError, "structured validation"):
            llm.generate(ResearchPlanDraft, system_prompt="Plan", user_prompt="Question")

    def test_incomplete_refused_and_empty_responses_are_rejected(self) -> None:
        llm = OpenAICompatibleLLM(settings=self.settings, client=self.client)
        cases = (
            (_response("{}", finish_reason="length"), "did not finish"),
            (_response(None, refusal="no"), "refused"),
            (_response(None), "no structured content"),
            (SimpleNamespace(choices=[]), "no choices"),
        )
        for response, message in cases:
            with self.subTest(message=message):
                self.client.chat.completions.create.return_value = response
                with self.assertRaisesRegex(LLMError, message):
                    llm.generate(ResearchPlanDraft, system_prompt="Plan", user_prompt="Question")

    def test_missing_key_is_reported_without_network(self) -> None:
        llm = OpenAICompatibleLLM(settings=Settings(
            _env_file=None, llm_base_url="https://compatible.example/v1", llm_api_key=None,
            llm_model="third-party-model",
        ))
        with self.assertRaisesRegex(LLMConfigurationError, "TRACESCHOLAR_LLM_API_KEY"):
            llm.generate(ResearchPlanDraft, system_prompt="Plan", user_prompt="Question")
