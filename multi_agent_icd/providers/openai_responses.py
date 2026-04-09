from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


class OpenAIResponsesLLM:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for OpenAIResponsesLLM.")

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise EnvironmentError("OPENAI_API_KEY is required to call the OpenAI API.")

        self.model_name = model_name
        self.client = OpenAI(api_key=resolved_api_key, timeout=timeout_seconds)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        response_model: type | None = None,
    ) -> dict[str, Any]:
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
            metadata=metadata or None,
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            output_text = getattr(response, "output_text", None)
            raise RuntimeError(f"OpenAI API returned no parsed output. Raw output: {output_text}")

        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        raise TypeError("Parsed response was not a Pydantic model or dict.")
