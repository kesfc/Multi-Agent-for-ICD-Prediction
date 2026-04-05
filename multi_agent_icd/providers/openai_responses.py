from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


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
        self.api_key = resolved_api_key
        self.timeout_seconds = timeout_seconds
        self.url = "https://api.openai.com/v1/responses"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }

        if metadata:
            payload["metadata"] = {
                key: str(value)
                for key, value in metadata.items()
                if value is not None
            }

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API request failed with status {exc.code}: {error_body}"
            ) from exc

        parsed = json.loads(body)
        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))

        output_text = self._extract_output_text(parsed)
        if not output_text:
            raise RuntimeError("OpenAI API returned no output text.")

        return json.loads(output_text)

    @staticmethod
    def _extract_output_text(response_body: dict[str, Any]) -> str:
        if isinstance(response_body.get("output_text"), str) and response_body["output_text"].strip():
            return response_body["output_text"]

        texts: list[str] = []
        for item in response_body.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                content_type = content.get("type")
                if content_type == "refusal":
                    raise RuntimeError(content.get("refusal", "Model refused the request."))
                if content_type in {"output_text", "text"} and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        return "\n".join(texts).strip()
