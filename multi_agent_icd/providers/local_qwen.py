from __future__ import annotations

import inspect
import json
import os
import re
from threading import Lock
from typing import Any


DEFAULT_QWEN_MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
)
DEFAULT_QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "2048"))


def _strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = _strip_thinking_blocks(raw_text).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"Local Qwen model did not return valid JSON. Raw output: {cleaned[:1200]}")


def _response_schema_prompt(response_model: type | None) -> str:
    if response_model is None or not hasattr(response_model, "model_json_schema"):
        return ""
    schema = response_model.model_json_schema()
    return "\n\nReturn a JSON object that matches this schema:\n" + json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )


def _validate_response(parsed: dict[str, Any], response_model: type | None) -> dict[str, Any]:
    if response_model is None:
        return parsed
    if hasattr(response_model, "model_validate"):
        return response_model.model_validate(parsed).model_dump()
    return parsed


class LocalQwenLLM:
    _cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    _cache_lock = Lock()

    def __init__(
        self,
        model_name: str | None = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
        enable_thinking: bool = False,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        self.model_name = model_name or DEFAULT_QWEN_MODEL_NAME
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def _load_model(self) -> tuple[Any, Any]:
        cache_key = (self.model_name, self.torch_dtype, self.device_map)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for LocalQwenLLM. "
                    "Install the local-model dependencies from pyproject.toml first."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_kwargs: dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }
            try:
                import torch
            except ImportError:
                torch = None

            if self.device_map == "auto" and torch is not None and not torch.cuda.is_available():
                model_kwargs["device_map"] = None
            else:
                model_kwargs["device_map"] = self.device_map

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            self._cache[cache_key] = (tokenizer, model)
            return tokenizer, model

    def _build_chat_prompt(
        self,
        tokenizer: Any,
        system_prompt: str,
        user_prompt: str,
        response_model: type | None = None,
    ) -> str:
        full_user_prompt = user_prompt + _response_schema_prompt(response_model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt},
        ]

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return f"System:\n{system_prompt}\n\nUser:\n{full_user_prompt}\n\nAssistant:\n"

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            signature = inspect.signature(apply_chat_template)
            if "enable_thinking" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            ):
                template_kwargs["enable_thinking"] = self.enable_thinking
        except (TypeError, ValueError):
            template_kwargs["enable_thinking"] = self.enable_thinking

        try:
            return apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return apply_chat_template(messages, **template_kwargs)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
        response_model: type | None = None,
    ) -> dict[str, Any]:
        del metadata

        tokenizer, model = self._load_model()
        prompt = self._build_chat_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )

        model_inputs = tokenizer([prompt], return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is None:
            model_device = next(model.parameters()).device
        model_inputs = model_inputs.to(model_device)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        generated_ids = model.generate(**model_inputs, **generation_kwargs)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed = _extract_json_object(raw_output)
        return _validate_response(parsed, response_model=response_model)
