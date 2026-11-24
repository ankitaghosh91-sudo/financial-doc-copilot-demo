"""Provider-independent LLM client for evaluation and RAG tasks."""

import logging
import os
import httpx
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from src.config import AppConfig

logger = logging.getLogger(__name__)


class RateLimitExhaustedError(RuntimeError):
    """Raised when rate limits are hit across primary and fallback LLM providers."""


def is_rate_limit_error(exc: Exception) -> bool:
    exc_type = type(exc).__name__.lower()
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or "ratelimit" in exc_type:
        return True
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg


def _should_fallback(exc: Exception) -> bool:
    exc_type = type(exc).__name__.lower()
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)

    if status_code in (400, 408, 429, 500, 502, 503, 504):
        return True

    if (
        "badrequesterror" in exc_type
        or "apistatuserror" in exc_type
        or "apiconnectionerror" in exc_type
        or "ratelimit" in exc_type
    ):
        return True

    return any(
        token in msg
        for token in (
            "error code: 400",
            "status code: 400",
            "maximum context length",
            "input_tokens",
            "context_length_exceeded",
            "context length",
            "too many tokens",
            "429",
            "rate limit",
            "resource_exhausted",
            "timeout",
            "timed out",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _trim_messages_for_fallback(messages: list[dict], max_chars: int = 12000) -> list[dict]:
    """Keep system message intact and trim user/assistant content if very long."""
    if not messages:
        return messages

    trimmed: list[dict] = []
    remaining = max_chars

    for msg in messages:
        if msg.get("role") == "system":
            trimmed.append(dict(msg))

    for msg in reversed(messages):
        if msg.get("role") == "system":
            continue
        content = str(msg.get("content", ""))
        if len(content) <= remaining:
            trimmed.append({**msg, "content": content})
            remaining -= len(content)
        elif remaining > 200:
            trimmed.append({**msg, "content": content[:remaining]})
            remaining = 0

    system_msgs = [m for m in trimmed if m.get("role") == "system"]
    non_system_msgs = [m for m in trimmed if m.get("role") != "system"]
    non_system_msgs.reverse()
    return system_msgs + non_system_msgs


def _split_gemini_messages(messages: list[dict]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    convo_parts: list[str] = []

    for msg in messages:
        role = str(msg.get("role", "user")).lower().strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        else:
            text = str(content).strip()

        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            convo_parts.append(f"Assistant: {text}")
        else:
            convo_parts.append(f"User: {text}")

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(convo_parts)
    return system_instruction, prompt


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        enable_fallback: bool = True,
    ):
        AppConfig.from_env()

        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()

        if self.provider == "groq":
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is missing.")

            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            self.client = Groq(
                api_key=api_key,
                timeout=30.0,
                max_retries=1,
            )

        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL")
            ca_cert = os.getenv("QWEN_CA_CERT")

            if not api_key or not base_url or not ca_cert:
                raise ValueError(
                    "Qwen API configuration is incomplete. "
                    "Check QWEN_API_KEY, QWEN_BASE_URL, and QWEN_CA_CERT."
                )

            if not os.path.isfile(ca_cert):
                raise FileNotFoundError(f"Qwen CA certificate not found: {ca_cert}")

            self.model = model or os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")
            http_client = httpx.Client(
                verify=ca_cert,
                timeout=30.0,
            )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                max_retries=1,
            )

        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is missing.")

            self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            self.client = genai.Client(api_key=api_key)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self.fallback_clients: list[LLMClient] = []
        if enable_fallback:
            self.fallback_clients = self._build_fallback_clients()

        logger.info(
            "LLM_CLIENT_READY | provider=%s | model=%s | fallbacks=%s",
            self.provider,
            self.model,
            [(fc.provider, fc.model) for fc in self.fallback_clients],
        )

    @property
    def fallback_client(self) -> "LLMClient | None":
        return self.fallback_clients[0] if self.fallback_clients else None

    def _build_fallback_clients(self) -> list["LLMClient"]:
        specs = [
            (
                os.getenv("FALLBACK_LLM_PROVIDER", "").strip().lower(),
                os.getenv("FALLBACK_LLM_MODEL", "").strip() or None,
            ),
            (
                os.getenv("SECOND_FALLBACK_LLM_PROVIDER", "").strip().lower(),
                os.getenv("SECOND_FALLBACK_LLM_MODEL", "").strip() or None,
            ),
        ]

        clients: list[LLMClient] = []
        seen: set[tuple[str, str | None]] = {(self.provider, self.model)}

        for fb_provider, fb_model in specs:
            if not fb_provider and not fb_model:
                continue
            target_provider = fb_provider or self.provider
            if target_provider == "gemini" and fb_model is None:
                fb_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            elif target_provider == "groq" and fb_model is None:
                fb_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            elif target_provider == "qwen" and fb_model is None:
                fb_model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")

            key = (target_provider, fb_model)
            if key in seen:
                continue
            seen.add(key)

            try:
                clients.append(
                    LLMClient(
                        provider=target_provider,
                        model=fb_model,
                        enable_fallback=False,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "FALLBACK_LLM_INIT_FAILED | provider=%s | model=%s | error=%s",
                    target_provider,
                    fb_model,
                    exc,
                )

        return clients

    def _call_provider(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        *,
        _retry_on_length: bool = True,
    ) -> str:
        if self.provider == "gemini":
            system_instruction, prompt = _split_gemini_messages(messages)
            cfg_kwargs: dict = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction:
                cfg_kwargs["system_instruction"] = system_instruction
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                cfg_kwargs["response_mime_type"] = "application/json"

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**cfg_kwargs),
            )
            content = (getattr(response, "text", None) or "").strip()
            if not content:
                raise ValueError("Gemini returned an empty response.")
            return content

        request_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            request_args["response_format"] = response_format

        response = self.client.chat.completions.create(**request_args)

        choice = response.choices[0]
        content = choice.message.content

        if not content:
            if choice.finish_reason == "length" and _retry_on_length:
                bumped_tokens = min(max(max_tokens * 2, 1500), 3000)
                logger.warning(
                    "LLM_EMPTY_ON_LENGTH | provider=%s | model=%s | retrying max_tokens=%d",
                    self.provider,
                    self.model,
                    bumped_tokens,
                )
                return self._call_provider(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=bumped_tokens,
                    response_format=response_format,
                    _retry_on_length=False,
                )

            raise ValueError(
                f"LLM returned an empty response "
                f"(finish_reason={choice.finish_reason!r})"
            )

        return content.strip()

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: dict | None = None,
    ) -> str:
        """Generate a text response, falling back through configured secondary LLMs."""
        errors: list[tuple[str, str, Exception]] = []

        try:
            return self._call_provider(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            errors.append((self.provider, self.model, exc))
            body = getattr(exc, "body", None)
            if not self.fallback_clients or not _should_fallback(exc):
                logger.error(
                    "LLM_PRIMARY_FAILED_NO_FALLBACK | provider=%s | model=%s | has_fallback=%s | body=%r",
                    self.provider,
                    self.model,
                    bool(self.fallback_clients),
                    body,
                )
                if is_rate_limit_error(exc):
                    raise RateLimitExhaustedError(str(exc)) from exc
                raise

        trimmed_messages = _trim_messages_for_fallback(messages)
        fallback_max_tokens = min(max(max_tokens, 800), 1200)

        for fb_client in self.fallback_clients:
            logger.warning(
                "LLM_FALLBACK_TRIGGERED | primary_provider=%s | primary_model=%s | "
                "fallback_provider=%s | fallback_model=%s | last_error=%s",
                self.provider,
                self.model,
                fb_client.provider,
                fb_client.model,
                errors[-1][2],
            )
            try:
                return fb_client._call_provider(
                    messages=trimmed_messages,
                    temperature=temperature,
                    max_tokens=fallback_max_tokens,
                    response_format=response_format,
                )
            except Exception as fb_exc:
                errors.append((fb_client.provider, fb_client.model, fb_exc))
                logger.error(
                    "LLM_FALLBACK_FAILED | fallback_provider=%s | fallback_model=%s | body=%r | error=%s",
                    fb_client.provider,
                    fb_client.model,
                    getattr(fb_exc, "body", None),
                    fb_exc,
                )

        if all(is_rate_limit_error(err) for _, _, err in errors):
            raise RateLimitExhaustedError(
                f"Rate limit exhausted across providers: {[(p, m) for p, m, _ in errors]}"
            ) from errors[-1][2]

        raise errors[-1][2]