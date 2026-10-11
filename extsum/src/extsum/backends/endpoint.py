"""OpenAI-compatible endpoint backend (OpenRouter for the large tier).

Configured through OPENAI_BASE_URL and OPENAI_API_KEY.
"""
from __future__ import annotations

from typing import List, Optional

from .base import GenConfig


class EndpointBackend:
    """OpenAI-compatible remote chat backend."""

    wants_full_batch = False

    def __init__(
        self,
        model_id: str,
        gen: GenConfig,
        disable_reasoning: bool = True,
    ):
        self.model_id = model_id
        self.gen = gen
        self.disable_reasoning = disable_reasoning
        self._client = None

    def _ensure(self) -> None:
        """Create the OpenAI-compatible client."""
        if self._client is not None:
            return

        import os

        from openai import OpenAI

        base_url = os.environ.get(
            "OPENAI_BASE_URL"
        )
        api_key = os.environ.get(
            "OPENAI_API_KEY"
        )

        if not base_url:
            raise EnvironmentError(
                "OPENAI_BASE_URL is required for endpoint models."
            )

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for endpoint models."
            )

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(
                os.environ.get(
                    "OPENROUTER_TIMEOUT",
                    "120",
                )
            ),
            max_retries=2,
        )

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one remote response per prompt."""
        if not prompts:
            return []

        self._ensure()

        outputs: List[str] = []

        for prompt in prompts:
            messages = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            extra_body = (
                {"reasoning": {"enabled": False}}
                if self.disable_reasoning
                else None
            )

            response = (
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=int(
                        self.gen.max_new_tokens
                    ),
                    temperature=float(
                        self.gen.temperature
                    ),
                    seed=int(self.gen.seed),
                    extra_body=extra_body,
                )
            )

            outputs.append(
                response.choices[0].message.content
                or ""
            )

        return outputs

    def free_memory(self) -> None:
        """Remote endpoints have no local model memory to release."""
        return None
