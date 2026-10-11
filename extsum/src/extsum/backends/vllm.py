"""vLLM backend: continuous batching, the fast path on a rented GPU."""
from __future__ import annotations

from typing import List, Optional

from .base import GenConfig


class VLLMBackend:
    """Fast local inference using vLLM continuous batching."""

    # The runner may send all prompts for a variant in one call.
    wants_full_batch = True

    def __init__(
        self,
        hf_id: str,
        gen: GenConfig,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        disable_reasoning: bool = True,
    ):
        self.hf_id = hf_id
        self.gen = gen
        self.quantization = quantization
        self.tp = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.disable_reasoning = disable_reasoning

        self._llm = None
        self._tok = None
        self._sp = None

    def _ensure(self) -> None:
        """Initialize vLLM and its sampling parameters."""
        if self._llm is not None:
            return

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tok = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
        )

        kwargs = {
            "model": self.hf_id,
            "trust_remote_code": True,
            "tensor_parallel_size": int(self.tp),
            "gpu_memory_utilization": float(
                self.gpu_mem
            ),
            "dtype": "bfloat16",
        }

        if self.max_model_len:
            kwargs["max_model_len"] = int(
                self.max_model_len
            )

        if self.quantization:
            kwargs["quantization"] = self.quantization

            if self.quantization == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"

        self._llm = LLM(**kwargs)

        self._sp = SamplingParams(
            temperature=float(self.gen.temperature),
            max_tokens=int(self.gen.max_new_tokens),
            seed=(
                int(self.gen.seed)
                if self.gen.temperature > 0
                else None
            ),
        )

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one vLLM output per prompt."""
        if not prompts:
            return []

        self._ensure()

        tokenizer = self._tok

        template_kwargs = {}

        if (
            "qwen" in self.hf_id.lower()
            and self.disable_reasoning
        ):
            template_kwargs["enable_thinking"] = False

        chats: List[str] = []

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

            chats.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )

        outputs = self._llm.generate(
            chats,
            self._sp,
            use_tqdm=False,
        )

        return [
            output.outputs[0].text
            for output in outputs
        ]

    def free_memory(self) -> None:
        """vLLM manages its own model and KV-cache memory."""
        return None
