"""Hugging Face Transformers backend (one GPU, no continuous batching)."""
from __future__ import annotations

from typing import List, Optional

from .base import GenConfig, set_random_seed


class LocalHFBackend:
    """Hugging Face Transformers generation backend."""

    wants_full_batch = False

    def __init__(
        self,
        hf_id: str,
        gen: GenConfig,
        device_map=None,
        load_in_4bit: bool = False,
    ):
        self.hf_id = hf_id
        self.gen = gen

        # Default to a single GPU. ``device_map="auto"`` may be configured in
        # models.yaml for models that require sharding.
        self.device_map = (
            device_map
            if device_map is not None
            else {"": 0}
        )

        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tok = None

        # The first generation call uses gen.seed, the next uses gen.seed + 1,
        # and so forth. This produces distinct but reproducible sampled calls.
        self._generation_call_index = 0

    def _ensure(self) -> None:
        """Load the tokenizer and model once."""
        if self._model is not None:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        # Seed model initialization and any library-side random operations.
        set_random_seed(self.gen.seed)

        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16

        # Prevent device_map="auto" from spilling model layers to CPU or disk.
        if self.device_map == "auto":
            gpu_count = torch.cuda.device_count()

            if gpu_count <= 0:
                raise RuntimeError(
                    "device_map='auto' was requested, but no CUDA GPU "
                    "is available."
                )

            model_kwargs["max_memory"] = {
                gpu_index: "23GiB"
                for gpu_index in range(gpu_count)
            }

        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            **model_kwargs,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.hf_id!r} has neither a pad token "
                    "nor an EOS token."
                )

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        self._model = model.eval()
        self._tok = tokenizer

    def _next_generation_seed(self) -> int:
        """Return the next deterministic local-generation seed."""
        seed = (
            int(self.gen.seed)
            + self._generation_call_index
        )

        self._generation_call_index += 1
        return seed

    def reset_generation_seed(self) -> None:
        """Reset local generation to the configured base seed.

        This is useful for tests or when deliberately restarting the complete
        generation sequence. Do not call it between self-consistency samples.
        """
        self._generation_call_index = 0

    def _input_device(self):
        """Return a device suitable for tokenizer output tensors."""
        import torch

        model = self._model

        # For ordinary non-sharded models, model.device is sufficient.
        try:
            device = model.device

            if str(device) != "meta":
                return device
        except (AttributeError, RuntimeError):
            pass

        # Accelerate-dispatched models expose a Hugging Face device map.
        hf_device_map = getattr(model, "hf_device_map", None)

        if isinstance(hf_device_map, dict):
            for mapped_device in hf_device_map.values():
                if mapped_device in {"cpu", "disk"}:
                    continue

                if isinstance(mapped_device, int):
                    return torch.device(
                        f"cuda:{mapped_device}"
                    )

                mapped_device = str(mapped_device)

                if mapped_device.startswith("cuda"):
                    return torch.device(mapped_device)

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    def _gen_chats(
        self,
        chats: List[str],
    ) -> List[str]:
        """Generate outputs for already rendered chat-template strings."""
        import torch

        if not chats:
            return []

        tokenizer = self._tok
        model = self._model

        if tokenizer is None or model is None:
            raise RuntimeError(
                "Local Hugging Face backend was not initialized."
            )

        generation_seed = self._next_generation_seed()
        set_random_seed(generation_seed)

        encoded = tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
        )

        input_device = self._input_device()

        encoded = {
            key: tensor.to(input_device)
            for key, tensor in encoded.items()
        }

        prompt_lengths = encoded[
            "attention_mask"
        ].sum(dim=1)

        do_sample = float(self.gen.temperature) > 0.0

        generation_kwargs = {
            "max_new_tokens": int(
                self.gen.max_new_tokens
            ),
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Temperature should not be passed for deterministic greedy decoding.
        if do_sample:
            generation_kwargs["temperature"] = float(
                self.gen.temperature
            )

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
            )

        texts = [
            tokenizer.decode(
                output_row[int(prompt_length):],
                skip_special_tokens=True,
            )
            for output_row, prompt_length in zip(
                generated,
                prompt_lengths,
            )
        ]

        del encoded
        del generated

        return texts

    def generate_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
    ) -> List[str]:
        """Generate one output for each prompt."""
        import torch

        if not prompts:
            return []

        self._ensure()

        tokenizer = self._tok

        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer was not initialized."
            )

        template_kwargs = {}

        # Disable Qwen thinking mode because the benchmark requires a direct
        # JSON response rather than a long internal reasoning trace.
        if "qwen" in self.hf_id.lower():
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

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

            chats.append(rendered)

        # Try the complete batch first. If the batch is too large, retry each
        # prompt individually while clearing unused CUDA cache between calls.
        try:
            return self._gen_chats(chats)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            texts: List[str] = []

            for chat in chats:
                texts.extend(
                    self._gen_chats([chat])
                )

                torch.cuda.empty_cache()

            return texts

    def free_memory(self) -> None:
        """Release unused CUDA cache without unloading the model."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
