"""Tiny deterministic Qwen2 model factory used as an end-to-end DLI example."""

from __future__ import annotations

import torch


def _import_qwen2_classes() -> tuple[type, type]:
    """Imports the Qwen2 config and causal-LM classes lazily.

    Returns:
        A ``(Qwen2Config, Qwen2ForCausalLM)`` tuple.
    """
    # Some CPU-only dev environments have a broken torchvision registration path.
    # Qwen2 is text-only, so mark torchvision unavailable before importing transformers.
    try:
        import transformers.utils
        import transformers.utils.import_utils

        transformers.utils.is_torchvision_available = lambda: False
        transformers.utils.import_utils.is_torchvision_available = lambda: False
    except Exception:
        pass
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2Config, Qwen2ForCausalLM


def tiny_config() -> object:
    """Builds a minimal Qwen2 config (1 layer, hidden size 4, vocab 8).

    Returns:
        A ``Qwen2Config`` small enough to export and run instantly.
    """
    Qwen2Config, _ = _import_qwen2_classes()
    return Qwen2Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )


def _init_tiny_qwen2(model: torch.nn.Module) -> torch.nn.Module:
    """Fills every parameter with a deterministic pattern in [-6/13, 6/13].

    Args:
        model: The randomly initialized model to overwrite.

    Returns:
        The same model, switched to eval mode.
    """
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
            parameter.copy_(((values + index) % 13 - 6) / 13.0)
    return model.eval()


def create_model(model_id: str | None = None, *, local_files_only: bool = False) -> torch.nn.Module:
    """Creates a Qwen2 causal LM.

    Args:
        model_id: Hugging Face model id or local path. When omitted, a tiny
            deterministic model is built instead.
        local_files_only: If true, never hit the network when loading ``model_id``.

    Returns:
        The model in eval mode.
    """
    _, Qwen2ForCausalLM = _import_qwen2_classes()
    if model_id:
        return Qwen2ForCausalLM.from_pretrained(model_id, local_files_only=local_files_only).eval()
    return _init_tiny_qwen2(Qwen2ForCausalLM(tiny_config()))
