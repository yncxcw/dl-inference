from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace

import torch

from dli_export.qwen3_5 import (
    export_qwen3_5_static_cache,
    qwen3_5_static_cache_to_dli,
)


try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    _HAS_QWEN3_5 = True
except (ImportError, AttributeError):
    Qwen3_5ForCausalLM = None
    Qwen3_5TextConfig = None
    _HAS_QWEN3_5 = False


def _tiny_model():
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        dtype="float32",
    )
    return Qwen3_5ForCausalLM(config).eval()


def _clone_tensors(values):
    return tuple(value.clone() for value in values)


def _assert_tensors_close(test: unittest.TestCase, actual, expected) -> None:
    test.assertEqual(len(actual), len(expected))
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(_HAS_QWEN3_5, "Transformers does not provide Qwen3.5")
class Qwen3_5ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = export_qwen3_5_static_cache(
            _tiny_model(),
            max_cache_len=8,
            strict=True,
        )

    def test_state_schema_is_deterministic_and_fp32(self) -> None:
        model = self.bundle.prefill.eager_module.model
        self.assertFalse(model.training)
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertFalse(parameter.requires_grad)
        self.assertEqual(
            self.bundle.state_names,
            (
                "layer.0.linear.conv",
                "layer.0.linear.recurrent",
                "layer.1.attention.key",
                "layer.1.attention.value",
                "cursor",
            ),
        )
        state = self.bundle.initial_state()
        self.assertEqual(state["layer.0.linear.conv"].shape, (1, 48, 4))
        self.assertEqual(state["layer.0.linear.recurrent"].shape, (1, 2, 8, 8))
        self.assertEqual(state["layer.1.attention.key"].shape, (1, 1, 8, 16))
        self.assertEqual(state["layer.1.attention.value"].shape, (1, 1, 8, 16))
        self.assertEqual(state["cursor"].shape, ())
        for name, tensor in state.items():
            expected_dtype = torch.int64 if name == "cursor" else torch.float32
            self.assertEqual(tensor.dtype, expected_dtype)
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)

        for stage in (self.bundle.prefill, self.bundle.decode):
            self.assertEqual(tuple(stage.state_inputs.values()), self.bundle.state_names)
            self.assertEqual(tuple(stage.state_outputs.values()), self.bundle.state_names)
            self.assertEqual(tuple(stage.state_initializers), self.bundle.state_names)
        self.assertIs(self.bundle.prefill.program, self.bundle.decode.program)
        self.assertIs(self.bundle.prefill.eager_module, self.bundle.decode.eager_module)

    def test_prefill_eager_export_parity_at_cursor_zero(self) -> None:
        input_ids = torch.tensor([5], dtype=torch.int64)
        eager_inputs = (input_ids, *self.bundle.flat_initial_state())
        exported_inputs = _clone_tensors(eager_inputs)
        eager = self.bundle.prefill.eager_module(*eager_inputs)
        exported = self.bundle.prefill.program.module()(*exported_inputs)
        _assert_tensors_close(self, exported, eager)
        self.assertEqual(exported[-1].item(), 1)

    def test_decode_eager_export_parity_across_tokens(self) -> None:
        first_token = torch.tensor([5], dtype=torch.int64)
        eager_prefill = self.bundle.prefill.eager_module(
            first_token, *self.bundle.flat_initial_state()
        )
        exported_prefill = self.bundle.prefill.program.module()(
            first_token, *self.bundle.flat_initial_state()
        )
        eager_state = tuple(eager_prefill[1:])
        exported_state = tuple(exported_prefill[1:])

        for expected_cursor, token_id in enumerate((7, 9, 11), start=2):
            input_ids = torch.tensor([token_id], dtype=torch.int64)
            eager_inputs = _clone_tensors(eager_state)
            exported_inputs = _clone_tensors(exported_state)
            eager = self.bundle.decode.eager_module(input_ids, *eager_inputs)
            exported = self.bundle.decode.program.module()(input_ids, *exported_inputs)
            _assert_tensors_close(self, exported, eager)
            self.assertEqual(exported[-1].item(), expected_cursor)
            for original, after in zip(eager_state, eager_inputs):
                torch.testing.assert_close(original, after)
            for original, after in zip(exported_state, exported_inputs):
                torch.testing.assert_close(original, after)
            eager_state = tuple(eager[1:])
            exported_state = tuple(exported[1:])

    def test_dli_conversion_uses_state_edges_and_shared_weights(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            graph_paths, weights_path = qwen3_5_static_cache_to_dli(
                self.bundle,
                output_dir,
                stem="tiny_qwen3_5",
                model_id="test/tiny-qwen3.5",
            )
            self.assertEqual(set(graph_paths), {"prefill", "decode"})
            prefill = json.loads(graph_paths["prefill"].read_text(encoding="utf-8"))
            decode = json.loads(graph_paths["decode"].read_text(encoding="utf-8"))
            self.assertEqual(prefill["weights"], weights_path.name)
            self.assertEqual(decode["weights"], weights_path.name)
            self.assertEqual(prefill["inputs"], ["input_ids"])
            self.assertEqual(decode["inputs"], ["input_ids"])
            self.assertEqual(prefill["outputs"], ["logits"])
            self.assertEqual(decode["outputs"], ["logits"])
            self.assertEqual(graph_paths["prefill"], graph_paths["decode"])
            self.assertEqual(prefill, decode)
            for graph in (prefill, decode):
                self.assertEqual(len(graph["state_inputs"]), len(self.bundle.state_specs))
                self.assertEqual(len(graph["state_outputs"]), len(self.bundle.state_specs))
                self.assertEqual(
                    len(graph["state_initializers"]), len(self.bundle.state_specs)
                )
            bundle_path = graph_paths["prefill"].parent / "tiny_qwen3_5.dli.bundle.json"
            manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest,
                {
                    "format": "dli.autoregressive.v1",
                    "prefill_graph": graph_paths["prefill"].name,
                    "decode_graph": graph_paths["decode"].name,
                    "weights": weights_path.name,
                    "max_context_tokens": 8,
                    "input_shape": [1],
                    "model_id": "test/tiny-qwen3.5",
                },
            )

    def test_chatbot_bundle_rejects_batched_export(self) -> None:
        batched = replace(self.bundle, batch_size=2)
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "batch_size=1"):
                qwen3_5_static_cache_to_dli(batched, output_dir)


if __name__ == "__main__":
    unittest.main()
