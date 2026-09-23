from __future__ import annotations

import tempfile
import unittest

import torch


try:
    import dli
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from dli_export.qwen3_5 import (
        export_qwen3_5_static_cache,
        qwen3_5_static_cache_to_dli,
    )

    _HAS_RUNTIME = True
except (ImportError, AttributeError):
    _HAS_RUNTIME = False


@unittest.skipUnless(_HAS_RUNTIME, "Qwen3.5 and the native DLI module are required")
class Qwen3_5DliRuntimeTest(unittest.TestCase):
    def test_exported_hybrid_model_matches_pytorch_across_decode_steps(self) -> None:
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
        model = Qwen3_5ForCausalLM(config).eval()
        exported = export_qwen3_5_static_cache(
            model,
            max_cache_len=8,
            strict=True,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            graph_paths, weights_path = qwen3_5_static_cache_to_dli(
                exported,
                output_dir,
                stem="tiny_qwen3_5",
            )
            prefill_graph = dli.Graph.from_json_file(str(graph_paths["prefill"]))
            decode_graph = dli.Graph.from_json_file(str(graph_paths["decode"]))
            weights = dli.load_weights(str(weights_path))
            engine = dli.Engine()
            state = dli.ExecutionState()

            token = torch.tensor([5], dtype=torch.int64)
            expected = exported.prefill.eager_module(
                token, *exported.flat_initial_state()
            )
            actual = engine.run(
                prefill_graph,
                {**weights, "input_ids": token},
                state=state,
            )["logits"]
            torch.testing.assert_close(actual, expected[0], rtol=1e-4, atol=1e-5)
            expected_state = tuple(tensor.clone() for tensor in expected[1:])
            self.assertEqual(state.tensor_count, len(exported.state_specs))

            for token_id in (7, 9, 11):
                token = torch.tensor([token_id], dtype=torch.int64)
                expected = exported.decode.eager_module(token, *expected_state)
                actual = engine.run(
                    decode_graph,
                    {**weights, "input_ids": token},
                    state=state,
                )["logits"]
                torch.testing.assert_close(actual, expected[0], rtol=1e-4, atol=1e-5)
                expected_state = tuple(tensor.clone() for tensor in expected[1:])


if __name__ == "__main__":
    unittest.main()
