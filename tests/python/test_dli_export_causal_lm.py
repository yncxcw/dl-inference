from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

try:
    import transformers

    from dli_export.causal_lm import causal_lm_to_dli, default_stem, export_causal_lm

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import dli

    _HAS_DLI = True
except ImportError:
    _HAS_DLI = False

CAPACITY = 8
TOKENS = (5, 7, 9, 11, 13, 2)


def tiny_models() -> dict:
    """One small random model per architecture family, all exported the same way."""

    t = transformers
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2)
    return {
        "llama": lambda: t.LlamaForCausalLM(
            t.LlamaConfig(**common, num_attention_heads=4, num_key_value_heads=2)
        ),
        "qwen3": lambda: t.Qwen3ForCausalLM(
            t.Qwen3Config(**common, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        ),
        "qwen3_5": lambda: t.Qwen3_5ForCausalLM(
            t.Qwen3_5TextConfig(
                **common,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_conv_kernel_dim=4,
                layer_types=["linear_attention", "full_attention"],
            )
        ),
    }


def exported_models():
    for name, factory in tiny_models().items():
        torch.manual_seed(0)
        model = factory()
        exported = export_causal_lm(model, max_cache_len=CAPACITY)
        with torch.no_grad():
            reference = model(input_ids=torch.tensor([TOKENS])).logits[0]
        yield name, exported, reference


@unittest.skipUnless(_HAS_TRANSFORMERS, "Transformers is required")
class CausalLMExportTest(unittest.TestCase):
    def test_state_is_discovered_from_the_cache_of_each_architecture(self) -> None:
        for name, exported, _ in exported_models():
            with self.subTest(model=name):
                names = exported.state_names
                self.assertEqual(names[-1], "cursor")
                self.assertEqual(len(set(names)), len(names))
                kv = [
                    spec
                    for spec in exported.state_specs
                    if spec.path[1:] in (("keys",), ("values",))
                ]
                self.assertTrue(kv)
                self.assertTrue(all(spec.shape[2] == CAPACITY for spec in kv))
                recurrent = [
                    s for s in exported.state_specs if s.path[1:2] == ("recurrent_states",)
                ]
                self.assertEqual(bool(recurrent), name == "qwen3_5")
                self.assertRegex(exported.model_type, rf"^{name}(_text)?_step$")
                self.assertEqual(set(exported.state_inputs.values()), set(names))

    def test_step_module_matches_the_full_sequence_forward(self) -> None:
        for name, exported, reference in exported_models():
            with self.subTest(model=name), torch.no_grad():
                state = exported.initial_state()
                for position, token in enumerate(TOKENS):
                    logits, *state = exported.step_module(torch.tensor([token]), *state)
                    torch.testing.assert_close(
                        logits[0, -1], reference[position], rtol=1e-4, atol=1e-4
                    )

    @unittest.skipUnless(_HAS_DLI, "the native DLI module is required")
    def test_dli_runtime_matches_the_full_sequence_forward(self) -> None:
        for name, exported, reference in exported_models():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                bundle_path = causal_lm_to_dli(
                    exported, directory, stem=f"tiny_{name}", model_id=f"tiny/{name}"
                )
                bundle = json.loads(bundle_path.read_text())
                self.assertEqual(bundle["format"], "dli.autoregressive.v1")
                self.assertEqual(bundle["max_context_tokens"], CAPACITY)
                self.assertEqual(bundle["prefill_graph"], bundle["decode_graph"])
                graph = dli.Graph.from_json_file(str(Path(directory) / bundle["decode_graph"]))
                weights = dli.load_weights(str(Path(directory) / bundle["weights"]))
                engine = dli.Engine()
                state = dli.ExecutionState()
                for position, token in enumerate(TOKENS):
                    logits = engine.run(
                        graph,
                        {**weights, "input_ids": torch.tensor([token])},
                        state=state,
                    )["logits"]
                    torch.testing.assert_close(
                        logits[0, -1], reference[position], rtol=1e-4, atol=1e-4
                    )
                self.assertEqual(state.tensor_count, len(exported.state_specs))

    def test_qwen3_5_alias_keeps_its_file_stem(self) -> None:
        from unittest import mock

        from dli_export import qwen3_5

        torch.manual_seed(0)
        model = tiny_models()["qwen3_5"]()
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = qwen3_5.export_qwen3_5_to_dli(model, directory, max_cache_len=CAPACITY)
            self.assertEqual(bundle_path.name, "qwen3_5.dli.bundle.json")
            bundle = json.loads(bundle_path.read_text())
            self.assertEqual(bundle["decode_graph"], "qwen3_5_step.dli.json")
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            qwen3_5.export_qwen3_5_to_dli(model, "unused", max_cache_len=CAPACITY, batch_size=2)

        arguments = ["--model-id", "m", "--output-dir", "d", "--max-context-tokens", "8"]
        with mock.patch.object(qwen3_5, "_causal_lm_main", return_value=0) as generic_main:
            qwen3_5._main(arguments)
            qwen3_5._main([*arguments, "--stem=custom"])
        self.assertEqual(generic_main.call_args_list[0].args[0], [*arguments, "--stem", "qwen3_5"])
        self.assertEqual(generic_main.call_args_list[1].args[0], [*arguments, "--stem=custom"])

    def test_default_stem(self) -> None:
        self.assertEqual(default_stem("Qwen/Qwen3-0.6B"), "qwen3-0.6b")
        self.assertEqual(default_stem("/models/My Model/"), "my_model")


if __name__ == "__main__":
    unittest.main()
