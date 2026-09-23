from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from dli_export.torch_export import TorchExportError, exported_program_to_dli
from dli_export.weights import WeightWriter


class RichAtenArguments(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, 0.5]))
        self.register_buffer("enabled", torch.tensor(True), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.tensor([0], dtype=torch.int64, device=x.device)
        selected = x[:, index]
        joined = torch.cat([selected, selected], dim=1)
        positions = torch.arange(0, 2, device=x.device)[:, None]
        converted = joined.to(torch.float32).clone(memory_format=torch.contiguous_format)
        return converted + positions + self.weight + self.enabled + 1.25


class StatefulAdd(torch.nn.Module):
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state.add_(x)
        return state * 2


class TupleOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = torch.max(x, dim=1)
        return values


class SpecializedKwargs(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x * attention_mask
        return x + logits_to_keep


class UnsupportedDType(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float64)


class UserInputMatchesParameter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.0]))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.weight


class CollidingOutputAliases(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = x + 1
        intermediate = first * 2
        final = intermediate + 3
        return first, final


class DuplicateOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = x + 1
        return result, result


class DuplicateInputOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exported_program_serializes_generic_aten_arguments_and_weights() -> None:
    program = torch.export.export(
        RichAtenArguments().eval(),
        (torch.arange(6, dtype=torch.int64).reshape(2, 3),),
        strict=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = exported_program_to_dli(
            program,
            tmp,
            model_type="rich_aten_test",
            stem="rich",
            output_names=["logits"],
        )
        graph = _load(graph_path)
        manifest = _load(weights_path)

    assert graph["format"] == "dli.graph.v1"
    assert graph["model_type"] == "rich_aten_test"
    assert graph["inputs"] == ["x"]
    assert graph["outputs"] == ["logits"]
    assert graph["nodes"][-1]["outputs"] == ["logits"]
    assert all(node["op"] == "aten" for node in graph["nodes"])
    assert all(node["attrs"]["name"] != "aten::_assert_tensor_metadata" for node in graph["nodes"])

    by_name = {node["attrs"]["name"]: node for node in graph["nodes"]}
    assert by_name["aten::cat"]["attrs"]["arguments"][0].startswith("tl:")
    assert by_name["aten::index"]["attrs"]["arguments"][1].startswith("otl:n,")
    assert "dtype:float32" in by_name["aten::_to_copy"]["attrs"]["arguments"]
    assert "memory_format:contiguous" in by_name["aten::clone"]["attrs"]["arguments"]

    arange = by_name["aten::arange"]
    device_argument = next(
        argument for argument in arange["attrs"]["arguments"] if argument.startswith("device:")
    )
    device_input = int(device_argument.removeprefix("device:"))
    assert 0 <= device_input < len(arange["inputs"])

    scalar_add = next(
        node
        for node in graph["nodes"]
        if node["attrs"]["name"] == "aten::add" and node["attrs"]["overload"] == "Scalar"
    )
    assert "f:1.25" in scalar_add["attrs"]["arguments"]
    assert "i:1" in scalar_add["attrs"]["arguments"]  # materialized alpha default

    dtypes = {metadata["dtype"] for metadata in manifest["tensors"].values()}
    assert {"bool", "float32", "int64"} <= dtypes
    assert any(
        metadata["dtype"] == "bool" and metadata["shape"] == []
        for metadata in manifest["tensors"].values()
    )


def test_exported_program_functionalizes_and_declares_request_state() -> None:
    program = torch.export.export(
        StatefulAdd(),
        (torch.ones(2), torch.zeros(2)),
        strict=True,
    )

    # run_decompositions({}) rewrites add_ into a functional add and identifies
    # that result as the replacement value for user input "state".
    decomposed = program.run_decompositions({})
    mutation = next(
        spec
        for spec in decomposed.graph_signature.output_specs
        if spec.kind.name == "USER_INPUT_MUTATION"
    )
    mutation_name = mutation.arg.name

    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = exported_program_to_dli(
            program,
            tmp,
            stem="stateful",
            output_names=["logits"],
            state_inputs={"state": "accumulator"},
            state_outputs={mutation_name: "accumulator"},
            state_initializers={"accumulator": torch.zeros(2)},
        )
        graph = _load(graph_path)
        manifest = _load(weights_path)

    assert graph["inputs"] == ["x"]
    assert graph["outputs"] == ["logits"]
    assert graph["state_inputs"] == {"state": "accumulator"}
    assert graph["state_outputs"] == {mutation_name: "accumulator"}
    initializer = graph["state_initializers"]["accumulator"]
    assert manifest["tensors"][initializer] == {
        "dtype": "float32",
        "shape": [2],
        "offset": manifest["tensors"][initializer]["offset"],
        "nbytes": 8,
    }
    assert all(not node["attrs"]["name"].endswith("_") for node in graph["nodes"])


def test_exported_program_resolves_builtin_getitem_from_multi_output_aten() -> None:
    program = torch.export.export(TupleOutput(), (torch.randn(2, 3),), strict=True)

    with tempfile.TemporaryDirectory() as tmp:
        graph_path, _ = exported_program_to_dli(
            program,
            tmp,
            stem="tuple_output",
            output_names=["logits"],
        )
        graph = _load(graph_path)

    assert len(graph["nodes"]) == 1
    maximum = graph["nodes"][0]
    assert maximum["attrs"]["name"] == "aten::max"
    assert maximum["attrs"]["overload"] == "dim"
    assert maximum["outputs"] == ["logits", "max_1_1"]
    assert graph["outputs"] == ["logits"]


def test_exported_program_skips_specialized_non_tensor_user_inputs() -> None:
    program = torch.export.export(
        SpecializedKwargs(),
        (torch.ones(2),),
        kwargs={"attention_mask": None, "logits_to_keep": 1},
        strict=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        graph_path, _ = exported_program_to_dli(program, tmp, stem="specialized_kwargs")
        graph = _load(graph_path)

    assert graph["inputs"] == ["x"]
    assert graph["outputs"] == ["logits"]
    assert graph["nodes"][0]["attrs"]["name"] == "aten::add"
    assert graph["nodes"][0]["attrs"]["overload"] == "Scalar"
    assert "i:1" in graph["nodes"][0]["attrs"]["arguments"]


def test_exported_program_can_share_one_weight_manifest() -> None:
    module = torch.nn.Linear(3, 2).eval()
    program = torch.export.export(module, (torch.ones(1, 3),), strict=True)

    with tempfile.TemporaryDirectory() as tmp:
        shared = WeightWriter(tmp, "shared")
        prefill_graph_path, prefill_weights_path = exported_program_to_dli(
            program,
            tmp,
            stem="prefill",
            weight_writer=shared,
        )
        first_manifest = _load(prefill_weights_path)
        decode_graph_path, decode_weights_path = exported_program_to_dli(
            program,
            tmp,
            stem="decode",
            weight_writer=shared,
        )
        prefill_graph = _load(prefill_graph_path)
        decode_graph = _load(decode_graph_path)
        final_manifest = _load(decode_weights_path)

    assert prefill_weights_path == decode_weights_path
    assert prefill_graph["weights"] == "shared.dli.weights.json"
    assert decode_graph["weights"] == "shared.dli.weights.json"
    assert final_manifest["tensors"] == first_manifest["tensors"]


def test_exported_program_rejects_tensor_dtypes_the_runtime_cannot_preserve() -> None:
    program = torch.export.export(UnsupportedDType(), (torch.ones(2),), strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            exported_program_to_dli(program, tmp, stem="unsupported_dtype")
        except Exception as error:
            assert "unsupported tensor dtype torch.float64" in str(error)
        else:
            raise AssertionError("unsupported float64 graph was accepted")


def test_lifted_weight_name_does_not_alias_same_named_user_input() -> None:
    program = torch.export.export(
        UserInputMatchesParameter(), (torch.tensor([2.0]),), strict=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, weights_path = exported_program_to_dli(program, tmp, stem="namespaced")
        graph = _load(graph_path)
        manifest = _load(weights_path)

    add = graph["nodes"][0]
    assert graph["inputs"] == ["weight"]
    assert add["inputs"][0] == "weight"
    assert add["inputs"][1] != "weight"
    assert add["inputs"][1] in manifest["tensors"]


def test_output_alias_collision_is_materialized_after_live_edge_consumers() -> None:
    program = torch.export.export(CollidingOutputAliases(), (torch.tensor([1.0]),), strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, _ = exported_program_to_dli(
            program,
            tmp,
            stem="colliding_outputs",
            output_names=["mul", "logits"],
        )
        graph = _load(graph_path)

    assert graph["outputs"] == ["mul", "logits"]
    assert [node["outputs"] for node in graph["nodes"][:3]] == [
        ["add"],
        ["mul"],
        ["logits"],
    ]
    alias = graph["nodes"][-1]
    assert alias["attrs"]["name"] == "aten::alias"
    assert alias["inputs"] == ["add"]
    assert alias["outputs"] == ["mul"]


def test_duplicate_returned_tensor_gets_independent_output_aliases() -> None:
    program = torch.export.export(DuplicateOutput(), (torch.tensor([1.0]),), strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, _ = exported_program_to_dli(
            program,
            tmp,
            stem="duplicate_output",
            output_names=["first", "second"],
        )
        graph = _load(graph_path)

    assert graph["outputs"] == ["first", "second"]
    assert graph["nodes"][0]["outputs"] == ["add"]
    aliases = graph["nodes"][1:]
    assert [node["attrs"]["name"] for node in aliases] == ["aten::alias", "aten::alias"]
    assert [node["inputs"] for node in aliases] == [["add"], ["add"]]
    assert [node["outputs"] for node in aliases] == [["first"], ["second"]]


def test_returned_input_gets_materialized_output_aliases() -> None:
    program = torch.export.export(DuplicateInputOutput(), (torch.tensor([1.0]),), strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        graph_path, _ = exported_program_to_dli(
            program,
            tmp,
            stem="input_output",
            output_names=["first", "second"],
        )
        graph = _load(graph_path)

    assert graph["inputs"] == ["x"]
    assert graph["outputs"] == ["first", "second"]
    assert [node["attrs"]["name"] for node in graph["nodes"]] == [
        "aten::alias",
        "aten::alias",
    ]
    assert [node["inputs"] for node in graph["nodes"]] == [["x"], ["x"]]
    assert [node["outputs"] for node in graph["nodes"]] == [["first"], ["second"]]


def test_output_alias_cannot_overwrite_another_returned_source() -> None:
    program = torch.export.export(CollidingOutputAliases(), (torch.tensor([1.0]),), strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            exported_program_to_dli(
                program,
                tmp,
                stem="source_collision",
                output_names=["add_1", "first"],
            )
        except TorchExportError as error:
            assert "aliases another returned tensor edge" in str(error)
        else:
            raise AssertionError("an output alias overwrote another returned source")


if __name__ == "__main__":
    test_exported_program_serializes_generic_aten_arguments_and_weights()
    test_exported_program_functionalizes_and_declares_request_state()
    test_exported_program_resolves_builtin_getitem_from_multi_output_aten()
    test_exported_program_skips_specialized_non_tensor_user_inputs()
    test_exported_program_can_share_one_weight_manifest()
    test_exported_program_rejects_tensor_dtypes_the_runtime_cannot_preserve()
    test_lifted_weight_name_does_not_alias_same_named_user_input()
    test_output_alias_collision_is_materialized_after_live_edge_consumers()
    test_duplicate_returned_tensor_gets_independent_output_aliases()
    test_returned_input_gets_materialized_output_aliases()
    test_output_alias_cannot_overwrite_another_returned_source()
