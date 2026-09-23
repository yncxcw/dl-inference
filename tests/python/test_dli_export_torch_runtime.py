from __future__ import annotations

import tempfile
import unittest

import torch

import dli
from dli_export.torch_export import exported_program_to_dli


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


class IdentityOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TorchExportDliRuntimeTest(unittest.TestCase):
    def _convert_and_run(
        self,
        module: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        *,
        output_names: list[str],
    ) -> dict[str, torch.Tensor]:
        program = torch.export.export(module.eval(), tuple(inputs.values()), strict=True)
        with tempfile.TemporaryDirectory() as output_dir:
            graph_path, weights_path = exported_program_to_dli(
                program,
                output_dir,
                stem="runtime",
                output_names=output_names,
            )
            graph = dli.Graph.from_json_file(str(graph_path))
            weights = dli.load_weights(str(weights_path))
            return dli.Engine().run(graph, {**weights, **inputs})

    def test_lifted_weight_and_same_named_user_input_remain_distinct(self) -> None:
        outputs = self._convert_and_run(
            UserInputMatchesParameter(),
            {"weight": torch.tensor([2.0])},
            output_names=["logits"],
        )
        torch.testing.assert_close(outputs["logits"], torch.tensor([5.0]))

    def test_output_alias_collision_preserves_returned_value(self) -> None:
        outputs = self._convert_and_run(
            CollidingOutputAliases(),
            {"x": torch.tensor([1.0])},
            output_names=["mul", "logits"],
        )
        torch.testing.assert_close(outputs["mul"], torch.tensor([2.0]))
        torch.testing.assert_close(outputs["logits"], torch.tensor([7.0]))

    def test_user_input_can_be_returned_directly(self) -> None:
        outputs = self._convert_and_run(
            IdentityOutput(),
            {"x": torch.tensor([11.0])},
            output_names=["logits"],
        )
        torch.testing.assert_close(outputs["logits"], torch.tensor([11.0]))


if __name__ == "__main__":
    unittest.main()
