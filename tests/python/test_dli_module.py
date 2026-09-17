from __future__ import annotations

import json

import torch

import dli


def test_python_engine_runs_aten_graph() -> None:
    graph = dli.Graph.from_json(
        json.dumps(
            {
                "format": "dli.graph.v1",
                "model_type": "python_binding_test",
                "inputs": ["x", "bias"],
                "outputs": ["y"],
                "nodes": [
                    {
                        "name": "add",
                        "op": "aten",
                        "inputs": ["x", "bias"],
                        "outputs": ["y"],
                        "attrs": {
                            "name": "aten::add",
                            "overload": "Tensor",
                            "attr_order": ["alpha"],
                            "alpha": 1.5,
                        },
                    }
                ],
            }
        )
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    bias = torch.tensor([10.0, 20.0], dtype=torch.float32)
    outputs = dli.Engine().run(graph, {"x": x, "bias": bias})

    assert set(outputs) == {"y"}
    torch.testing.assert_close(outputs["y"], x + bias * 1.5)


def test_python_graph_round_trip() -> None:
    graph = dli.Graph.from_json(
        json.dumps(
            {
                "format": "dli.graph.v1",
                "model_type": "round_trip",
                "inputs": ["x"],
                "outputs": ["x"],
                "nodes": [],
            }
        )
    )

    parsed = json.loads(graph.to_json())
    assert parsed["model_type"] == "round_trip"
    assert parsed["inputs"] == ["x"]


def test_python_execution_state_lifecycle() -> None:
    state = dli.ExecutionState()
    assert state.cache_size == 0
    assert state.tensor_count == 0
    state.reset()

    engine = dli.Engine()
    assert engine.cache_size == 0
    graph = dli.Graph.from_json(
        json.dumps(
            {
                "format": "dli.graph.v1",
                "model_type": "state_binding_test",
                "inputs": ["x"],
                "outputs": ["x"],
                "nodes": [],
            }
        )
    )
    x = torch.tensor([1.0])
    outputs = engine.run(graph, {"x": x}, state=state, position_offset=4)
    torch.testing.assert_close(outputs["x"], x)
    engine.reset()


if __name__ == "__main__":
    test_python_engine_runs_aten_graph()
    test_python_graph_round_trip()
    test_python_execution_state_lifecycle()
