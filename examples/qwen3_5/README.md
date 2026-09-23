# Qwen3.5 DLI Chatbot

This text-only chatbot runs Qwen3.5 model forwards through the native DLI graph
engine. Transformers is used during the one-time checkpoint export and for the
tokenizer/chat template; the chat process does not instantiate a Transformers
model or call `model.generate()`.

The implementation intentionally adds no Qwen-specific DLI operator. It uses:

```text
Qwen3_5ForCausalLM
  -> torch.export ExportedProgram / FX
  -> generic DLI aten graph
  -> dli.Engine
```

## Build and install dependencies

Build the native binding (the Triton AOT plugin is not required for this path):

```bash
cmake -S . -B build \
  -DDLI_BUILD_TESTS=ON \
  -DDLI_BUILD_EXAMPLES=ON \
  -DDLI_ENABLE_TRITON_AOT=OFF
cmake --build build -j
```

Install a Transformers release with Qwen3.5 support:

```bash
python3 -m pip install --upgrade "transformers>=5.2.0" torch
export PYTHONPATH="$PWD/build/python:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## Export the checkpoint

Export a fixed-capacity FP32 artifact bundle:

```bash
python3 -m dli_export.qwen3_5 \
  --model-id Qwen/Qwen3.5-0.8B \
  --output-dir build/examples/qwen3_5 \
  --max-context-tokens 4096
```

The directory contains:

```text
qwen3_5.dli.bundle.json
qwen3_5_step.dli.json
qwen3_5.dli.weights.json
qwen3_5.dli.weights.bin
```

Prefill and decode currently share the same recurrent step graph. Prefill resets
the request-local `dli.ExecutionState`, injects zero cache initializers declared
by the graph, and folds the prompt one token at a time. Decode advances the same
explicit linear-attention and K/V cache tensors for generated tokens. The bundle
keeps separate stage fields so a future chunked prefill graph can be introduced
without changing the chatbot interface.

The current DLI tensor format preserves FP32, int64, and bool, so the exporter
converts the checkpoint to FP32. Expect a roughly 3 GB weight artifact for the
0.8B model. Native BF16 preservation and a chunked prefill graph are future
optimizations.

To export an already cached or local checkpoint without Hub access:

```bash
python3 -m dli_export.qwen3_5 \
  --model-id /path/to/qwen3.5 \
  --output-dir build/examples/qwen3_5 \
  --max-context-tokens 4096 \
  --local-files-only
```

## Run the chatbot

```bash
python3 examples/qwen3_5/main.py \
  --artifacts build/examples/qwen3_5/qwen3_5.dli.bundle.json
```

Use `/reset` to clear history and request state, or `/exit` to quit. Useful
options include:

```bash
python3 examples/qwen3_5/main.py \
  --artifacts build/examples/qwen3_5/qwen3_5.dli.bundle.json \
  --system-prompt "Answer briefly and accurately." \
  --max-new-tokens 256 \
  --temperature 0 \
  --device cpu
```

Use `--device cuda` to load weights and execute generic ATen nodes on CUDA, or
`--device auto` to select CUDA when available. DLI currently supports CPU and
CUDA, not MPS. The tokenizer model ID defaults to the ID recorded in the
bundle; override it with `--model-id` when needed.

One-shot mode streams one response and exits:

```bash
python3 examples/qwen3_5/main.py \
  --artifacts build/examples/qwen3_5/qwen3_5.dli.bundle.json \
  --prompt "Explain KV caching in two sentences."
```

The completed message history is re-rendered and re-prefilled for each turn.
If tokenization, inference, decoding, or output fails, the pending user message
is rolled back and the DLI request state is reset.

## Tests

Frontend tests use fake tokenizer and DLI objects and require no checkpoint:

```bash
PYTHONPATH=python:examples/qwen3_5 python3 examples/qwen3_5/test_chatbot.py
```

The native integration test exports a tiny hybrid Qwen3.5 with one linear-
attention and one full-attention layer, converts its FX graph, runs several
tokens through `dli.Engine`, and compares logits to PyTorch:

```bash
PYTHONPATH=build/python:python python3 tests/python/test_qwen3_5_dli_runtime.py
```
