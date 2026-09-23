# Exported Autoregressive Chatbot Design

## Decision

Qwen3.5 is not implemented as a DLI operator. Model math follows one generic
path:

```text
Qwen3.5 PyTorch module
        |
        | torch.export.export(...)
        v
fixed-shape ExportedProgram / FX graph
        |
        | run_decompositions({}) + generic ATen serialization
        v
dli.graph.v1 (aten nodes) + shared weights
        |
        v
DLI Engine + per-request ExecutionState
```

The model-specific Python frontend only defines the generation boundary and
flattens Transformers' cache object into tensors. The converter and runtime do
not contain Qwen-specific kernels, node types, or dispatch code.

The first implementation is text-only, batch one, and FP32. It deliberately
uses fixed tensor shapes and tokenwise prompt processing to establish a small,
testable correctness baseline. Chunked prefill, dynamic batching, BF16, and
fused kernels are independent follow-up optimizations.

## Prefill and Decode Lifecycle

Transformers' Qwen3.5 implementation takes different eager branches for an
empty cache and an initialized cache, especially in Gated DeltaNet layers. Its
empty-cache branch uses the chunked implementation even for one token, making
that FX graph much larger than the recurrent graph. Instead, the exporter
captures one fixed-shape functional recurrent program with explicit state:

```text
step(input_ids[batch], cache tensors)
    -> logits, replacement cache tensors
```

Zero recurrent/KV state is numerically equivalent to the empty-cache branch for
the first token. At prefill, DLI resets the request state, injects graph-declared
zero initializers, and folds all `N` prompt tokens through `step`. Decode keeps
the published state and runs `step` once per generated token. The bundle retains
separate prefill/decode fields, but both currently reference the same graph.
A future multi-token prefill graph can replace only the prefill field without
changing the cache ABI or chatbot.

The program is exported with fixed batch and cache capacity. This avoids
symbolic shape values in DLI IR and prevents K/V tensors from reallocating as
the sequence grows.

## Generic ExportedProgram Conversion

`dli_export.torch_export.exported_program_to_dli` consumes a
`torch.export.ExportedProgram`. It:

1. runs `run_decompositions({})` to functionalize mutations while retaining
   useful high-level ATen operations such as `linear`, `conv1d`, and scaled
   dot-product attention;
2. writes lifted parameters, buffers, tensor constants, and state initializers
   with `WeightWriter`;
3. lowers every computation node to DLI's generic `aten` operator;
4. serializes schema arguments in their exact order, including tensor lists,
   optional tensor lists, `None`, scalar/list values, dtype, layout, memory
   format, and a runtime tensor anchor for device values;
5. resolves FX `getitem` nodes into the multiple outputs of their producing
   ATen node; and
6. emits graph-level state bindings supplied by the generation frontend.

The DLI `aten` operator reconstructs boxed `IValue` arguments and invokes the
normal PyTorch dispatcher. This makes the import path reusable for other
fixed-shape exported PyTorch models.

## Cache ABI

Cache tensors are ordinary graph edges. They are not hidden inside an operator.
For Qwen3.5 each linear-attention layer contributes:

- a causal-convolution history tensor;
- a Gated DeltaNet recurrent-state tensor.

Each full-attention layer contributes:

- a fixed-capacity key tensor;
- a fixed-capacity value tensor.

One scalar cursor records the number of occupied positions. All full-attention
layers advance in lockstep, so one cursor is sufficient. The official 0.8B
layout is:

- 18 linear-attention layers, each with convolution state
  `[1, 6144, 4]` and recurrent state `[1, 16, 128, 128]`;
- 6 full-attention layers, each with K/V
  `[1, 2, max_context_tokens, 256]`; and
- one scalar `int64` cursor.

The graph format declares three mappings:

```json
{
  "state_inputs": {"past_0": "layer.0.linear.conv"},
  "state_outputs": {"next_0": "layer.0.linear.conv"},
  "state_initializers": {
    "layer.0.linear.conv": "qwen3_5_step.__state_initializer__.layer.0.linear.conv"
  }
}
```

Before execution, `Engine` injects each state input from the request's
`ExecutionState`, using the named initializer when the key is absent. After all
nodes and public outputs validate, it publishes every state output. Publication
is transactional: a failed node cannot leave a partially advanced request.

The graph and weights are immutable and shared. Every conversation receives a
separate `ExecutionState`, so caches cannot leak between chats.

## Runtime Flow

`EngineCausalLM` owns the stage graph handles, shared static weights, one
explicit `ExecutionState`, and the current position. In the current bundle both
handles resolve to the same step graph:

```text
render chat template -> token IDs
    reset request state
    for token in prompt: step_graph(token)       # prefill lifecycle
    sample next token from logits
    step_graph(sampled token)                    # decode lifecycle
    repeat until stop/context limit
```

`generate_tokens` remains backend-neutral and supplies greedy, temperature,
top-k, top-p, deterministic seed, EOS, and context-limit behavior. The chatbot
uses Transformers only for the checkpoint-to-artifact export and tokenizer. At
chat runtime, all model forwards go through `dli.Engine`; it never calls
`PreTrainedModel.generate`.

Each new chat turn re-renders and re-prefills the full committed history. This
matches the tokenizer's chat-template semantics and makes a failed turn easy to
roll back.

## Artifact Bundle

One step graph and its weights manifest/data file are shared by both lifecycle
stages. A small `dli.autoregressive.v1` bundle points the chatbot at:

- prefill graph;
- decode graph;
- shared weights;
- maximum context length;
- one-token input shape; and
- optional source model ID.

Paths in the bundle are relative to the bundle file, so the artifact directory
is relocatable.

## Acceptance Gates

- No Qwen-specific C++ operator is registered.
- A generic exported MLP executes through DLI with exact PyTorch parity.
- Generic exported input mutation becomes transactional named graph state.
- Tiny hybrid Qwen3.5 (`linear_attention`, `full_attention`) matches PyTorch for
  the first token and multiple decode tokens through the native DLI engine.
- Prefill and decode lifecycle entries share one graph and weights file.
- Two explicit request states remain isolated; reset reproduces a clean first
  step; failed execution does not publish cache updates.
- The chatbot tests fail if the frontend depends on `model.generate`.

## Current Limits

- FP32 only for this first Qwen3.5 path. DLI tensor/weight support must preserve
  BF16 before exporting the checkpoint in its native dtype.
- Batch size one in the chatbot.
- Prompt prefill is token-by-token and therefore slow.
- Fixed cache capacity is chosen at export time.
- The generic ATen dispatcher path prioritizes correctness, not fusion or
  kernel-launch efficiency.
- Text only; vision/video and conditional-generation inputs are out of scope.
