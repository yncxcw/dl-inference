# Exported Autoregressive Chatbot Design

## Design

We build a generic export script, `dli_export.causal_lm`, for LLMs with a
causal attention mask: the decoder-only Hugging Face models that load with
`AutoModelForCausalLM`. It captures the model with `torch.export` as a
one-token step program, lowers every ATen call in that program to a node of
DLI's generic `aten` operator, and writes a bundle that `dli.Engine` runs with
a separate cache for each request. Qwen3.5 is the first target; the tests
export tiny Llama and Qwen3 models with the same exporter.

```text
Hugging Face causal LM (PyTorch)
        |
        |  Part 1: export a one-token step and lower each ATen call
        |          to an aten node
        v
step graph (dli.graph.v1) + weights + dli.autoregressive.v1 bundle
        |
        |  Part 2: run prefill and decode on dli.Engine, with the KV cache
        |          in a per-request ExecutionState
        v
generated tokens
        |
        |  Part 3: the chatbot renders the chat and streams the reply
        v
terminal chat
```

This version is text-only, batch size one and FP32, and its graph has fixed
shapes: no tensor size depends on how many tokens the sequence holds.

- Each graph run takes one token, so the token input is always `[1]`.
- The cache capacity, `max_context_tokens`, is chosen at export. K/V tensors
  are `[1, kv_heads, max_context_tokens, head_dim]` however many tokens they
  hold.
- The sequence length is a value, not a shape: an int64 `cursor` input holds
  the position of the current token, and an attention mask over the whole
  capacity hides the slots not written yet.

Chunked prefill, batching, BF16 and fused kernels are later optimizations.

### Why the step takes one token

- **Decode needs a one-token graph anyway.** A model generates one token at a
  time, so decode runs one token per step. Running prompt tokens through the
  same graph leaves one graph to export, convert and test.
- **It keeps shapes fixed.** DLI graphs have no symbolic shapes. A graph that
  took a whole prompt would need a different input shape for every prompt
  length; with one token per run, the length moves into the `cursor` value.
- **It avoids a much larger graph for Qwen3.5.** Transformers' Qwen3.5 code
  runs a different branch for an empty cache than for a filled one. In Gated
  DeltaNet layers the empty-cache branch uses the chunked algorithm even for
  one token, which traces into a much larger graph. After a warm-up step the
  cache counts as filled, so tracing records the small recurrent branch, and
  running it from zero state gives the same result as the empty-cache branch.
- **It is a simple correctness baseline.** Prefill does exactly what decode
  does, so the two cannot diverge, and each token's logits can be checked
  against Transformers.

The cost is speed: a prompt of N tokens takes N graph runs. A chunked prefill
graph that takes a fixed number of tokens per run is the planned follow-up,
and the bundle keeps a separate `prefill_graph` field for it.

The rest of this doc has three parts:

1. **Export:** how a PyTorch model becomes a graph of DLI operators.
2. **Run:** prefill, decode, and how the KV cache is managed.
3. **End-to-end workflow:** how the chatbot uses the pieces.

## Part 1: Export a PyTorch model to DLI operators

Export runs once, offline, where PyTorch and Transformers are installed:

```bash
python3 -m dli_export.causal_lm \
  --model-id Qwen/Qwen3.5-0.8B \
  --output-dir build/examples/qwen3_5 \
  --max-context-tokens 4096 \
  --stem qwen3_5
```

It has two stages. `dli_export.causal_lm.export_causal_lm` captures the model
as a one-token step program, and
`dli_export.torch_export.exported_program_to_dli` lowers that program to DLI
nodes. The second stage accepts any fixed-shape `torch.export` program, not
only language models.

### The one-token step program

A language model's forward pass reads and updates a cache, but `torch.export`
needs a function whose state is explicit. The exporter wraps the model in
`CausalLMStep`:

```text
step(input_ids[1], *state) -> (logits[1, 1, vocab], *next_state)
```

`state` is every cache tensor followed by a scalar int64 `cursor`, the
position of the token being run; the step returns `cursor + 1`. The exporter
finds the cache tensors without knowing the architecture:

1. It builds a Transformers `StaticCache` for the model's config, with
   `max_context_tokens` slots. `StaticCache` picks the layer type each layer
   needs (full attention or linear attention) and keeps all per-step state in
   tensors.
2. It runs a warm-up step so every layer allocates its tensors. Each tensor a
   layer holds, as an attribute or inside a dict or list, becomes one named
   state tensor, such as `layer.3.keys`. A second warm-up step checks that no
   non-tensor layer state changes between steps; tracing would freeze such
   state, so export fails instead.
3. `CausalLMStep.forward` rebuilds a cache around copies of the state inputs,
   runs the model on one token at position `cursor` with an attention mask
   that shows cache slots `0` to `cursor`, and returns the logits and the
   cache's updated tensors.
4. While tracing, functions that Transformers' Hub-kernel loader wraps are
   replaced by their PyTorch reference implementations, so optional fused
   kernels never reach the graph.

The exporter switches the model to FP32 evaluation mode, traces the step with
`torch.export.export` using zero state as the example input, and runs
`run_decompositions({})`. That call functionalizes the program, so the model's
in-place cache writes become ordinary outputs. With an empty decomposition
table it keeps high-level operators such as `aten::linear`, `aten::conv1d` and
`aten::scaled_dot_product_attention` whole instead of breaking them into
primitives.

### How each operator is exported

`exported_program_to_dli` walks the FX graph of the functional program in
order and handles each node by kind:

| FX node | DLI result |
| --- | --- |
| Tensor user input (`placeholder`) | Graph input edge, such as `input_ids` or `state_0` |
| Non-tensor user input, such as `logits_to_keep=1` | Nothing; `torch.export` specialized it into the program |
| Parameter, buffer, lifted constant or `get_attr` | Weight named after its module path, such as `model.model.layers.0.linear_attn.in_proj_qkv.weight` |
| ATen call (`call_function` on an `OpOverload`) | One `aten` node |
| `operator.getitem` | No node; selects one output edge of the multi-output call it reads |
| `aten::_assert_tensor_metadata` | Dropped; it only checks export-time metadata |
| `output` | Graph outputs, named `logits` and `next_state.<state name>` |

Anything else, such as a custom operator or a higher-order operator left after
decomposition, fails the export with a `TorchExportError`. So does a tensor
whose dtype is not bool, float32 or int64, the dtypes DLI stores unchanged.

An ATen call becomes a node that names the operator and overload and lists
every schema argument in order, defaults included. These two nodes come from a
two-layer test model:

```json
{
  "name": "linear",
  "op": "aten",
  "inputs": ["type_as", "model.model.layers.0.linear_attn.in_proj_qkv.weight"],
  "outputs": ["linear"],
  "attrs": {"name": "aten::linear", "overload": "", "arguments": ["t:0", "t:1", "n"]}
}
{
  "name": "scaled_dot_product_attention",
  "op": "aten",
  "inputs": ["cat_5", "view_12", "view_13", "expand_3"],
  "outputs": ["scaled_dot_product_attention"],
  "attrs": {
    "name": "aten::scaled_dot_product_attention",
    "overload": "",
    "arguments": ["t:0", "t:1", "t:2", "t:3", "f:0.0", "b:false", "f:0.25", "b:false"]
  }
}
```

The first is `aten::linear(input, weight, bias=None)`: node inputs 0 and 1,
then None for the bias. The second passes query, key, value and mask, then
`dropout_p=0.0`, `is_causal=False`, `scale=0.25` and `enable_gqa=False`. An
empty overload means the default overload.

Each argument is encoded by its schema type:

| Schema type | Encoding | Example |
| --- | --- | --- |
| `Tensor` | `t:<i>`, node input `i` | `t:0` |
| `Tensor[]`, `Tensor?[]` | `tl:` or `otl:` with input indices; `n` marks None | `tl:0,1`, `otl:1,2` |
| Missing optional value | `n` | `n` |
| `int`, `float`, `bool`, `str` | `i:`, `f:`, `b:`, `s:` | `i:-1`, `f:0.25`, `b:true` |
| `int[]`, `float[]`, `bool[]` | `il:`, `fl:`, `bl:` | `il:1,1,32` |
| `ScalarType` | `dtype:<name>` | `dtype:float32` |
| `Layout` | `layout:strided` | `layout:strided` |
| `MemoryFormat` | `memory_format:<name>` | `memory_format:contiguous` |
| `Device` | `device:<i>`, the device of node input `i` | `device:0` |

Some cases need extra care:

- **Devices.** The graph runs on whatever device its weights are loaded on, so
  a device is never written literally. The converter picks a tensor already on
  that device, preferring the call's own arguments, adds it as a node input
  and records its index. `aten::arange(8, device=...)` uses the embedding
  weight as its anchor: `["i:8", "n", "n", "device:0", "b:false"]`.
- **Overloads.** `torch.export` can record an overload that rejects the
  recorded arguments, such as `aten.add.Tensor` with a Python float. The
  converter binds the arguments against every overload of the operator, picks
  the best fit, and fails if two fit equally well.
- **Tensor literals** in arguments are written to the weights file.
- **Several results.** A call that returns several tensors gets one output
  edge per tensor: `aten::split_with_sizes` produces `split_with_sizes_0`,
  `_1` and `_2`, and each `getitem` maps to one of them.
- **Output names.** If renaming a result to its output name could overwrite an
  edge that is still in use, the converter adds a final `aten::alias` node
  instead.

### Running an exported operator

One runtime operator, `aten` in `src/aten_operator.cc`, runs all of these
nodes. For each node it:

1. decodes every argument onto a boxed `c10::Stack`, taking tensors from the
   node's inputs;
2. looks up the schema with
   `c10::Dispatcher::findSchemaOrThrow(name, overload)`;
3. rejects schemas that write to an argument, since graphs must be
   functional;
4. calls the operator through the PyTorch dispatcher; and
5. flattens the returned tensors, including tensor lists and tuples, onto the
   node's output edges.

Any ATen operator that PyTorch can run on CPU or CUDA therefore runs in DLI
without per-operator code. The cost is one dispatcher call per node and no
fusion across nodes.

### State edges and the artifact bundle

The graph declares which of its inputs and outputs are request state rather
than ordinary tensors. From the same test model, with one entry of each map
shown:

```json
"inputs": ["input_ids"],
"outputs": ["logits"],
"state_inputs": {"state_0": "layer.0.conv_states.0"},
"state_outputs": {"next_state.layer.0.conv_states.0": "layer.0.conv_states.0"},
"state_initializers": {
  "layer.0.conv_states.0": "tiny_step.__state_initializer__.layer.0.conv_states.0"
}
```

`state_inputs` maps a graph input to a state name, `state_outputs` maps a
produced edge to the state it replaces, and `state_initializers` names a zero
tensor in the weights file for each state, used while a request has no value
yet. Part 2 describes how the engine uses them.

Export writes four files:

```text
qwen3_5.dli.bundle.json    dli.autoregressive.v1 manifest
qwen3_5_step.dli.json      the step graph
qwen3_5.dli.weights.json   weight manifest
qwen3_5.dli.weights.bin    weights, state initializers and tensor literals
```

The bundle names the step graph as both its `prefill_graph` and its
`decode_graph`, and records the `weights` file, `max_context_tokens` (the
cache capacity), the one-token `input_shape` `[1]` and the source `model_id`.
Paths are relative to the bundle, so the directory can be moved. Separate
prefill and decode fields let a later multi-token prefill graph replace only
`prefill_graph`.

## Part 2: Run the model

`dli.EngineCausalLM` drives the step graph for one conversation. It holds the
loaded graphs, the shared weights, one `dli.ExecutionState` with the request's
cache, and the number of tokens in that cache. Each graph run processes one
token:

```python
outputs = engine.run(graph, {"input_ids": token, **weights},
                     state=execution_state, position_offset=position)
```

`dli.generate_tokens` is the generation loop. It works with any backend that
has `prefill` and `decode` methods, so sampling does not depend on DLI.

### Prefill

`EngineCausalLM.prefill(prompt_ids)` runs the prompt and returns the logits
for the token after it:

1. If the prompt starts with exactly the tokens already in the cache and adds
   at least one more, the cache is kept and only the new tokens run. This is
   the usual case for the next turn of a chat. Otherwise the request state is
   reset and every prompt token runs.
2. Each token is one run of the step graph. The first token after a reset runs
   the bundle's `prefill_graph`, and later tokens run `decode_graph`; today
   both are the same graph.
3. The logits of the last prompt token are returned with a decode handle.

A prompt of N new tokens costs N graph runs; the Design section explains why
prefill uses the one-token graph. A chunked prefill graph can later replace
`prefill_graph` without changing the cache layout or the chatbot.

### Decode

`generate_tokens(backend, prompt_ids, config)` prefills the prompt, then
repeats these steps:

1. Pick the next token from the latest logits: the most likely token at
   temperature 0; otherwise apply the temperature, then top-k, then top-p,
   and draw from a generator that can be seeded.
2. Yield the token, so callers can stream it.
3. Stop after a stop token (which is still yielded), after `max_new_tokens`
   tokens, or when the context is full.
4. Otherwise call `decode(token, handle)`, which runs the step graph once for
   that token and returns the next logits and a new handle.

The last generated token never runs through the graph, because its logits
would not be used. `decode` accepts only the newest handle from the same
adapter; a handle issued before a later prefill, decode or reset is rejected,
so two callers cannot interleave steps of different sequences on one cache.

### KV cache management

**Where the cache lives.** Graphs and weights are immutable and shared. All
mutable state of a conversation lives in its own `ExecutionState`, so caches
cannot leak between conversations. For an exported model that state is the
set of named tensors found at export. Qwen3.5-0.8B has 18 linear-attention and
6 full-attention layers, which gives 55 state tensors:

| Layer type | State tensor | Shape |
| --- | --- | --- |
| Linear attention (18 layers) | `conv_states.0`: causal-convolution history | `[1, 6144, 4]` |
| | `recurrent_states.0`: Gated DeltaNet state | `[1, 16, 128, 128]` |
| Full attention (6 layers) | `keys`, `values` | `[1, 2, max_context_tokens, 256]` |
| | `cumulative_length`: number of tokens written | int64 scalar |
| Whole model | `cursor`: position of the next token | int64 scalar |

Each name is prefixed with its layer, as in `layer.3.keys`. Other models get
whatever their `StaticCache` holds; a dense model such as Qwen3 has `keys`,
`values` and `cumulative_length` in every layer.

**Fixed capacity.** K/V tensors have `max_context_tokens` slots from the first
step. Each step writes the new token's keys and values at slot `cursor`, and
the attention mask hides the slots after it. Generation stops when the context
is full; a longer conversation needs a reset.

**One step, one transaction.** `Engine::run` works on a private copy of the
request state:

1. Each state input is read from the copy. If the request has no value yet,
   as after a reset, the engine uses the graph's zero initializer.
2. The nodes run.
3. Each state output is stored under its state name in the copy.
4. Only when every node and output check has succeeded does the copy replace
   the request state.

A failed step therefore leaves the cache exactly as it was. The copy is cheap:
`aten` nodes are functional, so a step produces new state tensors instead of
writing the old ones, and the copy can share tensor storage with the original.
Graphs that also contain DLI's own (non-`aten`) operators get a deep copy of
the named tensors, because those operators may write in place.

**Reset, failure and reuse.** `ExecutionState.reset()` drops every tensor, so
the next run starts from the initializers. `EngineCausalLM` resets the state
whenever a run fails or is interrupted, because a partial step cannot be
resumed; the reset also makes earlier decode handles stale. Between runs the
adapter records which tokens the cache holds, which is how prefill reuses a
shared prefix. `ExecutionState.copy()` returns a snapshot that later runs
cannot change, without copying tensor data, for callers that want to keep a
prefix and branch from it.

**DLI's own attention operators.** `ExecutionState` also holds a `KVCache` for
DLI's native attention operators, such as the Triton AOT kernels on the Qwen2
path; exported graphs do not use it. An append concatenates the new keys and
values along the sequence axis into new tensors and never writes existing
entries, and `RunOptions.position_offset` gives the rotary and attention
kernels the absolute position of each decode step.

## Part 3: End-to-end workflow: the chatbot

`examples/qwen3_5/` is a terminal chatbot built from the pieces above. At chat
time Transformers provides only the tokenizer and chat template: every model
forward pass runs on `dli.Engine`, and the chatbot never calls
`PreTrainedModel.generate`.

1. **Export once** with the `dli_export.causal_lm` command from Part 1.
2. **Start the chatbot:**

   ```bash
   python3 examples/qwen3_5/main.py \
     --artifacts build/examples/qwen3_5/qwen3_5.dli.bundle.json
   ```

   `load_session` reads the bundle, loads the tokenizer for the bundle's
   `model_id`, loads the graphs, places the weights on the device (`--device
   cpu`, `cuda` or `auto`), and creates one `dli.Engine`, one
   `dli.ExecutionState` and one `dli.EngineCausalLM` for the session.
3. **Chat.** `ChatSession.stream_reply` runs each turn:

   ```text
   user message
     -> append it to the history
     -> tokenizer.apply_chat_template(whole history) -> prompt token IDs
     -> generate_tokens(EngineCausalLM, prompt token IDs)
          prefill: run the step graph for each prompt token not yet in the cache
          loop:    sample a token -> stream it -> decode: run the step graph once
     -> incremental detokenizer -> print text as it arrives
     -> append the assistant reply to the history
   ```

Rendering the whole history each turn keeps the chat template's exact
formatting, and prefix reuse means only the new tokens run. If the template
renders earlier turns differently from the tokens in the cache, the prompt no
longer extends the cache, and prefill runs the whole history again.

If tokenization, inference or decoding fails, the turn's user message is
dropped from the history and the request state is reset, so the next turn
starts clean. Ctrl-C instead keeps the partial reply in the history; the cache
still matches the tokens it consumed, so the next turn can reuse it. `/reset`
clears both the history and the request state.

## Acceptance gates

- DLI registers no Qwen-specific operator, and the exporter has no
  per-architecture code: tiny Llama, Qwen3 and hybrid Qwen3.5 models export
  with the same code.
- The converter encodes every argument kind in the table above, and exported
  input mutation becomes declared request state.
- Each tiny model's bundle, run one token at a time on `dli.Engine`, matches
  the PyTorch full-sequence logits at every position.
- The bundle's prefill and decode entries share one graph and one weights
  file.
- Request states stay isolated, a failed step publishes no state, and a reset
  reproduces a clean first step.
- The chatbot tests fail if the chatbot calls `model.generate`.
- Exported Qwen3.5-0.8B and Qwen3-0.6B give the same greedy replies as
  Transformers, token for token.

## Current limits

- FP32 only. DLI tensors keep FP32, int64 and bool; checkpoints can be
  exported in their native BF16 once DLI supports it.
- Batch size one.
- Prefill runs one token per graph run, so long prompts are slow.
- Cache capacity is fixed at export time.
- Each step writes new copies of the cache tensors, because the graph is
  functional; updating them in place is a later optimization.
- Each ATen call is a separate dispatcher call, with no fusion across calls.
- Text only; vision, video and other conditional-generation inputs are out of
  scope.
