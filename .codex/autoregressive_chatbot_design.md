# Autoregressive Chatbot and Qwen3.5 Design

## Decision Summary

Build autoregressive inference as a stateful layer above the otherwise reusable
graph engine. A graph and its weights are shared; each conversation owns an
`ExecutionState`, its current position, and generation policy. The model runner
has two logical operations:

```text
prefill(prompt tokens, state) -> last-token logits, populated state
decode(next token, state)     -> next-token logits, updated state
```

The first working product is a text-only, batch-one chatbot using the
post-trained `Qwen/Qwen3.5-0.8B` checkpoint. The terminal example uses the
official Transformers implementation as the correctness reference and as an
immediately usable backend. Native DLI execution is delivered in slices because
Qwen3.5's hybrid architecture needs substantially more than the repository's
current Qwen2 attention graph.

The initial scope excludes vision/video inputs, sparse MoE checkpoints, tools,
multi-token prediction, continuous batching, and remote serving.

## Why State Is Separate from `Engine`

An engine contains immutable execution machinery and loaded operator plugins.
Conversation state is mutable and request-specific. Keeping it in the engine
would make two chats contaminate each other and would prevent safe scheduling.

```text
                         +---------------- shared ----------------+
Chat A -> GenerationSession A --+                                 |
                                +-> Engine -> Graph + Weights + Operators
Chat B -> GenerationSession B --+                                 |
          |                                                        |
          +-> ExecutionState B          ExecutionState A <---------+
```

`ExecutionState` has two storage classes:

- an attention K/V cache;
- named tensors for other recurrent state.

The named tensor store is required by hybrid models. It can hold Qwen3.5's
depthwise-convolution history and Gated DeltaNet recurrent matrices without
teaching the generic engine about one model architecture.

For backward compatibility, `Engine` retains a default state, but callers that
serve more than one conversation must pass an explicit state to each run.
Each `ExecutionState` is single-writer: overlapping runs must use different
states, and engine/plugin configuration must finish before concurrent inference.

## Public Runtime Contract

The implemented C++ direction is:

```cpp
class ExecutionState {
 public:
  void reset();
  KVCache& kvCache();
  const Tensor* findTensor(const std::string& name) const;
  void setTensor(std::string name, Tensor value);
};

struct RunOptions {
  ExecutionState* state = nullptr;
  std::int64_t position_offset = 0;
};

TensorMap Engine::run(const Graph& graph, TensorMap inputs);
TensorMap Engine::run(const Graph& graph, TensorMap inputs,
                      const RunOptions& options);
```

Python exposes `dli.ExecutionState`, `state.reset()`, and
`engine.run(..., state=state, position_offset=position)`. The reusable
`EngineCausalLM` adapter implements the `prefill`/`decode` protocol for existing
single-token decoder graphs, while `generate_tokens` owns greedy or sampled
token selection and stop-token handling.

## Correct Decode Semantics

Every decode step must satisfy four invariants:

1. The input contains only tokens not already represented by the state.
2. Rotary embeddings use the absolute position, not position zero on every
   invocation.
3. State is updated exactly once after a successful step.
4. Resetting or replacing state reproduces a clean first-token execution.

The runtime now forwards a dynamic `position_offset` to operators. Rotary adds
that offset to any graph-local base position and validates it against the table.
Single-token attention keeps the cache length as a runtime argument, so decode
does not require one compiled kernel for every possible context length.

The current K/V append remains a correctness-first, reallocating operation. Its
copy layout is fixed to concatenate dimension 2 independently for each
batch/head plane. Production decode should replace it with capacity growth or a
paged token-major cache; otherwise repeated append still performs O(T^2) total
copying.

## Qwen3.5 Text Architecture Contract

The 0.8B checkpoint has 24 layers arranged as six repetitions of three linear
attention layers followed by one full-attention layer. Native lowering must
cover:

- hidden size 1024, SwiGLU intermediate size 3584, vocabulary 248,320, and tied
  input/output embeddings;
- full attention with 8 query heads, 2 K/V heads, head dimension 256, per-head
  Q/K normalization, grouped-query attention, and a sigmoid output gate;
- partial RoPE over 64 of 256 head dimensions, theta 10,000,000, with correct
  absolute positions and multimodal-RoPE-compatible layout;
- linear attention with 16 Q/K and 16 value heads of dimension 128, depthwise
  causal convolution of width 4, and the Gated DeltaNet recurrence;
- zero-centered RMSNorm, `rms(x) * (1 + weight)`, rather than the existing
  Qwen2-style scale;
- BF16 parameters with FP32 normalization and recurrent accumulation.

A K/V cache alone is insufficient. Per linear-attention layer, batch-one decode
needs convolution state shaped `[1, 6144, 4]` (Q, K, and V each contribute
`16 * 128 = 2048` channels) and an FP32 recurrent matrix shaped
`[1, 16, 128, 128]`. Full-attention layers retain K/V shaped
`[1, 2, tokens, 256]`.

## Native Operator Boundary

The native implementation should use two stateful composite decode operators.
This keeps recurrent updates atomic and avoids materializing many intermediate
tensors in the graph:

`gated_attention_decode`

- Q/K normalization and projections;
- partial rotate-half RoPE;
- GQA cache append/read;
- causal attention and sigmoid output gate.

`gated_delta_decode`

- Q/K/V, Z, beta, and decay projections;
- depthwise causal-convolution state update;
- FP32 normalized gated-delta recurrence;
- gated RMSNorm and output projection.

Embedding, residual add, linear, SiLU, multiply, and the surrounding MLP can
reuse generic operators after BF16 support is added. A correctness-first native
prefill may run the decode graph token by token. A chunked DeltaNet prefill and
fused attention path are later performance work.

## Chat and Tokenization

Tokenization stays in Python. The checkpoint's `AutoTokenizer` and
`apply_chat_template` are the source of truth; the template must not be copied
or approximated in the runtime.

For the default non-thinking chatbot:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)
```

Generation stops on both `<|endoftext|>` (248044) and `<|im_end|>` (248046).
The first version re-renders and re-prefills the full message history for each
user turn. This is deliberate: the official template may remove prior thinking
content, so a newly rendered conversation is not guaranteed to be a byte-for-
byte extension of the previously cached prompt.

## Implemented Vertical Slice

This change provides:

- explicit, independently resettable `ExecutionState` objects;
- transactional state publication, so a failed graph step cannot leak a partial
  cache update;
- dynamic runtime position offsets;
- arbitrary single-token attention cache lengths in the AOT decode kernel;
- corrected multi-head cache concatenation, rotate-half RoPE, and grouped-query
  8Q/2KV head mapping;
- a backend-neutral generation loop with greedy, temperature, top-k, top-p,
  EOS, seed, and context-limit controls;
- an `EngineCausalLM` adapter for existing single-token DLI graphs;
- a Qwen3.5 terminal chatbot with multi-turn history, streamed output,
  `/reset`, `/exit`, one-shot mode, offline loading, and deterministic mode;
- offline tests that need neither a model download nor a GPU.

The terminal example intentionally runs `Qwen3_5ForCausalLM` through
Transformers. It is the usable chatbot and parity oracle while native DLI
Qwen3.5 operators are developed; it is not presented as native DLI execution.

## Delivery Sequence for Native Qwen3.5

1. Add BF16 tensor/weight support and preserve tied-weight aliases.
2. Replace reallocating K/V append with bounded preallocation or paged blocks.
3. Implement GQA, partial RoPE, Q/K norm, and gated full-attention decode.
4. Implement causal-convolution and recurrent Gated DeltaNet decode state.
5. Add model-specific lowering under `dli_export/models/qwen3_5.py` with
   explicit rejection of multimodal, MoE, and unsupported RoPE variants.
6. Compare every layer and every next-token logit with Transformers on a tiny
   four-layer `[linear, linear, linear, full]` configuration.
7. Compare cached token-by-token logits and the first N greedy tokens on the
   real 0.8B checkpoint.
8. Add chunked prefill, cache reuse, batching, and a streaming service only
   after parity is stable.

## Acceptance Gates

- Two explicit states used with one engine remain isolated.
- Reset produces the same first-step state and logits.
- Positions advance from prompt through decode and fail cleanly at the context
  limit.
- Multi-head and GQA cache contents retain the correct head/token order.
- Full-forward and cached token-by-token reference logits agree at every
  position.
- Tiny Qwen3.5 outputs agree across a DeltaNet layer and a full-attention layer.
- Both termination tokens stop streaming.
- Unsupported model variants fail during export with a precise error rather
  than silently lowering the wrong architecture.

## Reference Sources

- Qwen3.5 0.8B model card:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>
- Official checkpoint configuration:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json>
- Official chat template:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja>
- Transformers Qwen3.5 documentation:
  <https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5>
- Reference model implementation:
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py>
