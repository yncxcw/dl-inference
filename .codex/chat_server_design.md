# OpenAI-Compatible Chat Server Design

## Decision

Serve exported DLI causal-LM bundles behind an OpenAI-compatible HTTP API, in
the style of `vllm serve`. Three clients drive it: a dependency-free Python
client with an OpenAI-shaped interface, a browser chat UI served by the same
process, and any OpenAI SDK pointed at its `base_url`.

```text
browser UI  /  dli_client  /  openai SDK
                 |  HTTP + SSE
                 v
dli_serve.api      FastAPI app on uvicorn (asyncio event loop)
                 |  submit / per-request asyncio.Queue
                 v
dli_serve.scheduler  one worker thread, iteration-level round robin
                 |                         |
                 v                         v
dli_serve.runner (one-token step)    dli_serve.kv_cache (prefix snapshots)
                 |
                 v
dli.Engine.run(graph, inputs, state=<per-sequence ExecutionState>)
```

The default model is `Qwen/Qwen3.5-0.8B`. The server only depends on the
`dli.autoregressive.v1` bundle contract plus a Hugging Face tokenizer, and the
generic exporter (`python -m dli_export.causal_lm`) produces that bundle for
any Hugging Face causal LM, so dense Qwen3 serves the same way (see Model
support).

## Scope

In scope for the first implementation:

- `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
  `GET /health`, `GET /stats`, and the chat UI at `GET /`.
- Streaming (server-sent events) and non-streaming responses with OpenAI
  object shapes, error bodies, and `usage`.
- KV reuse within a request, across chat turns, and across requests.
- Concurrent requests through iteration-level scheduling of batch-1 graphs.
- `dli_client`: `Client(...).chat.completions.create(...)`,
  `completions.create(...)`, `models.list()`, with `stream=True` iterators.

Out of scope, with the follow-up that addresses each: `n > 1`, `logprobs`,
tools and multimodal input (rejected with HTTP 400); batched kernels and
multi-token prefill graphs (milestones M2 and M4); BF16; paged KV.

## Runtime facts the design relies on

- Exported Qwen3.5 graphs process exactly one token per `Engine.run` with
  batch 1 in FP32. Prefill feeds the prompt token by token through the same
  graph.
- Attention K/V for the 6 full-attention layers lives in static
  `[1, kv_heads, max_context_tokens, head_dim]` state tensors. The 18
  linear-attention layers carry fixed-size conv and recurrent state. An
  in-graph `cursor` tensor holds the position. All of it is request state in a
  `dli.ExecutionState`.
- `Engine.run` never writes a published state tensor in place: each step
  publishes replacement tensors and commits only after the whole graph
  succeeds. A shallow copy of an `ExecutionState` is therefore an isolated,
  O(number of tensors) snapshot. This change binds it to Python as
  `ExecutionState.copy()`.
- `Engine.run` releases the GIL, so a worker thread can step the model while
  the asyncio loop serves HTTP.
- Per-sequence state for Qwen3.5-0.8B is about 20 MiB fixed plus 24 KiB per
  context position, roughly 116 MiB at `max_context_tokens=4096`. A step
  briefly holds about twice that until the old state is released.

## KV cache

The cache works at three levels.

1. **Within a request.** Each active sequence owns an `ExecutionState`. Every
   step consumes one token and appends its K/V (and advances the recurrent
   state), so no token is recomputed. Capacity is the exported
   `max_context_tokens`; reaching it ends generation with
   `finish_reason="length"`.
2. **Across turns and requests: prefix snapshots.** `kv_cache.PrefixCache`
   stores `(token_ids, ExecutionState snapshot, last-token logits)` entries.
   A new request starts from the longest entry whose tokens are a prefix of
   its prompt, then prefills only the remaining tokens. An exact match reuses
   the stored logits and skips prefill entirely. Snapshots are needed instead
   of truncation because Qwen3.5's recurrent linear-attention state cannot be
   rewound; a dense attention model could additionally rewind its cursor.
3. **Bounded memory.** The cache holds at most `--prefix-cache-size` entries
   with LRU eviction, and at most `--max-num-seqs` sequences are active. Both
   are reported by `/stats`.

Snapshot points for a chat request:

- **History boundary**: the prompt rendered without the generation prompt
  (`add_generation_prompt=False`). Chat templates re-render earlier assistant
  turns (Qwen drops `<think>` blocks, for example), so the tokens a model
  generated are not always a prefix of the next prompt. The history boundary
  is, which guarantees the next turn reuses everything up to the end of the
  previous user message.
- **End of prompt**, which makes an identical retry or regenerate free.
- **End of generation** (prompt plus consumed reply tokens), which serves
  raw-completion continuations and templates that re-render replies exactly.

Each request reports reuse as `usage.prompt_tokens_details.cached_tokens`,
the field OpenAI uses for prompt caching.

## Scheduling

One worker thread owns the engine. Each loop iteration:

1. Admits queued requests while fewer than `--max-num-seqs` are active,
   looking each prompt up in the prefix cache.
2. Picks the next active sequence round robin and runs one step: either the
   next prompt token (prefill) or the last sampled token (decode).
3. After the last prompt token, and after each decode step, samples the next
   token with `dli.generation.sample_token` under the request's
   `GenerationConfig` and seeded generator.
4. Detokenizes incrementally, holds back text that could start a stop string,
   and pushes deltas to the request's `asyncio.Queue` with
   `loop.call_soon_threadsafe`.
5. Finishes on a stop token, stop string, `max_tokens`, or context capacity,
   records snapshots, and frees the sequence.

Interleaving batch-1 steps does not raise throughput, but every concurrent
client streams progressively instead of waiting behind another reply. A client
disconnect or a stopped UI generation cancels its sequence at the next step
boundary. Engine errors fail only the affected request; its state is
discarded.

## HTTP API

Supported request fields follow OpenAI: `model`, `messages` (string or
text-part content), `prompt` (string or token IDs, completions only),
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `seed`,
`stop`, `stream`, `stream_options.include_usage`, `user`. vLLM-compatible
extensions: `top_k` and `chat_template_kwargs` (for example
`{"enable_thinking": true}`). `n > 1`, `logprobs`, `tools`, and non-text
content parts return HTTP 400 in OpenAI's error format. `--api-key` enables
bearer authentication on `/v1/*`.

Streaming chat responses send a role delta, content deltas, a final chunk
with `finish_reason`, an optional usage chunk, and `data: [DONE]`.

## Client and UI

`dli_client` uses only the standard library so it installs anywhere. Response
objects allow attribute access (`resp.choices[0].message.content`) and
`to_dict()`. Errors raise `APIStatusError` (with `status_code`) or
`APIConnectionError`. The official `openai` package works unchanged against
the same server; a test keeps that true.

The UI is static HTML, CSS and JavaScript with no build step, served from
`dli_serve/static`. It streams with `fetch` and SSE parsing, renders text
safely (escaped, with code blocks), supports stop, new chat, system prompt,
sampling settings and thinking mode, keeps the conversation in
`localStorage`, and shows tokens, tokens per second and cached prompt tokens
for each reply.

## Model support

`python -m dli_export.causal_lm --model-id ID --output-dir D
--max-context-tokens N` exports any Hugging Face causal LM whose cache works as
a Transformers `StaticCache`, with no model-specific code (see
`.codex/exporter_design.md`). `python -m dli_serve --artifacts
D/<stem>.dli.bundle.json` serves the result. Both models below match
Transformers token for token on greedy chat replies:

- **Qwen3.5-0.8B (default):** hybrid linear and full attention; 13.8 ms per
  step (about 71 tokens/s); about 116 MiB of state per sequence at 4096
  positions.
- **Qwen3-0.6B:** dense; 17.2 ms per step (about 52 tokens/s). FP32 K/V costs
  224 KiB per position (448 MiB per sequence at 2048), about nine times
  Qwen3.5-0.8B, so `--max-num-seqs` and `--prefix-cache-size` must shrink
  accordingly. Rewinding the cursor would give dense models arbitrary-prefix
  reuse without snapshots.

The exporter downloads only model files, so the server fetches the tokenizer
at startup; with `--local-files-only` and no cached tokenizer it can load one
without a chat template. Saving the tokenizer into the bundle directory would
make bundles self-contained.

## Known limits and performance work

- Prefill runs one token per graph launch, so long uncached prompts are slow;
  the prefix cache keeps multi-turn chats from re-prefilling history.
- `Engine::run` formats `LOG_INFO` strings for every node even when the log
  level hides them, and converts the full weight dict on every call. Both are
  per-token overheads to remove first.
- Static K/V buffers are rewritten whole each step (out-of-place updates), an
  O(capacity) copy per token.

## Measured (M1)

RTX 4090, Qwen3.5-0.8B exported in FP32 with `max_context_tokens=4096`:

- Export takes 36 s and writes a 4.1 GB weights file; the server is ready
  10 s after launch.
- A model step takes 13.5 to 13.8 ms, so one sequence decodes at about 70
  tokens/s, and uncached prefill costs the same per token.
- First token: 0.57 s for a 24-token cold prompt; 0.42 s for the second
  turn (17 of 49 prompt tokens cached); 0.01 s for an exact repeat (all
  cached, no prefill).
- Two concurrent streams interleave and each runs at roughly half speed,
  as expected for batch-1 steps.
- Greedy replies match Transformers token for token on the prompts tried,
  including turns served from cached snapshots.

Two findings shape the next steps:

- Qwen3.5's template re-renders earlier assistant turns without the empty
  `<think>` block the generation prompt contains, so the end-of-generation
  snapshot never matches the next turn; only the history snapshot does.
  **Idle-time warming** fixes this: after a reply, prefill the re-rendered
  reply onto the history snapshot in the background, so the next turn
  resumes after the whole previous exchange.
- `Qwen/Qwen3.5-0.8B` ships no `generation_config.json`, so API callers that
  omit sampling parameters get OpenAI defaults (temperature 1.0). The UI sends
  0.7 / 0.8. Flags for server-side defaults would serve API callers too.

## Milestones

1. **M1 (this change):** `ExecutionState.copy()`, `dli_serve` (API,
   scheduler, prefix cache, runner, CLI, UI), `dli_client`, tests with a fake
   runner, and an end-to-end run with Qwen3.5-0.8B on a GPU.
2. **M2:** engine overheads above, then a multi-token prefill graph.
3. **M3:** cursor-rewind prefix reuse for dense models, and self-contained
   bundles that include the tokenizer. (Dense Qwen3 itself already exports
   through the generic exporter.)
4. **M4:** batch > 1 decode graphs for throughput, then BF16.

## Testing

- Unit: prefix cache lookup and eviction, scheduler interleaving, stop
  strings, cancellation, and error isolation with a deterministic fake runner.
- API: FastAPI `TestClient` for every endpoint, SSE framing, error bodies,
  and authentication.
- Client: `dli_client` and the official `openai` SDK against a live uvicorn
  server backed by the fake runner.
- End to end (GPU, opt-in): export Qwen3.5-0.8B, serve it, chat through the
  client, and confirm cached-token reuse on the second turn.

Server tests skip when FastAPI is not installed, like the Qwen tests skip
without Transformers.
