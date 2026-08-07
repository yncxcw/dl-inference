# Profiling Library Design

## Decision Summary

Add an opt-in profiling session to the C++ runtime and instrument graph-node
execution in `Engine::run`. For every operator invocation, the profiler records:

- CPU submit/execution time with `std::chrono::steady_clock` around
  `Operator::compute`.
- GPU execution time with a pair of CUDA events recorded on the same stream as
  the operator, without synchronizing after each operator.
- Correlation metadata: run, iteration, graph node, operator type, device,
  stream, and optional tensor shapes.

The session writes a stable `dli.profile.v1` JSON file. A Python CLI reads that
file to print operator summaries, compare profiles, export CSV, and create a
Perfetto/Chrome trace. JSON is the source of truth; the trace is a visualization
derived from it.

This design deliberately measures operator-level GPU work. It does not use
CUPTI to attribute individual kernels in the first version.

## Goals

1. Measure CPU time spent invoking each graph operator.
2. Measure GPU time attributable to each CUDA graph operator.
3. Preserve CPU and GPU measurements as separate metrics; never add them into a
   synthetic "total" because they can overlap.
4. Keep profiling disabled by default and cheap to bypass.
5. Avoid a device synchronization after every operator.
6. Support both the native C++ engine and its Python binding.
7. Produce artifacts that are useful in scripts, CI, and an interactive trace
   viewer.

## Non-goals for the First Version

- Kernel-, memcpy-, or runtime-API attribution through CUPTI.
- Hardware counters such as SM occupancy, memory bandwidth, or cache hit rate.
- A statistically rigorous benchmark harness. The analyzer reports the samples
  it receives and warns when there are too few.
- A globally time-aligned CPU/GPU concurrency view. CUDA events provide accurate
  device durations and ordering on a stream, but not an exact host-clock start
  timestamp.
- Distributed aggregation across processes or hosts.

## Terminology and Timing Boundaries

For one graph node:

```text
host thread:  record(gpu_start) |-- Operator::compute --| record(gpu_end)
                                      ^             ^
                                      cpu_start     cpu_end

GPU stream:              ... |-- work launched by this operator --| ...
```

- `cpu_time`: wall time inside `Operator::compute`. For a CUDA operator this is
  primarily allocation, dispatch, argument preparation, and kernel submission;
  it is not GPU completion time.
- `gpu_time`: CUDA-event elapsed time between the start and end markers on the
  operator's stream.
- `run_wall_time`: host wall time around the complete `Engine::run` call. This
  is reported independently from operator sums.
- `operator_sum`: a sum over peer graph-node events only. Nested user ranges,
  if added later, must not be included because that would double-count time.

CPU and GPU values are never summed. The CLI labels them `CPU submit` and
`GPU execution` to make the distinction visible.

## Architecture

```text
Engine::run
  -> ProfileSession::beginRun
  -> for each graph node
       -> ProfileSession::beginOperator
            records CUDA start event, then host start timestamp
       -> Operator::compute(..., ExecutionContext)
       -> ProfileSession::endOperator
            records host end timestamp, then CUDA end event
  -> ProfileSession::endRun

ProfileSession::finalize
  -> wait for pending terminal CUDA events
  -> resolve CUDA event pairs to durations
  -> write dli.profile.v1 JSON

python -m dli.profile summarize/compare/export/trace
  -> validate dli.profile.v1
  -> aggregate peer operator samples
  -> table, JSON, CSV, or Perfetto/Chrome trace
```

### Components

`ProfileSession`

- Owns configuration, event storage, string interning, CUDA event handles, and
  run IDs.
- Is explicitly finalized. Its destructor releases resources but does not write
  files or throw.
- Supports repeated `Engine::run` calls so percentiles have multiple samples.
- Is internally mutex-protected. Each run still belongs to one host thread.

`ProfileScope`

- Move-only RAII object returned by `beginOperator`.
- Ends the CPU interval and records the GPU end marker in its destructor.
- Records an error status during stack unwinding so a failed node still has a
  useful CPU sample. A failed span has no GPU duration unless its end event was
  safely recorded.

`CudaEventPool`

- Reuses timing-enabled CUDA events across runs.
- Keeps event pairs alive until `finalize` resolves them.
- Is implemented through the project's dynamically loaded CUDA runtime layer,
  preserving the current no-hard-`libcudart` dependency behavior.

`dli.profile` analyzer

- Pure Python and dependent only on the standard library in the first version.
- Treats the JSON schema as a public input contract rather than importing the
  native runtime.

## Public C++ API

Proposed declarations:

```cpp
namespace dli {

struct ProfileConfig {
  bool cpu = true;
  bool gpu = true;
  bool record_shapes = false;
  std::size_t max_events = 1'000'000;
};

class ProfileSession {
 public:
  explicit ProfileSession(ProfileConfig config = {});

  void finalize();
  void writeJson(const std::string& path) const;
  const ProfileReport& report() const;
};

struct RunOptions {
  ProfileSession* profiler = nullptr;  // non-owning; valid through run()
};

class Engine {
 public:
  TensorMap run(const Graph& graph, TensorMap inputs,
                const RunOptions& options = {});
};

}  // namespace dli
```

Example:

```cpp
dli::ProfileSession profile({.cpu = true, .gpu = true});
for (int i = 0; i < 10; ++i) {
  outputs = engine.run(graph, inputs, {.profiler = &profile});
}
profile.finalize();
profile.writeJson("qwen2-profile.json");
```

`finalize()` is the only profiling operation that may wait for outstanding GPU
work. Calling `report()` or `writeJson()` before finalization is an error. A
second `finalize()` is harmless. Starting another run after finalization is an
error.

### Operator Extension API

Graph-node instrumentation requires no changes to an operator. Later, an
operator may add nested CPU-only annotations through the profiler pointer in
`ExecutionContext`:

```cpp
struct ExecutionContext {
  KVCache* kv_cache = nullptr;
  ProfileSession* profiler = nullptr;
  void* cuda_stream = nullptr;
};
```

Nested ranges use an explicit category and parent ID. They are visible in the
trace but excluded from the default graph-operator totals.

## CUDA Stream Contract

Correct CUDA timing requires both event markers and operator work to use the
same stream. The current AOT templates call `CudaAotKernel::launch` without a
stream, while ATen follows its current CUDA stream. Before enabling GPU timing,
the runtime must establish one stream contract:

1. Resolve the current ATen stream for the run's CUDA device.
2. Store its raw handle in `ExecutionContext::cuda_stream`.
3. Pass that handle from every AOT operator template to
   `CudaAotKernel::launch`.
4. Record profiler CUDA events on that exact handle.
5. Guard ATen dispatch so it uses the same stream for the duration of the run.

The first version supports one CUDA device per run and records its device ID.
It rejects a graph invocation spanning multiple CUDA devices with profiling
enabled. Multi-device support can later maintain one stream/event timeline per
device.

An operator is considered CUDA-backed when any input is a CUDA tensor. This
matches the current AOT operator contract. After `compute`, the profiler also
checks outputs; a CPU-input operator that unexpectedly creates a CUDA output is
marked `gpu_unmeasured` because no valid start event exists.

### Why CUDA Events

CUDA events measure elapsed device time in the stream and naturally include all
work enqueued between the two markers. Event pairs are resolved after the run,
so normal execution remains asynchronous. In contrast, synchronizing each node
would substantially perturb small operators and hide pipeline behavior.

CUDA-event timing includes kernels and asynchronous copies enqueued by the
operator on the measured stream. It does not see work sent to an undisclosed
secondary stream. Operators that introduce internal streams must register those
streams with the profiling scope or declare their GPU time incomplete.

## Python API

Expose `ProfileConfig`, `ProfileSession`, and the optional `profile` argument on
`Engine.run`:

```python
profile = dli.ProfileSession(gpu=True, record_shapes=True)
for _ in range(10):
    outputs = engine.run(graph, inputs, profile=profile)
profile.finalize()
profile.write_json("qwen2-profile.json")
```

The Python binding continues to release the GIL during `Engine::run` and also
releases it while `finalize()` waits for CUDA events.

## Trace Schema

The source artifact is one JSON object. Durations and timestamps are integers
in nanoseconds to avoid unit ambiguity and floating-point loss.

```json
{
  "schema": "dli.profile.v1",
  "clock": "steady_clock",
  "metadata": {
    "created_utc": "2026-08-05T20:00:00Z",
    "process_id": 1234,
    "graph_fingerprint": "sha256:...",
    "runtime_version": "...",
    "cuda_driver_version": "...",
    "cuda_timing": true,
    "record_shapes": false,
    "dropped_events": 0
  },
  "runs": [
    {
      "run_id": 0,
      "start_ns": 0,
      "duration_ns": 93000,
      "status": "ok"
    }
  ],
  "events": [
    {
      "event_id": 1,
      "run_id": 0,
      "parent_id": null,
      "category": "graph_operator",
      "node_index": 3,
      "node_name": "layers.0.self_attn.q_proj",
      "op_type": "linear",
      "host_thread_id": 7,
      "cpu_start_ns": 12000,
      "cpu_duration_ns": 7400,
      "gpu_duration_ns": 18100,
      "device": 0,
      "stream": "0x...",
      "status": "ok"
    }
  ]
}
```

Optional shape metadata contains dtype, device, dimensions, and byte count, but
never tensor values. Shape recording is off by default because dynamic shapes
can enlarge traces and may reveal request characteristics.

Unknown fields must be ignored by readers. Missing required fields or an
unknown major schema version are errors.

## Analyzer Utility

The CLI lives in `python/dli/profile.py` and runs as:

```bash
PYTHONPATH=python python3 -m dli.profile summarize profile.json
PYTHONPATH=python python3 -m dli.profile summarize profile.json --group-by op
PYTHONPATH=python python3 -m dli.profile summarize profile.json --skip-runs 2
PYTHONPATH=python python3 -m dli.profile compare before.json after.json
PYTHONPATH=python python3 -m dli.profile export profile.json --format csv -o profile.csv
PYTHONPATH=python python3 -m dli.profile trace profile.json -o trace.json
```

### `summarize`

Default grouping is `(node_name, op_type)`. It reports:

- call count;
- CPU submit total, mean, p50, p95, and maximum;
- GPU execution total, mean, p50, p95, and maximum;
- percent of peer operator CPU time and percent of peer operator GPU time;
- run wall-time mean and p50/p95 when multiple runs exist;
- unmeasured, failed, and dropped event counts.

Percentiles use nearest-rank semantics and the output states that choice. The
table sorts by GPU total when GPU samples exist, otherwise by CPU total.
`--skip-runs N` excludes warmup runs without deleting them from the source
artifact.

### `compare`

Joins rows by node name and operator type and reports absolute and percentage
changes for median CPU and GPU time. It marks added/removed nodes and refuses to
claim a regression when device metadata, shape signatures, or sample counts are
incompatible unless `--force` is supplied.

### `trace`

Writes Chrome Trace Event Format accepted by Perfetto. CPU spans use actual host
timestamps. GPU spans are placed on one synthetic track per `(device, stream)`
in CUDA-event order. The trace metadata explicitly labels their host alignment
as approximate; GPU durations remain accurate.

## Overhead and Failure Behavior

- Disabled path: one null-pointer branch per graph node.
- CPU-only path: two steady-clock reads and one buffered event append.
- GPU path: two CUDA event records per CUDA graph node; no per-node wait.
- Event storage is preallocated in chunks. At `max_events`, the profiler stops
  recording new events, increments `dropped_events`, and lets inference proceed.
- CUDA profiling initialization failure disables only GPU timing, records the
  error in metadata, and preserves CPU profiling.
- File I/O happens only in `writeJson`, never in the inference loop.
- Profiler errors must not replace an active inference exception.

## Integration with the Current Repository

Proposed files:

```text
include/dli/profiler.h              public config/session/report API
src/profiler.cc                     host event collection and serialization
include/dli/cuda_runtime.h          CUDA event wrapper declarations
src/cuda_runtime.cc                 dynamically loaded CUDA event symbols
include/dli/operator.h              profiler and stream in ExecutionContext
include/dli/engine.h                RunOptions overload
src/engine.cc                       graph-node scopes and run scope
src/python_bindings.cc              Python profile bindings
python/dli/profile.py               analysis CLI
tests/cpp/test_profiler.cc          lifecycle, CPU spans, failures, limits
tests/python/test_dli_profile.py    schema, aggregation, compare, trace export
```

Every AOT `template.cc` must pass `context.cuda_stream` to `launch`. The AOT
generator should also provide a shared launch helper so future templates do not
silently omit the stream.

## Validation Plan

1. **CPU deterministic test:** use a test operator with a controllable clock or
   injectable timestamp source; verify node identity, parentage, failure status,
   and run boundaries without relying on sleep-based assertions.
2. **Disabled-path test:** verify an unprofiled run creates no session state and
   preserves the existing `Engine::run(graph, inputs)` API.
3. **Schema round trip:** write JSON, parse it with the Python analyzer, and
   compare every required field.
4. **Aggregation test:** use fixed fixture durations to verify totals,
   nearest-rank percentiles, grouping, and that nested ranges are excluded.
5. **Limit/error tests:** exhaust `max_events`, simulate unavailable CUDA, and
   make an operator throw; inference behavior must remain correct.
6. **GPU test:** on a GPU runner, profile a known-duration CUDA workload and
   compare CUDA-event duration with an independently timed event pair using a
   tolerant bound.
7. **Stream test:** run on a non-default ATen stream and verify AOT work and both
   profiler markers use that stream.
8. **Overhead benchmark:** compare profiling off, CPU-only, and CPU+GPU over at
   least 1,000 small operator calls. Record results rather than imposing an
   arbitrary CI threshold initially.

## Delivery Sequence

1. Implement the event model, CPU scopes, JSON writer, and analyzer fixtures.
2. Establish the explicit CUDA stream contract across Engine, ATen, and AOT
   templates.
3. Add asynchronous CUDA-event collection and GPU-only tests.
4. Add Python bindings and CLI summarize/export commands.
5. Add compare and Perfetto trace conversion.
6. Measure overhead, document usage in `README.md`, and then consider enabling a
   self-hosted GPU CI test.

Keeping CPU profiling usable before GPU support lands gives each stage an
independently testable result, while the stream contract prevents publishing
misleading GPU numbers.

## Future Extensions

- NVTX ranges for correlation in Nsight Systems.
- CUPTI activity collection for kernel-level attribution and exact host/device
  clock correlation.
- Memory allocation/high-watermark events.
- Distributed profile merging keyed by host, process, rank, and request ID.
- Sampling and name filters for very large graphs.
- A benchmark runner that manages warmup, repetitions, and input-shape labels.
