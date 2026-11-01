# Rules for AI agents

These rules apply to every AI coding agent (Claude Code, Codex, ...) that changes this
repository. Follow them for every future pull request.

## 1. Follow the coding guides

### Python

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Every module, class, function and method has a Google-style docstring (`Args:`, `Returns:`,
  `Raises:`, `Attributes:` as applicable). Test functions need a one-line docstring saying what
  they verify.
- Every function signature has type hints (`-> None` included). Annotate non-obvious attributes.
- Comment the *why* of non-obvious logic, not the *what*.
- Format with Black 24.10, line length 100 (`./format`).

### C++

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
  (`.clang-format`: Google base, 100 columns).
- Every header and source starts with a Doxygen `@file` / `@brief` block. Every class, struct,
  enum, public function and non-trivial member in a header has a Doxygen comment with `@brief`,
  `@param`, `@return` and `@throws` where applicable. Comment non-obvious logic in `.cc` files.
- `python/dli_ops/aot/*/template.cc` are code-generation templates: keep their placeholders
  intact (they are excluded from clang-format).
- Format with clang-format 18 (`./format`).

New or changed code must meet these rules; do not merge undocumented code "to be fixed later".

## 2. Test locally before pushing, and show the proof

Do not push a branch or open/update a pull request until every local check below passes. Then
include the evidence in the PR description (and in your final message to the user).

1. Formatting: `./format --check`
2. Build: `./build.sh` (on a machine without a GPU use
   `DLI_ENABLE_TRITON_AOT=OFF DLI_BUILD_DIR=build-cpu ./build.sh`)
3. Tests: `ctest --test-dir <build-dir> --output-on-failure`. CTest runs the C++ unit tests,
   the Python tests (`tests/python`, each with the right `PYTHONPATH`), the operator tests and
   the end-to-end model tests. Do not run bare `pytest` on `tests/python`; it lacks the
   `PYTHONPATH` that CTest sets up.

"Proof" means the actual command lines and their pass summaries (for example the `ctest` line
`100% tests passed, 0 tests failed out of N`) pasted under a
**Test plan** heading in the PR description. If a check cannot run locally (for example, no
GPU), say exactly which one and why; never claim a check passed that you did not run, and never
push with a failing check.
