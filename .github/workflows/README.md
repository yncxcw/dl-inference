# Continuous integration

`ci.yml` defines one GitHub Actions workflow, **CI**. It runs on every pull request, on every push
to `main`, and on demand from the Actions tab (`workflow_dispatch`). A new run cancels any
unfinished run for the same branch or pull request. The workflow can only read the repository
(`permissions: contents: read`).

## Jobs

The three jobs run in parallel:

| Job | Runs on | What it does |
| --- | --- | --- |
| Formatting | GitHub-hosted `ubuntu-24.04` | Checks formatting |
| Build (Python 3.14) | GitHub-hosted `ubuntu-24.04`, no GPU | Builds everything and the Python wheel; runs no tests |
| GPU tests (CUDA 13.2) | The self-hosted GPU runner, in a CUDA 13.2 container | Builds everything, runs every test, and tests the installed wheel |

### Formatting

Installs Python 3.14, Black 24.10.0 and clang-format 18.1.5, then runs `./format --check`.

### Build (Python 3.14)

1. Installs Python 3.14, the CPU-only PyTorch 2.15.0, NumPy 2.5.4, Triton 3.9.0, Transformers
   5.22.1, `build` and `scikit-build-core`.
2. Configures `build-ci` with the tests, examples, Python bindings and Triton AOT operators (gRPC
   off), and builds it. The runner has no GPU, so `tools/build_triton_aot_ops.py` compiles the
   Triton kernels for its default target, `sm_80`, using the `ptxas` that ships in the Triton
   wheel; it needs neither the CUDA toolkit nor the driver.
3. Builds the wheel with `python -m build --wheel --no-isolation`, against the installed PyTorch.
   `pyproject.toml` turns the tests and Triton AOT off for wheel builds.
4. Uploads the wheel as the `dl-inference-python-3.14` artifact. Download it from the run's
   summary page, under Artifacts; GitHub keeps it for 90 days.

### GPU tests (CUDA 13.2)

Runs on the self-hosted runner (labels `self-hosted, linux, x64, gpu`), inside the
`nvidia/cuda:13.2.0-devel-ubuntu24.04` container with every GPU attached. GPU jobs never overlap:
they share the `gpu-runner` concurrency group, and a new one waits for the running one instead of
cancelling it. GitHub keeps only the newest waiting job, so an older job that is still waiting is
cancelled. The job is skipped for pull requests from forks, because the runner executes the
workflow's commands on a shared host.

1. Installs the build tools and the container's Python 3.12, then a virtual environment in
   `/opt/venv` with the CUDA 13.2 build of PyTorch 2.15.0, NumPy 2.5.4, Triton 3.9.0,
   Transformers 5.22.1, `build` and `scikit-build-core`. It fails if PyTorch cannot see the GPU.
2. Configures `build-gpu` with Triton AOT on, and builds it.
3. Runs every test with `ctest --test-dir build-gpu --output-on-failure --parallel 1
   --no-tests=error`: one test at a time, because the tests share the GPU, and an error if no
   tests are registered.
4. Builds a wheel and tests it: installs it into `/opt/venv`, then, from an empty folder, imports
   `dli`, `dli_export` and `dli_ops` and runs a one-node `aten::relu` graph. This job builds its
   own wheel because its Python is 3.12, while the build job's wheel is for Python 3.14.

`tools/ci/setup_gpu_runner.sh` sets a machine up as the GPU runner: it configures the NVIDIA
container runtime for Docker, creates a `gh-runner` account in the `docker` group, and installs and
starts the runner service. Its header lists the registration token it needs and its options.
Membership in the `docker` group is root-equivalent on that host, which is why pull requests from
forks skip this job.

## Merging

The ruleset on `main` requires a pull request, the three jobs to pass, and the branch to be up to
date with `main`; it also blocks force-pushes to `main` and deleting it. GitHub counts a skipped
job as passed, so a pull request from a fork can merge without the GPU tests. To run them, push
its branch to this repository.

## Running the checks locally

`AGENTS.md` lists the checks to run before pushing: `./format --check`, `./build.sh`, and
`ctest --test-dir build --output-on-failure`. On a machine with a GPU, these cover what the GPU job
runs. To reproduce the build job's GPU-free build, install the CPU-only PyTorch and hide CUDA:

```bash
cmake -S . -B build-ci -DCMAKE_BUILD_TYPE=Release -DDLI_BUILD_TESTS=ON \
  -DDLI_BUILD_EXAMPLES=ON -DDLI_ENABLE_TRITON_AOT=ON -DDLI_ENABLE_GRPC=OFF \
  -DDLI_BUILD_PYTHON_BINDINGS=ON -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON
CUDA_VISIBLE_DEVICES= cmake --build build-ci --parallel 2
```

## Changing versions

`ci.yml` pins the versions of Python, PyTorch, NumPy, Triton, Transformers and the formatters.
Keep the two build jobs on the same PyTorch, NumPy, Triton and Transformers versions, so that both
build the same stack.
