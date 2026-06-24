#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget


OPERATOR_NAMES = (
    "embedding",
    "rms_norm",
    "linear",
    "matmul",
    "add",
    "mul",
    "relu",
    "silu",
    "max_pool2d",
    "softmax",
    "reshape",
    "transpose",
    "conv2d",
    "attention",
    "rotary",
)


@dataclass(frozen=True)
class AotSpec:
    id: str
    module: str
    kernel_name: str
    signature: tuple[str, ...]
    num_warps: int = 4
    num_stages: int = 3


@dataclass(frozen=True)
class CompiledSpec:
    spec: AotSpec
    symbol: str
    cubin: bytes
    shared: int


def constexpr(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return None


def hash_signature(signature: tuple[str, ...], num_warps: int, num_stages: int) -> str:
    text = " ".join(signature) + f" warps{num_warps} stages{num_stages}"
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_kernel_module(operator_dir: Path):
    path = operator_dir / "kernel.py"
    if not path.exists():
        raise FileNotFoundError(f"missing Triton kernel module: {path}")
    sys.path.insert(0, str(operator_dir))
    spec = importlib.util.spec_from_file_location(f"dli_aot_{operator_dir.name}_kernel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compile_spec(modules: dict[str, object], spec: AotSpec, arch: int) -> CompiledSpec:
    kernel = getattr(modules[spec.module], spec.kernel_name)
    signature_items = list(spec.signature)
    hints = {(i,): constexpr(s.split(":")[1]) for i, s in enumerate(signature_items) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature_items)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature_items)}
    for key in constants:
        signature[key] = "constexpr"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    ccinfo = triton.compile(
        src,
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": spec.num_warps, "num_stages": spec.num_stages},
    )
    scratch = int(getattr(ccinfo.metadata, "global_scratch_size", 0))
    if scratch > 0:
        raise RuntimeError(f"{spec.id}: global scratch AOT kernels are not supported")
    suffix = hash_signature(spec.signature, spec.num_warps, spec.num_stages)
    return CompiledSpec(
        spec=spec,
        symbol=f"{spec.id}_{suffix}",
        cubin=ccinfo.asm["cubin"],
        shared=int(ccinfo.metadata.shared),
    )


def default_specs() -> list[AotSpec]:
    specs: list[AotSpec] = []
    for hidden in (2, 4, 8, 128, 4096):
        block_d = max(16, 1 << (hidden - 1).bit_length())
        specs.append(AotSpec(f"embedding_h{hidden}", "embedding", "embedding_kernel",
                             ("*i64", "*fp32", "*fp32", "i32", str(hidden), "16", str(block_d))))
        specs.append(AotSpec(f"rms_norm_h{hidden}", "rms_norm", "rms_norm_kernel",
                             ("*fp32", "*fp32", "*fp32", "i32", "fp32", str(hidden), str(block_d))))
    for k in (2, 4, 8, 64, 128, 1024, 4096):
        for has_bias in (0, 1):
            specs.append(AotSpec(f"linear_k{k}_b{has_bias}", "linear", "linear_kernel",
                                 ("*fp32", "*fp32", "*fp32", "*fp32", "i32", "i32",
                                  str(k), str(has_bias), "16", "16", "32")))
        specs.append(AotSpec(f"matmul_k{k}", "matmul", "matmul_kernel",
                             ("*fp32", "*fp32", "*fp32", "i32", "i32", str(k), "16", "16", "32")))
    specs.append(AotSpec("add", "add", "add_kernel", ("*fp32", "*fp32", "*fp32", "i32", "i32", "256")))
    specs.append(AotSpec("mul", "mul", "mul_kernel", ("*fp32", "*fp32", "*fp32", "i32", "i32", "256")))
    specs.append(AotSpec("silu", "silu", "silu_kernel", ("*fp32", "*fp32", "i32", "256")))
    specs.append(AotSpec("relu", "relu", "relu_kernel", ("*fp32", "*fp32", "i32", "256")))
    for block in (8, 16, 32, 64, 128, 256, 512):
        specs.append(AotSpec(f"softmax_b{block}", "softmax", "softmax_kernel",
                             ("*fp32", "*fp32", "i32", "i32", str(block))))
    specs.append(AotSpec("transpose2d", "transpose", "transpose2d_kernel",
                         ("*fp32", "*fp32", "i32", "i32", "16", "16")))
    for in_c, kh, kw in ((1, 2, 2), (1, 3, 3), (3, 3, 3), (8, 3, 3), (16, 3, 3)):
        for has_bias in (0, 1):
            specs.append(AotSpec(f"conv_c{in_c}_k{kh}x{kw}_b{has_bias}", "conv2d", "conv2d_kernel",
                                 ("*fp32", "*fp32", "*fp32", "*fp32", "i32", "i32", "i32",
                                  "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
                                  str(in_c), str(kh), str(kw), str(has_bias), "128")))
    for kh, kw in ((2, 2), (3, 3)):
        specs.append(AotSpec(f"maxpool_k{kh}x{kw}", "max_pool2d", "max_pool2d_kernel",
                             ("*fp32", "*fp32", "i32", "i32", "i32", "i32", "i32",
                              "i32", "i32", "i32", "i32", "i32", "i32",
                              str(kh), str(kw), "128")))
    attention_shapes = [
        (1, 1, 2), (1, 2, 2), (2, 2, 2),
        (1, 1, 128), (1, 128, 128), (1, 512, 128), (1, 2048, 128), (128, 128, 128),
    ]
    for seq_q, seq_k, head_dim in attention_shapes:
        for causal in (0, 1):
            scale = 1.0 / (head_dim ** 0.5)
            block_d = max(16, 1 << (head_dim - 1).bit_length())
            specs.append(AotSpec(f"attention_q{seq_q}_k{seq_k}_d{head_dim}_c{causal}", "attention",
                                 "attention_kernel",
                                 ("*fp32", "*fp32", "*fp32", "*fp32", str(seq_q), str(seq_k),
                                  str(head_dim), str(causal), repr(scale), "16", "32", str(block_d))))
    for head_dim, seq, pairs in ((2, 1, 1), (2, 2, 1), (128, 1, 64), (128, 128, 64)):
        specs.append(AotSpec(f"rotary_d{head_dim}_s{seq}_p{pairs}", "rotary", "rotary_kernel",
                             ("*fp32", "*fp32", "*fp32", "*fp32", "*fp32", "*fp32",
                              "i32", "i32", str(head_dim), str(seq), str(pairs), "128")))
    return specs


def c_array(data: bytes) -> str:
    hexed = binascii.hexlify(data).decode()
    return ", ".join(f"0x{hexed[i:i + 2]}" for i in range(0, len(hexed), 2))


def emit_plugin(compiled: list[CompiledSpec], kernel_root: Path,
                operator_names: tuple[str, ...]) -> str:
    arrays = []
    replacements = {}
    for item in compiled:
        suffix = item.symbol[len(item.spec.id) + 1:]
        replacements[f"{{{{HASH_{item.spec.id}}}}}"] = suffix
        arrays.append(
            f"static const unsigned char {item.symbol}_cubin[] = {{ {c_array(item.cubin)} }};\n"
            f"static dli::CudaAotKernel& {item.symbol}_kernel() {{\n"
            f"  static dli::CudaAotKernel kernel(\"{item.spec.kernel_name}\", {item.symbol}_cubin,"
            f" sizeof({item.symbol}_cubin), {item.shared});\n"
            f"  return kernel;\n"
            f"}}\n"
        )

    templates = []
    for operator_name in operator_names:
        template_path = kernel_root / operator_name / "template.cc"
        if not template_path.exists():
            raise FileNotFoundError(f"missing AOT operator template: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        for needle, value in replacements.items():
            text = text.replace(needle, value)
        if "{{HASH_" in text:
            raise RuntimeError(f"unresolved kernel hash placeholder in {template_path}")
        templates.append(f"// {operator_name}\n" + text)

    registrations = "\n".join(f"  register_{operator_name}(registry);" for operator_name in operator_names)
    return (CPP_TEMPLATE
            .replace("/*__AOT_KERNELS__*/", "\n".join(arrays))
            .replace("/*__OPERATOR_TEMPLATES__*/", "\n\n".join(templates))
            .replace("/*__REGISTER_OPERATORS__*/", registrations))


CPP_TEMPLATE = r'''
#include "dli/cuda_driver.h"
#include "dli/cuda_runtime.h"
#include "dli/operator.h"
#include "dli/utils.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dli::attrDouble;
using dli::attrInts;
using dli::ceilDiv;
using dli::product;
using dli::ptr;
using dli::requireCudaInputs;
using dli::requireFloat;
using dli::reshapeShape;

/*__AOT_KERNELS__*/

/*__OPERATOR_TEMPLATES__*/

}  // namespace

extern "C" bool dli_register_operators(dli::OperatorRegistry* registry) {
/*__REGISTER_OPERATORS__*/
  return true;
}
'''


def compile_shared(source: Path, output: Path, include_dir: Path, core_library_dir: Path,
                   extra_include_dirs: list[Path]) -> None:
    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx,
        "-std=c++20",
        "-fPIC",
        "-shared",
        f"-I{include_dir}",
        *(f"-I{path}" for path in extra_include_dirs),
        str(source),
        f"-L{core_library_dir}",
        "-ldli_core",
        "-Wl,-rpath,$ORIGIN/..",
        "-o",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--core-library-dir", type=Path)
    parser.add_argument("--include-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--library-name", default="dli_triton_aot_ops")
    parser.add_argument("--kernel-source", type=Path, default=Path("python/dli_ops/aot"))
    parser.add_argument("--extra-include-dir", type=Path, action="append", default=[])
    parser.add_argument("--arch", type=int, default=int(os.environ.get("DLI_AOT_ARCH", "80")))
    args = parser.parse_args()

    specs = default_specs()
    if args.list:
        for spec in specs:
            print(f"{spec.id}: {spec.module}.{spec.kernel_name} {spec.signature}")
        return 0

    if args.core_library_dir is None or args.include_dir is None or args.output_dir is None:
        parser.error("--core-library-dir, --include-dir, and --output-dir are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules = {name: load_kernel_module(args.kernel_source / name) for name in OPERATOR_NAMES}
    compiled = [compile_spec(modules, spec, args.arch) for spec in specs]
    source = args.output_dir / f"{args.library_name}.cc"
    source.write_text(emit_plugin(compiled, args.kernel_source, OPERATOR_NAMES), encoding="utf-8")
    output = args.output_dir / f"lib{args.library_name}.so"
    compile_shared(source, output, args.include_dir, args.core_library_dir, args.extra_include_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
