"""CFFI build script for CPU ops C extension.

Run this script to compile the C extension via cffi:
    python picotorch/build_cpu.py

This reads the C header at csrc/ops_cpu.h and compiles ops_cpu.c into
a cffi-importable module (_ops_cpu_cffi).
"""

import os
from cffi import FFI

ffi = FFI()

_dir = os.path.dirname(os.path.abspath(__file__))
_csrc_dir = os.path.join(_dir, "csrc")

# Read the header (strip the include guard / non-declaration lines)
header_path = os.path.join(_csrc_dir, "ops_cpu.h")
with open(header_path, "r") as f:
    header = f.read()

# Strip preprocessor directives that cffi cannot handle
clean_lines = []
for line in header.splitlines():
    stripped = line.strip()
    if stripped.startswith("#"):
        continue
    clean_lines.append(line)
cdef_text = "\n".join(clean_lines)

ffi.cdef(cdef_text)

source_path = os.path.join(_csrc_dir, "ops_cpu.c")
with open(source_path, "r") as f:
    c_source = f.read()

ffi.set_source(
    "picotorch._ops_cpu_cffi",
    c_source,
    include_dirs=[_csrc_dir],
    extra_compile_args=["-O2"],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
