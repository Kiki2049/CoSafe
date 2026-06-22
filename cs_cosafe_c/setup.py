"""Cython build for cs_cosafe_c kernels.

Build in-place (run from CoLMDriver/, not from cs_cosafe_c/):
    cd CoLMDriver
    python cs_cosafe_c/setup.py build_ext --inplace

This compiles all .pyx files under cs_cosafe_c/kernels/ into platform-
specific shared objects (.so on Linux), placed back into cs_cosafe_c/
kernels/. Each kernel is independent — a failure to build one does not
prevent the others from being usable.

Set CS_COSAFE_DEBUG=1 to enable boundscheck/wraparound for debugging.
"""
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")
PARENT = os.path.dirname(HERE)
os.chdir(PARENT)

DEBUG = os.environ.get("CS_COSAFE_DEBUG") == "1"

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": True if DEBUG else False,
    "wraparound": True if DEBUG else False,
    "cdivision": True,
    "initializedcheck": False,
    "embedsignature": True,
}

EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-march=native"]
if DEBUG:
    EXTRA_COMPILE_ARGS = ["-O0", "-g"]

extensions = []
if os.path.isdir(KDIR):
    for fname in sorted(os.listdir(KDIR)):
        if not fname.endswith(".pyx"):
            continue
        modname = "cs_cosafe_c.kernels." + os.path.splitext(fname)[0]
        extensions.append(
            Extension(
                modname,
                sources=[os.path.join("cs_cosafe_c", "kernels", fname)],
                include_dirs=[np.get_include()],
                extra_compile_args=EXTRA_COMPILE_ARGS,
                language="c",
            )
        )

if not extensions:
    print("[cs_cosafe_c.setup] No .pyx files found under kernels/; "
          "nothing to build.")

setup(
    name="cs_cosafe_c.kernels",
    ext_modules=cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        annotate=False,
    ),
    zip_safe=False,
)
