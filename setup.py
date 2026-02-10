"""
Cython build setup for pyINLA.

This compiles Python modules to C extensions (.so files) to protect source code.
The compiled wheels contain only binary code - no readable Python source.

Usage:
    # Local build (for testing)
    pip install cython
    python setup.py build_ext --inplace

    # Build wheel
    pip install build cython
    python -m build --wheel
"""

import os
import sys
from pathlib import Path

# Must import setuptools before Cython
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Check if Cython is available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import Extension
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None
    Extension = None


# Package source directory
PACKAGE_DIR = Path(__file__).parent / "install" / "pyinla"

# Python files to compile (exclude __init__.py - they must stay as .py for imports)
# Paths are relative to PACKAGE_DIR, use / for subpackages
COMPILE_MODULES = [
    # --- Root modules ---
    "_api.py",
    "_safety.py",
    "sections.py",
    "models.py",
    "collect.py",
    "create_data_file.py",
    "control_defaults.py",
    "binary_io.py",
    "inla_call.py",
    "marginal_utils.py",
    "options.py",
    "os.py",
    "pyinla_report.py",
    "qinv.py",
    "read_graph.py",
    "rprior.py",
    "scale_model.py",
    "sm.py",
    "surv.py",
    "utils.py",
    "fmesher_io.py",
    "pc_bym.py",
    "sampling.py",
    # --- binary/ ---
    "binary/manager.py",
    "binary/call.py",
    # --- safety/ ---
    "safety/capture.py",
    "safety/control.py",
    "safety/errors.py",
    "safety/exposure.py",
    "safety/expression.py",
    "safety/family.py",
    "safety/family_variant.py",
    "safety/hyperstructure.py",
    "safety/random.py",
    "safety/response.py",
    "safety/support.py",
    "safety/utils.py",
    # --- model_defs/ ---
    "model_defs/likelihood.py",
    "model_defs/other_sections.py",
    "model_defs/transforms.py",
    # --- model_defs/latent/ ---
    "model_defs/latent/autoregressive.py",
    "model_defs/latent/basic.py",
    "model_defs/latent/copy.py",
    "model_defs/latent/generic.py",
    "model_defs/latent/measurement_error.py",
    "model_defs/latent/misc.py",
    "model_defs/latent/multidim.py",
    "model_defs/latent/random_walk.py",
    "model_defs/latent/spatial.py",
    "model_defs/latent/spde.py",
    # --- pure/ ---
    "pure/marginal_utils.py",
    # --- read/ ---
    "read/readers.py",
    # --- write/ ---
    "write/writers.py",
    # --- fmesher/ ---
    "fmesher/core.py",
    "fmesher/example.py",
    "fmesher/exceptions.py",
    "fmesher/install.py",
    "fmesher/mesh.py",
    "fmesher/spde.py",
]


def get_extensions():
    """Create Cython extension modules."""
    if not CYTHON_AVAILABLE:
        print("WARNING: Cython not available. Building pure Python package.")
        return []

    extensions = []
    for module_file in COMPILE_MODULES:
        module_path = PACKAGE_DIR / module_file
        if module_path.exists():
            # Convert path to dotted module name:
            #   _api.py -> pyinla._api
            #   safety/capture.py -> pyinla.safety.capture
            module_name = "pyinla." + module_file[:-3].replace("/", ".")
            relative_path = f"install/pyinla/{module_file}"
            extensions.append(
                Extension(
                    module_name,
                    [relative_path],
                    # Compiler directives for better performance and security
                    cython_directives={
                        'language_level': '3',
                        'embedsignature': False,  # Don't embed signatures (more secure)
                    }
                )
            )

    return cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'embedsignature': False,
        },
        # Don't generate .html annotation files
        annotate=False,
    )


class BuildExtCommand(build_ext):
    """Custom build_ext that removes .py source files after compilation."""

    def run(self):
        super().run()
        # Remove .py source files for compiled modules from build directory
        # This ensures the wheel only contains .so files, not readable source
        if self.build_lib:
            build_pyinla = Path(self.build_lib) / "pyinla"
            if build_pyinla.exists():
                for module_file in COMPILE_MODULES:
                    py_file = build_pyinla / module_file
                    if py_file.exists():
                        print(f"Removing source file: {py_file}")
                        py_file.unlink()


# Custom build_py that excludes compiled modules
from setuptools.command.build_py import build_py

class BuildPyCommand(build_py):
    """Custom build_py that excludes source files for compiled modules."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        # Build a set of (package, module_name) pairs from COMPILE_MODULES
        # e.g. "safety/capture.py" -> ("pyinla.safety", "capture")
        #      "_api.py"           -> ("pyinla", "_api")
        compiled_set = set()
        for m in COMPILE_MODULES:
            parts = m[:-3].replace("/", ".").rsplit(".", 1)
            if len(parts) == 1:
                compiled_set.add(("pyinla", parts[0]))
            else:
                compiled_set.add((f"pyinla.{parts[0]}", parts[1]))
        modules = [
            (pkg, mod, file)
            for pkg, mod, file in modules
            if (pkg, mod) not in compiled_set
        ]
        return modules


# Only use extensions if Cython is available
ext_modules = get_extensions() if CYTHON_AVAILABLE else []

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtCommand,
        'build_py': BuildPyCommand,
    } if ext_modules else {},
)
