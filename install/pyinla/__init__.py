"""
pyinla package entry point.

Usage:
    import pyinla

    pyinla()                    # Run INLA model
    pyinla.activate("CODE")     # Activate license
    pyinla.tmarginal(...)       # Transform marginal
    pyinla.download_binary()    # Download binary

Version: 0.1.3
"""

import sys as _sys
from types import ModuleType as _ModuleType

# Import all functions
from ._api import pyinla as _pyinla_func, activate as _activate
from ._api import PyINLAError, PyINLACrashError, PyINLACollectError
from ._api import PyINLAResult
from .surv import inla_surv
from .pure.marginal_utils import (
    inla_smarginal as _smarginal,
    inla_dmarginal as _dmarginal,
    inla_pmarginal as _pmarginal,
    inla_qmarginal as _qmarginal,
    inla_emarginal as _emarginal,
    inla_rmarginal as _rmarginal,
    inla_tmarginal as _tmarginal,
    inla_mmarginal as _mmarginal,
    inla_zmarginal as _zmarginal,
    inla_hpdmarginal as _hpdmarginal,
)
from .binary import (
    download_binary as _download_binary,
    list_available_binaries as _list_available_binaries,
    list_available_os as _list_available_os,
    is_binary_installed as _is_binary_installed,
)
from .sampling import (
    inla_qsample as _qsample,
    inla_posterior_sample as _posterior_sample,
    inla_posterior_sample_eval as _posterior_sample_eval,
    inla_hyperpar_sample as _hyperpar_sample,
)
from .sections import set_r_digits as _set_r_digits, get_r_digits as _get_r_digits


def _options(**kwargs):
    """Get or set pyINLA global options (mirrors R's ``options()``).

    Currently supported options:
        digits (int): Number of significant digits for numeric output in
            Model.ini.  Default is 7, matching R's ``options(digits = 7)``.

    Examples::

        pyinla.options(digits=4)   # match R's options(digits = 4)
        pyinla.options()           # return current options dict
    """
    if not kwargs:
        return {"digits": _get_r_digits()}
    for key, value in kwargs.items():
        if key == "digits":
            _set_r_digits(int(value))
        else:
            raise ValueError(f"Unknown option: {key!r}")
    return {"digits": _get_r_digits()}


class _CallableModule(_ModuleType):
    """A module that can be called directly: pyinla() instead of pyinla.pyinla()"""

    def __call__(self, *args, **kwargs):
        """Call pyinla() directly."""
        return _pyinla_func(*args, **kwargs)

    # Core functions
    activate = staticmethod(_activate)
    pyinla = staticmethod(_pyinla_func)  # Also available as pyinla.pyinla()

    # Marginal functions
    smarginal = staticmethod(_smarginal)
    dmarginal = staticmethod(_dmarginal)
    pmarginal = staticmethod(_pmarginal)
    qmarginal = staticmethod(_qmarginal)
    emarginal = staticmethod(_emarginal)
    rmarginal = staticmethod(_rmarginal)
    tmarginal = staticmethod(_tmarginal)
    mmarginal = staticmethod(_mmarginal)
    zmarginal = staticmethod(_zmarginal)
    hpdmarginal = staticmethod(_hpdmarginal)

    # Binary management
    download_binary = staticmethod(_download_binary)
    list_available_binaries = staticmethod(_list_available_binaries)
    list_available_os = staticmethod(_list_available_os)
    is_binary_installed = staticmethod(_is_binary_installed)

    # Survival
    inla_surv = staticmethod(inla_surv)

    # Sampling functions (matching R-INLA)
    qsample = staticmethod(_qsample)
    posterior_sample = staticmethod(_posterior_sample)
    posterior_sample_eval = staticmethod(_posterior_sample_eval)
    hyperpar_sample = staticmethod(_hyperpar_sample)

    # Options (mirrors R's options())
    options = staticmethod(_options)

    # Exceptions (as class attributes)
    PyINLAError = PyINLAError
    PyINLACrashError = PyINLACrashError
    PyINLACollectError = PyINLACollectError
    PyINLAResult = PyINLAResult


# Replace this module with callable version
_old_module = _sys.modules[__name__]
_new_module = _CallableModule(__name__)
_new_module.__dict__.update(_old_module.__dict__)
# Ensure options() is in the module namespace (not just a class attribute)
_new_module.options = _options
_sys.modules[__name__] = _new_module

__all__ = [
    "pyinla",
    "activate",
    "inla_surv",
    "PyINLAError",
    "PyINLACrashError",
    "PyINLACollectError",
    "PyINLAResult",
    # Marginal utility functions
    "smarginal",
    "dmarginal",
    "pmarginal",
    "qmarginal",
    "emarginal",
    "rmarginal",
    "tmarginal",
    "mmarginal",
    "zmarginal",
    "hpdmarginal",
    # Sampling functions (matching R-INLA)
    "qsample",
    "posterior_sample",
    "posterior_sample_eval",
    "hyperpar_sample",
    # Options
    "options",
    # Binary management
    "download_binary",
    "list_available_binaries",
    "list_available_os",
    "is_binary_installed",
]
