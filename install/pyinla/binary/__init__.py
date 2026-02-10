"""Binary download and path management for pyINLA.

This subpackage handles:
- Downloading INLA binaries on-demand (manager.py)
- Locating binary paths for different platforms (call.py)
"""

from .manager import (
    BinaryManager,
    download_binary,
    list_available_binaries,
    list_available_os,
    is_binary_installed,
    ensure_binary,
    _cli_install,
)

from .call import (
    inla_call_builtin,
    inla_call_no_remote,
    fmesher_call_builtin,
    inla_remote_script,
)

__all__ = [
    # Manager
    "BinaryManager",
    "download_binary",
    "list_available_binaries",
    "list_available_os",
    "is_binary_installed",
    "ensure_binary",
    "_cli_install",
    # Call
    "inla_call_builtin",
    "inla_call_no_remote",
    "fmesher_call_builtin",
    "inla_remote_script",
]
