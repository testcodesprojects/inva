"""
pyinla/write/writers.py - Core file writing functions for INLA.

This module contains only the functions responsible for writing data to files:
- Binary matrix files (fmesher format)
- Graph files (binary format)
- Model.ini configuration files
- Hyperparameter specifications
"""

from __future__ import annotations

import os
import re
import math
import time
import uuid
import errno
import struct
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# SciPy is optional for sparse matrix support
try:
    import scipy.sparse as sp
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# =============================================================================
# Utility Functions
# =============================================================================

def inla_dir_create(path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def inla_tempfile(tmpdir: Optional[str] = None, suffix: str = "") -> str:
    """Create a temporary file path in the given directory."""
    if tmpdir is None:
        tmpdir = os.getcwd()
    inla_dir_create(tmpdir)
    uid = uuid.uuid4().hex
    return os.path.join(tmpdir, f"inla_tmp_{int(time.time()*1e6)}_{uid}{suffix}")


# R-compatible number formatting
_R_DIGITS: int = 7


def _format_number(val: Any) -> str:
    """Format value the same way R's cat() does."""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return str(int(val))
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        text = format(val, f".{_R_DIGITS}g")
        if text == "-0":
            return "0"
        return text
    return str(val)


# =============================================================================
# Text File Writing
# =============================================================================

def writeln(filepath: str, text: str) -> None:
    """Append a single line to a file (creating it if needed)."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(text)


def inla_write_boolean_field(tag: str, val: Optional[bool], file: str) -> None:
    """Write 'tag = 1' or 'tag = 0' if val is not None."""
    if val is None:
        return
    writeln(file, f"{tag} = {1 if bool(val) else 0}\n")


# =============================================================================
# Binary File Writing - fmesher format
# =============================================================================

def inla_write_fmesher_file(obj: Any, filename: str, debug: bool = False) -> None:
    """
    Binary writer mirroring R's inla.write.fmesher.file for dense/sparse matrices.

    Format:
    - Header: [header_len, version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype]
    - Data: column-major float64 or int32 values

    For sparse matrices (if scipy available):
    - Stores COO format: row indices, col indices, values
    """
    version = 0

    # Handle sparse matrices
    if _HAVE_SCIPY and sp.issparse(obj):
        M = obj.tocoo()
        nrow, ncol = M.shape
        datatype = 1  # sparse
        valuetype = 0 if np.issubdtype(M.data.dtype, np.integer) else 1
        matrixtype = 0
        storagetype = 1
        elems = int(M.nnz)
        header = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)

        with open(filename, "wb") as fp:
            fp.write(np.int32([len(header)]).tobytes())
            fp.write(header.tobytes())
            fp.write((M.row.astype(np.int32)).tobytes(order="C"))
            fp.write((M.col.astype(np.int32)).tobytes(order="C"))
            if valuetype == 0:
                fp.write(M.data.astype(np.int32).tobytes(order="C"))
            else:
                fp.write(M.data.astype(np.float64).tobytes(order="C"))
        return

    # Handle dense arrays
    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("inla_write_fmesher_file: only vectors/matrices are supported.")

    nrow, ncol = arr.shape
    datatype = 0  # dense
    valuetype = 0 if np.issubdtype(arr.dtype, np.integer) else 1
    matrixtype = 0
    storagetype = 1
    elems = nrow * ncol
    header = np.array([version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype], dtype=np.int32)

    with open(filename, "wb") as fp:
        fp.write(np.int32([len(header)]).tobytes())
        fp.write(header.tobytes())
        if valuetype == 0:
            fp.write(arr.astype(np.int32).T.tobytes(order="C"))
        else:
            fp.write(arr.astype(np.float64).T.tobytes(order="C"))


# =============================================================================
# Graph File Writing
# =============================================================================

def _inla_graph_binary_magic() -> int:
    """Return the magic number for binary graph files (-1)."""
    return -1


def _write_graph_binary(adj_list: Dict[int, List[int]], n: int, filename: str) -> str:
    """
    Write graph in R-INLA binary format.

    Binary format:
      int32: -1 (magic number)
      int32: n (number of nodes)
      For each node i from 1 to n:
        int32: i (node number, 1-indexed)
        int32: num_neighbors
        int32[num_neighbors]: neighbor indices (1-indexed)
    """
    with open(filename, "wb") as f:
        # Write magic number and n
        f.write(np.array([_inla_graph_binary_magic(), n], dtype=np.int32).tobytes())
        # Write each node
        for node in range(1, n + 1):
            # Preserve order from input, remove duplicates
            seen = set()
            neighbors = []
            for nb in adj_list.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    neighbors.append(nb)
            # Write: node, num_neighbors, [neighbors...]
            header = np.array([node, len(neighbors)], dtype=np.int32)
            f.write(header.tobytes())
            if neighbors:
                f.write(np.array(neighbors, dtype=np.int32).tobytes())
    return filename


def inla_write_graph(graph: Any, filename: Optional[str] = None, mode: str = "binary") -> str:
    """
    Write a graph file in INLA format.

    Accepts:
      - str: path to existing .graph file (will be copied/converted)
      - dict {node: [neighbors]} (1-indexed)
      - scipy sparse matrix (adjacency matrix, 0-indexed)
      - numpy 2D array (adjacency matrix, 0-indexed)
      - iterable of (u,v) tuples (edge list)
      - 2-column numpy array (edge list)

    Parameters
    ----------
    graph : various
        Graph specification (see above)
    filename : str, optional
        Output filename. If None, a temp file is created.
    mode : str
        'binary' (default, matches R-INLA) or 'ascii'

    Returns the filename.
    """
    import shutil

    if filename is None:
        filename = inla_tempfile()

    # If graph is a string/Path to existing file, read and convert to binary
    if isinstance(graph, (str, os.PathLike)):
        graph_path = str(graph)
        if os.path.isfile(graph_path):
            # Check if already binary
            with open(graph_path, "rb") as f:
                first_int = np.frombuffer(f.read(4), dtype=np.int32)
                if len(first_int) > 0 and first_int[0] == _inla_graph_binary_magic():
                    # Already binary, just copy
                    shutil.copy(graph_path, filename)
                    return filename

            # Text format - parse and convert to binary
            with open(graph_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            # Parse text graph format
            parts = lines[0].split()
            n = int(parts[0])
            adj_list: Dict[int, List[int]] = {i: [] for i in range(1, n + 1)}

            idx = 1 if len(parts) == 1 else 0
            while idx < len(lines):
                parts = lines[idx].split()
                if len(parts) >= 2:
                    node = int(parts[0])
                    num_nb = int(parts[1])
                    neighbors = [int(x) for x in parts[2:2+num_nb]]
                    adj_list[node] = neighbors
                idx += 1

            return _write_graph_binary(adj_list, n, filename)
        else:
            raise ValueError(f"Graph file not found: {graph_path}")

    # Handle scipy sparse matrix
    if _HAVE_SCIPY and sp.issparse(graph):
        n = graph.shape[0]
        coo = graph.tocoo()
        adj_list = {i: [] for i in range(1, n + 1)}
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if val != 0 and i != j:
                adj_list[int(i) + 1].append(int(j) + 1)  # Convert to 1-indexed
        return _write_graph_binary(adj_list, n, filename)

    # Handle dense numpy array (adjacency matrix)
    if isinstance(graph, np.ndarray) and graph.ndim == 2 and graph.shape[0] == graph.shape[1]:
        n = graph.shape[0]
        adj_list = {i: [] for i in range(1, n + 1)}
        for node in range(n):
            neighbors = [j + 1 for j in range(n) if graph[node, j] != 0 and node != j]
            adj_list[node + 1] = neighbors
        return _write_graph_binary(adj_list, n, filename)

    # Handle dict format {node: [neighbors]} - assume already 1-indexed
    if isinstance(graph, dict):
        nodes = sorted(graph.keys())
        n = max(nodes) if nodes else 0
        adj_list = {i: list(graph.get(i, [])) for i in range(1, n + 1)}
        return _write_graph_binary(adj_list, n, filename)

    # Handle edge list format
    edges: List[tuple] = []
    arr = np.asarray(list(graph))
    if arr.ndim == 1 and len(arr) == 2:
        edges.append((int(arr[0]), int(arr[1])))
    elif arr.ndim == 2 and arr.shape[1] == 2:
        for row in arr:
            edges.append((int(row[0]), int(row[1])))
    else:
        raise ValueError("Unsupported graph type for inla_write_graph.")

    # Convert edge list to adjacency list format
    if edges:
        all_nodes = set()
        for u, v in edges:
            all_nodes.add(u)
            all_nodes.add(v)
        n = max(all_nodes)
        adj_list_edges: Dict[int, List[int]] = {i: [] for i in range(1, n + 1)}
        for u, v in edges:
            adj_list_edges[u].append(v)
        return _write_graph_binary(adj_list_edges, n, filename)
    else:
        # Empty graph
        return _write_graph_binary({}, 0, filename)


# =============================================================================
# ID Names File Writing
# =============================================================================

def write_id_names_file(names: List[str], filename: str) -> None:
    """
    Write factor level names in R-INLA's binary format.

    Format: int32 count, then for each name: int32 length + raw bytes.
    """
    with open(filename, "wb") as fp:
        fp.write(struct.pack("<i", len(names)))
        for name in names:
            encoded = name.encode("utf-8")
            fp.write(struct.pack("<i", len(encoded)))
            fp.write(encoded)


# =============================================================================
# Hyperparameter Writing
# =============================================================================

def inla_write_hyper(
    hyper: Optional[List[Dict[str, Any]]],
    file: str,
    prefix: str = "",
    data_dir: Optional[str] = None,
    ngroup: int = -1,
    low: float = float("-inf"),
    high: float = float("inf")
) -> List[Dict[str, Any]]:
    """
    Write hyperparameter blocks (list of dicts) to the INI file.

    Each element expects keys:
      - initial (float), fixed (bool), hyperid (int)
      - prior (string)
      - param (sequence of floats)
      - to.theta / from.theta (callable or str)

    Returns the (possibly modified) hyper list.
    """
    if not hyper:
        return []

    if data_dir is None:
        data_dir = os.getcwd()

    for k, hk in enumerate(hyper, start=1):
        suff = "" if len(hyper) == 1 else str(k - 1)

        initial_val = hk.get('initial')
        if initial_val is not None:
            writeln(file, f"{prefix}initial{suff} = {_format_number(initial_val)}\n")

        fixed_val = hk.get('fixed')
        if fixed_val is not None:
            writeln(file, f"{prefix}fixed{suff} = {_format_number(fixed_val)}\n")

        hyperid_val = hk.get('hyperid', 0)
        writeln(file, f"{prefix}hyperid{suff} = {_format_number(hyperid_val)}\n")

        # Process prior
        tmp_prior = str(hk.get("prior", "") or "")
        tmp_prior = tmp_prior.replace("\n", "")
        tmp_prior = re.sub(r"^[ \t]+", "", tmp_prior)
        tmp_prior = re.sub(r";*[ \t]*$", "", tmp_prior)

        # Normalize common prior names
        prior_lower = tmp_prior.lower()
        prior_map = {
            "pc.prec": "pcprec",
            "pc.cor0": "pccor0",
            "pc.cor1": "pccor1",
        }
        if prior_lower in prior_map:
            tmp_prior = prior_map[prior_lower]

        if tmp_prior:
            writeln(file, f"{prefix}prior{suff} = {tmp_prior}\n")

        # Process parameters
        param = hk.get("param")
        if param is not None:
            param_arr = np.atleast_1d(np.asarray(param)).astype(float)
            param_str = " ".join(_format_number(v) for v in param_arr)
            writeln(file, f"{prefix}parameters{suff} = {param_str}\n")

        # Process transform functions
        for key in ("to.theta", "from.theta"):
            fn_val = hk.get(key)
            if fn_val is not None:
                fn_str = str(fn_val) if not callable(fn_val) else repr(fn_val)
                key_out = key.replace(".", "_")
                writeln(file, f"{prefix}{key_out}{suff} = {fn_str}\n")

    return hyper
