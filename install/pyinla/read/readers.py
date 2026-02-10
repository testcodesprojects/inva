"""
pyinla/read/readers.py - Core file reading functions for INLA.

This module contains only the functions responsible for reading data from files:
- Binary matrix files (fmesher format)
- Raw binary float64/int32 streams
- Text files
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# SciPy is optional for sparse matrix support
try:
    from scipy.sparse import coo_matrix as scipy_coo
    _HAS_SCIPY = True
except ImportError:
    scipy_coo = None
    _HAS_SCIPY = False

# Canonical storage used by INLA/GMRFLib exports
_F8_LE = "<f8"


# =============================================================================
# Low-level Binary Reading
# =============================================================================

def read_bytes(f, nbytes: int) -> bytes:
    """Read exactly nbytes or raise EOFError."""
    b = f.read(nbytes)
    if len(b) != nbytes:
        raise EOFError("Unexpected EOF")
    return b


def read_int32s(f, count: int) -> np.ndarray:
    """Read count int32 values from file."""
    if count <= 0:
        return np.empty((0,), dtype=np.int32)
    b = read_bytes(f, 4 * count)
    return np.frombuffer(b, dtype="<i4")


def read_float64s(f, count: int) -> np.ndarray:
    """Read count float64 values from file."""
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    b = read_bytes(f, 8 * count)
    return np.frombuffer(b, dtype="<f8")


# =============================================================================
# Text File Reading
# =============================================================================

def read_lines(path: str) -> Optional[List[str]]:
    """Read all lines from a text file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f]


# =============================================================================
# INLA Binary File Reading
# =============================================================================

def inla_read_binary_file(path: str | Path) -> Optional[np.ndarray]:
    """
    Read an INLA *.dat style file (little-endian float64 stream).

    Parameters
    ----------
    path:
        File-system location of the binary file.

    Returns
    -------
    ndarray or None
        1-D float64 array with the raw contents, None when the file is
        missing, and an empty array when the file exists but has no payload.
    """
    p = Path(path)
    if not p.exists():
        return None
    data = p.read_bytes()
    if not data:
        return np.empty((0,), dtype=np.float64)
    # frombuffer keeps a reference to data; copy so caller can mutate
    return np.frombuffer(data, dtype=_F8_LE).copy()


def read_float64_auto(path: str) -> Optional[np.ndarray]:
    """
    Best-effort reader for INLA/GMRFLib .dat files:
    - If the file looks like [int32 n][n float64] EXACTLY, return the n doubles.
    - Otherwise, treat the whole file as a plain sequence of float64.
    """
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size == 0:
        return np.array([], dtype=np.float64)

    with open(path, "rb") as f:
        # Try length-prefixed format: 4 + n*8 bytes
        if size >= 12:  # at least one int + one double
            pos = f.tell()
            try:
                hdr = read_bytes(f, 4)
                n = struct.unpack("<i", hdr)[0]
            except Exception:
                n = -1
            rest = size - 4
            if 0 <= n <= (rest // 8) and rest == n * 8:
                try:
                    data = read_float64s(f, n)
                    if data.size == n:
                        return data
                except Exception:
                    pass
            f.seek(pos)

        # Fallback: whole file as doubles
        try:
            b = f.read()
            return np.frombuffer(b, dtype="<f8")
        except Exception:
            return None


def read_numeric_text_or_binary(path: str) -> Optional[np.ndarray]:
    """
    Read numeric data from either text or binary file.
    Tries UTF-8 text first, then falls back to binary.
    """
    if not os.path.exists(path):
        return None
    # Try UTF-8 text first (R's scan() behavior)
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            txt = f.read().strip()
        if txt:
            return np.array([float(tok) for tok in txt.split()], dtype=float)
    except Exception:
        pass
    # Fallback to binary reader
    return read_float64_auto(path)


# =============================================================================
# fmesher Format Reading
# =============================================================================

def read_fmesher(path: str):
    """
    Best-effort GMRFLib/INLA 'fmesher' dense/sparse reader (binary).

    Format:
    - Header: [header_len, version, elems, nrow, ncol, datatype, valuetype, matrixtype, storagetype]
    - Data: column-major values (dense) or COO triplets (sparse)

    Returns:
    - Dense: numpy array of shape (nrow, ncol)
    - Sparse: scipy COO matrix (if scipy available) or dict with rows, cols, data, shape
    - None: if file doesn't exist or can't be parsed
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            # Header length (int32)
            hb = read_bytes(f, 4)
            H = struct.unpack("<i", hb)[0]

            # Read H int32 in header
            hdr = read_bytes(f, 4 * H)
            ints = np.frombuffer(hdr, dtype="<i4")
            if ints.size < 6:
                return None

            elems = ints[1]
            nrow = ints[2]
            ncol = ints[3]
            dtype_flag = ints[4]
            valtype_flag = ints[5]
            val_dtype = "<i4" if valtype_flag == 0 else "<f8"

            if dtype_flag == 0:  # dense
                b = read_bytes(f, elems * (4 if valtype_flag == 0 else 8))
                data = np.frombuffer(b, dtype=val_dtype)
                if data.size != elems:
                    return None
                return data.reshape((nrow, ncol), order="F")
            else:  # COO sparse
                ib = read_bytes(f, elems * 4)
                jb = read_bytes(f, elems * 4)
                vb = read_bytes(f, elems * (4 if valtype_flag == 0 else 8))
                i = np.frombuffer(ib, dtype="<i4")
                j = np.frombuffer(jb, dtype="<i4")
                v = np.frombuffer(vb, dtype=val_dtype)
                if i.size != elems or j.size != elems or v.size != elems:
                    return None
                if _HAS_SCIPY:
                    return scipy_coo((v, (i, j)), shape=(nrow, ncol))
                return {"rows": i, "cols": j, "data": v, "shape": (nrow, ncol)}
    except Exception:
        return None


# =============================================================================
# Vector Interpretation (INLA marginal format)
# =============================================================================

def _split_vector_blocks(vec: Sequence[float]) -> List[Tuple[float, np.ndarray]]:
    """
    Interpret the flattened [idx, n, x1, y1, ...] representation that INLA
    uses for quantiles, modes, and marginal densities.
    """
    arr = np.asarray(vec, dtype=np.float64).ravel()
    blocks: List[Tuple[float, np.ndarray]] = []
    i = 0
    n = arr.size
    while i + 1 < n:
        idx = arr[i]
        count = int(round(arr[i + 1]))
        i += 2
        if count <= 0:
            blocks.append((idx, np.zeros((0, 2), dtype=np.float64)))
            continue
        needed = 2 * count
        chunk = arr[i: i + needed]
        if chunk.size < needed:
            # Truncated file - pad with NaNs
            padding = np.full((needed - chunk.size,), np.nan, dtype=np.float64)
            chunk = np.concatenate([chunk, padding])
            i = n
        else:
            i += needed
        blocks.append((idx, chunk.reshape(count, 2)))
    return blocks


def inla_interpret_vector(vec: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """
    Decode the INLA 'vector' format into a dense matrix.

    The returned array has shape (max_count, 1 + 2 * n_blocks):
      - column 0 holds the abscissa (probabilities, grid points, ...) from the
        first block when available;
      - for each subsequent block, column 1 + 2*k stores the y-values and
        column 1 + 2*k + 1 stores the corresponding x-values. Missing
        entries are padded with NaN.
    """
    if vec is None:
        return None
    blocks = _split_vector_blocks(vec)
    if not blocks:
        return np.empty((0, 0), dtype=np.float64)

    if len(blocks) == 1:
        # Common case for fixed effects / single marginals: keep native (N x 2)
        return blocks[0][1].copy()

    max_rows = max(block[1].shape[0] for block in blocks)
    n_blocks = len(blocks)

    if max_rows == 0:
        return np.zeros((0, 1 + 2 * n_blocks), dtype=np.float64)

    out = np.full((max_rows, 1 + 2 * n_blocks), np.nan, dtype=np.float64)

    first_pairs = blocks[0][1]
    if first_pairs.size:
        out[: first_pairs.shape[0], 0] = first_pairs[:, 0]

    for k, (_, pairs) in enumerate(blocks):
        if not pairs.size:
            continue
        rows = pairs.shape[0]
        out[:rows, 1 + 2 * k] = pairs[:, 1]      # y-values
        out[:rows, 1 + 2 * k + 1] = pairs[:, 0]  # x-values for reference
    return out


def inla_interpret_vector_list(
    vec: Optional[Sequence[float]],
) -> Optional[List[np.ndarray]]:
    """
    Decode a 'vector-of-vectors' marginal structure into a list of (n x 2) arrays.
    Each array contains the (x, y) pairs for one marginal.
    """
    if vec is None:
        return None
    blocks = _split_vector_blocks(vec)
    if not blocks:
        return None
    return [pairs.copy() for _, pairs in blocks]
