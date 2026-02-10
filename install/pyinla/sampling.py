"""
Posterior sampling functions for pyINLA.

Implements:
- inla_qsample: Sample from GMRF with precision matrix Q (calls INLA binary)
- inla_posterior_sample: Joint posterior sampling from fitted model
- inla_posterior_sample_eval: Evaluate function over posterior samples
- inla_hyperpar_sample: Sample from hyperparameter joint posterior

These functions match R-INLA behavior exactly by calling the same INLA binary.
"""

import os
import re
import tempfile
import subprocess
import shutil
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy.interpolate import interp1d
from typing import Any, Callable, Dict, List, Optional, Union
import warnings


# Module-level RNG state, matching R-INLA's GMRFLib.rng.state in inlaEnv.
# Saved after every inla_qsample call; restored when seed < 0.
_GMRFLib_rng_state: Optional[bytes] = None

# Valid reordering algorithms (matches R-INLA's inla.reorderings.list())
_VALID_REORDERINGS = {
    "auto", "default", "identity", "band", "metis", "genmmd",
    "amd", "amdbar", "md", "mmd", "amdc", "amdbarc", "reverseidentity",
}

# Valid SMTP (sparse matrix library) options
_VALID_SMTP = {"taucs", "band", "default", "pardiso"}


def _parse_num_threads(num_threads) -> str:
    """
    Normalize num.threads string, matching R-INLA's inla.parse.num.threads().

    Rules:
      - None        -> global option
      - ""          -> global option
      - ":"         -> global option
      - "N"         -> "N:1"   (single integer → serial inner)
      - "A:"        -> "A:1"
      - ":B"        -> "<ncpus>:B"
      - "A:B"       -> "A:B"   (already normalised)
    """
    from .options import inla_get_option

    if num_threads is None:
        return inla_get_option("num.threads")

    s = str(num_threads).replace("L", "").strip()
    # normalise separators
    s = re.sub(r"[\s:,]+", ":", s)

    if len(s) == 0 or s == ":":
        return inla_get_option("num.threads")

    # single integer "N" -> "N:1"
    try:
        n = int(s)
        return f"{max(0, n)}:1"
    except ValueError:
        pass

    # "A:" -> "A:1"
    if re.fullmatch(r"[0-9]+:", s):
        return s + "1"

    # ":B" -> "<ncpus>:B"
    if re.fullmatch(r":[0-9]+", s):
        return f"{os.cpu_count() or 1}{s}"

    return s


def _set_environment() -> None:
    """
    Match R-INLA's inla.set.environment(): set INLA-specific env vars
    before calling the binary.
    """
    from .options import inla_get_option

    pkg_root = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(pkg_root, "bin")

    import platform
    sysname = platform.system().lower()
    if sysname == "darwin":
        os_type = "mac"
    elif sysname == "linux":
        os_type = "linux"
    elif sysname.startswith("win"):
        os_type = "windows"
    else:
        os_type = sysname

    os.environ["INLA_PATH"] = bin_dir
    os.environ["INLA_OS"] = os_type
    os.environ["INLA_MALLOC_LIB"] = str(inla_get_option("malloc.lib") or "mi")


def _run_environment_set() -> Dict[str, str]:
    """
    Match R-INLA's inla.run.environment.set(): save and then override
    OMP / memory-allocator environment variables before calling the binary.
    """
    env_vars = [
        "OMP_NUM_THREADS",
        "OMP_SCHEDULE",
        "OMP_MAX_ACTIVE_LEVELS",
        "MIMALLOC_ARENA_EAGER_COMMIT",
        "MIMALLOC_PURGE_DELAY",
        "MIMALLOC_PURGE_DECOMMITS",
        "MIMALLOC_SHOW_STATS",
        "MIMALLOC_VERBOSE",
        "MIMALLOC_SHOW_ERRORS",
        "MALLOC_CONF",
        "TSAN_OPTIONS",
    ]
    saved = {v: os.environ.get(v, "") for v in env_vars}

    # unset all
    for v in env_vars:
        os.environ.pop(v, None)

    # set INLA defaults (matching R-INLA exactly)
    os.environ["MIMALLOC_ARENA_EAGER_COMMIT"] = "1"
    os.environ["MIMALLOC_PURGE_DELAY"] = "-1"
    os.environ["MIMALLOC_PURGE_DECOMMITS"] = "0"
    os.environ["MIMALLOC_SHOW_STATS"] = "0"
    os.environ["MIMALLOC_VERBOSE"] = "0"
    os.environ["MIMALLOC_SHOW_ERRORS"] = "0"
    os.environ["MALLOC_CONF"] = (
        "abort_conf:true,metadata_thp:always,"
        "dirty_decay_ms:-1,percpu_arena:percpu"
    )
    os.environ["TSAN_OPTIONS"] = "ignore_noninstrumented_modules=1"

    return saved


def _run_environment_unset(saved: Dict[str, str]) -> None:
    """
    Match R-INLA's inla.run.environment.unset(): restore saved env vars.
    """
    for var, val in saved.items():
        if val:
            os.environ[var] = val
        else:
            os.environ.pop(var, None)


def inla_qsample(
    n: int = 1,
    Q: Union[np.ndarray, sparse.spmatrix, str] = None,
    b: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    sample: Optional[np.ndarray] = None,
    constr: Optional[Dict[str, Any]] = None,
    reordering: str = "auto",
    seed: int = 0,
    logdens: Optional[bool] = None,
    compute_mean: Optional[bool] = None,
    num_threads: Optional[str] = None,
    selection: Optional[np.ndarray] = None,
    verbose: Optional[bool] = None,
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Generate samples from a GMRF using the INLA binary (GMRFLib).

    Matches R-INLA's inla.qsample() function EXACTLY by calling the same binary.

    Parameters
    ----------
    n : int
        Number of samples to generate
    Q : array or sparse matrix
        Precision matrix (symmetric positive definite)
    b : array, optional
        Linear term in log-density
    mu : array, optional
        Mean parameter
    sample : array, optional
        If provided, compute log-density for these samples instead of generating new ones
    constr : dict, optional
        Linear constraints with keys 'A' (matrix) and 'e' (vector)
        Constraint: A @ x = e
    reordering : str
        Reordering algorithm: 'auto', 'amd', 'metis', 'band', 'identity', etc.
    seed : int
        Random seed (0 for random, >0 for specific seed)
    logdens : bool
        If True, also compute log-density of each sample
    compute_mean : bool
        If True, also compute the (constrained) mean
    num_threads : str
        Number of threads in format 'A:B' (outer:inner parallelism)
    selection : array, optional
        Indices of elements to return from each sample (1-based like R)
    verbose : bool
        Print debug information

    Returns
    -------
    samples : array or dict
        If logdens=False and compute_mean=False: matrix where each column is a sample
        Otherwise: dict with 'sample', 'logdens', and 'mean' keys

    Notes
    -----
    The log-density has form: -1/2 (x-mu)^T Q (x-mu) + b^T x
    """
    from .binary.call import inla_call_no_remote
    from .fmesher_io import write_fmesher_file, read_fmesher_file
    from .options import inla_get_option

    global _GMRFLib_rng_state

    if Q is None:
        raise ValueError("Q (precision matrix) is required")
    if n < 1:
        raise ValueError("n must be >= 1")

    # Match R-INLA: logdens/compute_mean default to True when sample is provided,
    # False otherwise.  Unlike the old code, explicit user values are respected.
    if logdens is None:
        logdens = sample is not None
    if compute_mean is None:
        compute_mean = sample is not None

    # Match R-INLA: verbose defaults to global option
    if verbose is None:
        verbose = bool(inla_get_option("verbose"))

    # Match R-INLA: selection and sample cannot be used together
    if sample is not None and selection is not None:
        raise ValueError("Cannot use 'selection' and 'sample' at the same time")

    # Match R-INLA: validate smtp option
    smtp = inla_get_option("smtp") or "default"
    if smtp not in _VALID_SMTP:
        raise ValueError(
            f"Invalid smtp option '{smtp}'. Must be one of: {', '.join(sorted(_VALID_SMTP))}"
        )

    # Match R-INLA: reordering can be a dict (output of inla.qreordering)
    if isinstance(reordering, dict):
        reordering = reordering.get("name", "auto")

    # Match R-INLA: validate reordering
    if reordering.lower() not in _VALID_REORDERINGS:
        raise ValueError(
            f"Invalid reordering '{reordering}'. Must be one of: {', '.join(sorted(_VALID_REORDERINGS))}"
        )
    reordering = reordering.lower()

    # Match R-INLA: resolve NULL to global option (R lines 115-117)
    if num_threads is None:
        num_threads = inla_get_option("num.threads")

    # Match R-INLA: parse num_threads and handle seed!=0 (R lines 118-126)
    if seed != 0:
        num_threads_user = _parse_num_threads(num_threads)
        num_threads = _parse_num_threads("1:1")
        if num_threads != num_threads_user:
            warnings.warn(
                "Since 'seed!=0', parallel mode is disabled and serial mode is selected",
                stacklevel=2,
            )
    else:
        num_threads = _parse_num_threads(num_threads)

    inla_binary = inla_call_no_remote()

    # Create temp directory
    t_dir = tempfile.mkdtemp(prefix="inla_qsample_")

    try:
        # Match R-INLA: Q can be a filename string or a matrix
        if isinstance(Q, str):
            Q_file = Q
            # Need to read Q to get dimension
            Q_data = read_fmesher_file(Q)
            if sparse.issparse(Q_data):
                dim = Q_data.shape[0]
            else:
                dim = Q_data.shape[0]
        else:
            if not sparse.issparse(Q):
                Q_sparse = sparse.csc_matrix(Q)
            else:
                Q_sparse = Q.tocsc()
            dim = Q_sparse.shape[0]
            Q_file = os.path.join(t_dir, "Q.dat")
            write_fmesher_file(Q_sparse, filename=Q_file)

        # File paths
        x_file = os.path.join(t_dir, "x.dat")
        rng_file = os.path.join(t_dir, "rng.dat")
        sample_file = os.path.join(t_dir, "sample.dat")
        b_file = os.path.join(t_dir, "b.dat")
        mu_file = os.path.join(t_dir, "mu.dat")
        constr_file = os.path.join(t_dir, "constr.dat")
        cmean_file = os.path.join(t_dir, "cmean.dat")
        selection_file = os.path.join(t_dir, "selection.dat")

        # Write b if provided
        if b is not None:
            b = np.asarray(b).flatten()
            if len(b) != dim:
                raise ValueError(f"b must have length {dim}, got {len(b)}")
            write_fmesher_file(b.reshape(-1, 1), filename=b_file)

        # Write mu if provided
        if mu is not None:
            mu = np.asarray(mu).flatten()
            if len(mu) != dim:
                raise ValueError(f"mu must have length {dim}, got {len(mu)}")
            write_fmesher_file(mu.reshape(-1, 1), filename=mu_file)

        # Write constraints if provided
        if constr is not None:
            A = np.asarray(constr['A'])
            e = np.asarray(constr['e']).flatten()
            if A.ndim == 1:
                A = A.reshape(1, -1)
            if A.shape[1] != dim:
                raise ValueError(f"Constraint matrix A must have {dim} columns")
            if A.shape[0] != len(e):
                raise ValueError(f"Constraint A has {A.shape[0]} rows but e has {len(e)} elements")
            # Format: [nrow, A.flatten(column-major like R's c(A)), e]
            constr_data = np.concatenate([[A.shape[0]], A.flatten(order='F'), e]).reshape(-1, 1)
            write_fmesher_file(constr_data, filename=constr_file)

        # Write sample if provided (for log-density computation only)
        if sample is not None:
            sample = np.asarray(sample)
            if sample.ndim == 1:
                sample = sample.reshape(-1, 1)
            if sample.shape[0] != dim:
                raise ValueError(f"sample must have {dim} rows, got {sample.shape[0]}")
            if sample.shape[1] < 1:
                raise ValueError("sample must have at least 1 column")
            write_fmesher_file(sample, filename=sample_file)
            n = sample.shape[1]  # Redefine n

        # Write selection if provided (convert to 0-based)
        if selection is not None:
            selection = np.asarray(selection).flatten()
            if len(selection) > dim:
                raise ValueError(f"selection length ({len(selection)}) exceeds nrow(Q) ({dim})")
            # R uses 1-based, INLA binary expects 0-based
            write_fmesher_file((selection - 1).reshape(-1, 1).astype(np.int32), filename=selection_file)

        # Match R-INLA: handle seed < 0 (restore saved RNG state)
        if seed < 0:
            if _GMRFLib_rng_state is None:
                seed = 0  # No saved state, let GMRFLib set randomly
            else:
                with open(rng_file, "wb") as fp:
                    fp.write(_GMRFLib_rng_state)

        # Build command
        cmd = [
            inla_binary,
            "-s",  # Silent mode
            "-m", "qsample",  # qsample mode
            f"-t{num_threads}",  # Threads
            "-r", reordering,  # Reordering
            "-z", str(seed),  # Seed
            "-S", smtp,  # Sparse matrix library
        ]
        if verbose:
            cmd.append("-v")

        # Positional arguments (must match R-INLA order)
        cmd.extend([
            Q_file,
            x_file,
            str(int(n)),
            rng_file,
            sample_file,
            b_file,
            mu_file,
            constr_file,
            cmean_file,
            selection_file,
        ])

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        # Match R-INLA: set environment variables before binary call
        _set_environment()
        saved_env = _run_environment_set()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=t_dir
            )
        finally:
            # Match R-INLA: restore environment variables after binary call
            _run_environment_unset(saved_env)

        if result.returncode != 0:
            raise RuntimeError(f"INLA qsample failed: {result.stderr}\n{result.stdout}")

        if verbose and result.stdout:
            print(result.stdout)

        # Read results
        if not os.path.exists(x_file):
            raise RuntimeError("INLA qsample did not produce output file")

        x = read_fmesher_file(x_file)
        if isinstance(x, sparse.spmatrix):
            x = x.toarray()

        # Parse output: last row is log-densities, rest is samples
        nx = x.shape[0] - 1
        samples = x[:nx, :]
        ld = x[nx, :].flatten()

        # Match R-INLA: always read cmean (R line 246: cmean <- inla.read.fmesher.file(cmean.file))
        cmean_data = read_fmesher_file(cmean_file)
        if isinstance(cmean_data, sparse.spmatrix):
            cmean_data = cmean_data.toarray()
        cmean = cmean_data.flatten()

        # Match R-INLA: always save RNG state after call (R lines 239-243)
        if os.path.exists(rng_file):
            with open(rng_file, "rb") as fp:
                _GMRFLib_rng_state = fp.read()

    finally:
        # Match R-INLA: always cleanup temp directory (R: unlink(t.dir, recursive=TRUE))
        shutil.rmtree(t_dir, ignore_errors=True)

    # Return results (matching R-INLA: list if logdens or compute_mean, else matrix)
    if logdens or compute_mean:
        return {
            'sample': samples,
            'logdens': ld,
            'mean': cmean,  # Already flattened to 1D above, or None
        }
    else:
        return samples


# ============================================================================
# Skew-Normal Distribution Cache (matching R-INLA's sn.cache)
# ============================================================================

_SN_CACHE = None


def _create_sn_cache():
    """
    Create interpolation tables for fast skew-normal computations.
    Matches R-INLA's inla.create.sn.cache().
    """
    global _SN_CACHE

    if _SN_CACHE is not None:
        return _SN_CACHE

    try:
        from scipy.stats import skewnorm
    except ImportError:
        warnings.warn("scipy.stats.skewnorm not available, skewness correction disabled")
        return None

    # Parameters matching R-INLA
    dig = 2  # skewness precision
    sub = 0.99  # highest value for mapping
    slb = -sub  # lowest value
    step = 0.01  # step for skewness sequence

    s = np.round(np.arange(slb, sub + step/2, step), dig)
    s_pos = np.round(np.arange(0, sub + step/2, step), dig)
    points = np.linspace(-4, 4, 50)

    def sn_map(skew, mu=0, sigma=1):
        """Map skewness to skew-normal parameters (alpha, xi, omega)."""
        skew = np.atleast_1d(np.asarray(skew, dtype=float))
        skew = np.where(np.isnan(skew), 0, skew)

        # delta = alpha / sqrt(1 + alpha^2)
        delta = np.sign(skew) * np.sqrt(
            (np.pi / 2) * (np.abs(skew)**(2/3) /
                          (((4 - np.pi) / 2)**(2/3) + np.abs(skew)**(2/3) + 1e-10))
        )
        delta = np.clip(delta, -0.9999, 0.9999)

        alpha = delta / np.sqrt(np.maximum(1 - delta**2, 1e-10))
        xi = mu - delta * np.sqrt((2 * sigma**2) / np.maximum(np.pi - 2 * delta**2, 1e-10))
        omega = np.sqrt((np.pi * sigma**2) / np.maximum(np.pi - 2 * delta**2, 1e-10))

        return {'alpha': alpha, 'xi': xi, 'omega': omega}

    # Build interpolation tables
    qsn_table = {}  # quantile function (qsn)
    psn_table = {}  # CDF (psn)
    dsn_table = {}  # PDF derivative info for Jacobian

    for sk in s_pos:
        sk_round = round(sk, dig)
        params = sn_map(sk_round)
        alpha = float(params['alpha'][0]) if hasattr(params['alpha'], '__len__') else float(params['alpha'])
        xi = float(params['xi'][0]) if hasattr(params['xi'], '__len__') else float(params['xi'])
        omega = float(params['omega'][0]) if hasattr(params['omega'], '__len__') else float(params['omega'])

        # qsn: maps from N(0,1) to SN(xi, omega, alpha)
        try:
            qvals = skewnorm.ppf(norm.cdf(points), alpha, loc=xi, scale=omega)
            qsn_table[sk_round] = interp1d(points, qvals, kind='cubic',
                                           bounds_error=False, fill_value='extrapolate')
        except:
            qsn_table[sk_round] = None

    for sk in s:
        sk_round = round(sk, dig)
        params = sn_map(sk_round)
        alpha = float(params['alpha'][0]) if hasattr(params['alpha'], '__len__') else float(params['alpha'])
        xi = float(params['xi'][0]) if hasattr(params['xi'], '__len__') else float(params['xi'])
        omega = float(params['omega'][0]) if hasattr(params['omega'], '__len__') else float(params['omega'])

        try:
            pvals = skewnorm.cdf(points, alpha, loc=xi, scale=omega)
            dvals = skewnorm.pdf(points, alpha, loc=xi, scale=omega)
            psn_table[sk_round] = interp1d(points, pvals, kind='cubic',
                                           bounds_error=False, fill_value=(0, 1))
            dsn_table[sk_round] = interp1d(points, dvals, kind='cubic',
                                           bounds_error=False, fill_value=0)
        except:
            psn_table[sk_round] = None
            dsn_table[sk_round] = None

    _SN_CACHE = {
        'qsn': qsn_table,
        'psn': psn_table,
        'dsn': dsn_table,
        's': s,
        's_pos': s_pos,
        'dig': dig,
        'skew_max': sub
    }

    return _SN_CACHE


def _fast_qsn(x, skew, cache):
    """Fast vectorized qsn using interpolation (matches R-INLA speed.fsn)."""
    if cache is None:
        return x

    x = np.atleast_1d(x).astype(float)
    skew = np.atleast_1d(skew).astype(float)

    dig = cache['dig']
    skew_max = cache['skew_max']

    # Clamp and round skewness
    skew_clamped = np.round(np.clip(skew, -skew_max, skew_max), dig)

    result = x.copy()
    unique_skews = np.unique(skew_clamped)

    for sk in unique_skews:
        mask = skew_clamped == sk
        sk_abs = round(abs(sk), dig)

        if sk_abs < 0.01:
            continue  # No correction for ~zero skewness

        qsn_func = cache['qsn'].get(sk_abs)
        if qsn_func is None:
            continue

        if sk >= 0:
            result[mask] = qsn_func(x[mask])
        else:
            # Anti-symmetry for negative skewness
            result[mask] = -qsn_func(-x[mask])

    return result


def _fast_psn(x, skew, cache):
    """Fast vectorized psn using interpolation."""
    if cache is None:
        return norm.cdf(x)

    x = np.atleast_1d(x).astype(float)
    skew = np.atleast_1d(skew).astype(float)

    dig = cache['dig']
    skew_max = cache['skew_max']

    skew_clamped = np.round(np.clip(skew, -skew_max, skew_max), dig)

    result = norm.cdf(x)
    unique_skews = np.unique(skew_clamped)

    for sk in unique_skews:
        mask = skew_clamped == sk
        psn_func = cache['psn'].get(round(sk, dig))
        if psn_func is not None:
            result[mask] = psn_func(x[mask])

    return result


def _fast_dsn(x, skew, cache):
    """Fast vectorized dsn using interpolation."""
    if cache is None:
        return norm.pdf(x)

    x = np.atleast_1d(x).astype(float)
    skew = np.atleast_1d(skew).astype(float)

    dig = cache['dig']
    skew_max = cache['skew_max']

    skew_clamped = np.round(np.clip(skew, -skew_max, skew_max), dig)

    result = norm.pdf(x)
    unique_skews = np.unique(skew_clamped)

    for sk in unique_skews:
        mask = skew_clamped == sk
        dsn_func = cache['dsn'].get(round(sk, dig))
        if dsn_func is not None:
            result[mask] = dsn_func(x[mask])

    return result


def _fast_qsn_matrix(x_matrix, skew_vec, cache):
    """
    Vectorized qsn for a 2D matrix where each row has its own skewness.

    Matches R-INLA's speed.fsn() operating on matrices (posterior.sample.R line 699).

    Parameters
    ----------
    x_matrix : ndarray (nrows, ncols)
        Standardized sample matrix
    skew_vec : ndarray (nrows,)
        Skewness value for each row
    cache : dict
        Skew-normal interpolation cache

    Returns
    -------
    result : ndarray (nrows, ncols)
        Skew-normal transformed matrix
    """
    if cache is None:
        return x_matrix.copy()

    result = x_matrix.copy()
    dig = cache['dig']
    skew_max = cache['skew_max']

    skew_clamped = np.round(np.clip(skew_vec, -skew_max, skew_max), dig)
    unique_skews = np.unique(skew_clamped)

    for sk in unique_skews:
        sk_abs = round(abs(sk), dig)
        if sk_abs < 0.01:
            continue

        qsn_func = cache['qsn'].get(sk_abs)
        if qsn_func is None:
            continue

        rows = np.where(skew_clamped == sk)[0]
        for r in rows:
            if sk >= 0:
                result[r, :] = qsn_func(x_matrix[r, :])
            else:
                # Anti-symmetry for negative skewness
                result[r, :] = -qsn_func(-x_matrix[r, :])

    return result


# ============================================================================
# Selection Interpretation (matching R-INLA exactly)
# ============================================================================

def _interpret_selection_full(selection: Optional[Dict[str, Any]], contents: Dict[str, Any]) -> tuple:
    """
    Interpret selection specification exactly like R-INLA's
    inla.posterior.sample.interpret.selection().

    Returns
    -------
    sel : array of bool
        Boolean mask for full latent field
    sel_map : dict
        Mapping from position to name (1-based indices)
    """
    if not contents:
        return None, None

    tags = contents.get('tag', [])
    starts = contents.get('start', [])
    lengths = contents.get('length', [])

    n = sum(lengths)
    sel = np.zeros(n, dtype=bool)
    names = [None] * n

    # If selection is None or empty, select all
    if selection is None or len(selection) == 0:
        selection = {tag: 0 for tag in tags}

    used_selection = set()

    for k, (tag, start, length) in enumerate(zip(tags, starts, lengths)):
        if tag not in selection:
            continue

        used_selection.add(tag)
        idx_spec = selection[tag]

        # Convert to array
        if np.isscalar(idx_spec):
            idx_spec = np.array([idx_spec])
        else:
            idx_spec = np.asarray(idx_spec)

        # Interpret specification
        if np.all(idx_spec == 0):
            # Select all
            local_idx = np.arange(1, length + 1)
        elif np.all(idx_spec > 0):
            # Select only these (1-based)
            local_idx = idx_spec[idx_spec <= length]
        elif np.all(idx_spec < 0):
            # Select all except these
            exclude = set(-idx_spec)
            local_idx = np.array([i for i in range(1, length + 1) if i not in exclude])
        else:
            raise ValueError(f"Mixed positive/negative selection for tag={tag}")

        # Mark selected elements
        for i in local_idx:
            pos = int(start - 1 + i - 1)  # Convert to 0-based
            if 0 <= pos < n:
                sel[pos] = True
                names[pos] = f"{tag}:{i}"

    # Check for unused selections
    unused = set(selection.keys()) - used_selection
    if unused:
        raise ValueError(f"Some selections are not used: {', '.join(unused)}")

    # Build selection map (1-based indices with names)
    sel_map = {}
    for i in np.where(sel)[0]:
        sel_map[i + 1] = names[i]  # 1-based index -> name

    return sel, sel_map


# ============================================================================
# Posterior Sample (matching R-INLA exactly)
# ============================================================================


class _FakeResult:
    """
    Minimal wrapper for recursive inla_posterior_sample call in .preopt branch.
    Matches R-INLA's rfake object (posterior.sample.R lines 389-396).
    """

    def __init__(self, original, fake_misc):
        self.results = {'misc': fake_misc}
        self.mlik = original.mlik
        self.summary_hyperpar = getattr(original, 'summary_hyperpar', None)
        self.mode = getattr(original, 'mode', None)


def inla_posterior_sample(
    n: int,
    result: Any,
    selection: Optional[Dict[str, Any]] = None,
    intern: bool = False,
    use_improved_mean: bool = True,
    skew_corr: bool = True,
    add_names: bool = True,
    seed: int = 0,
    num_threads: Optional[str] = None,
    parallel_configs: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate samples from the approximated joint posterior of a fitted INLA model.

    Matches R-INLA's inla.posterior.sample() function EXACTLY.

    The hyperparameters are sampled from the configurations used for numerical
    integration. The latent field is sampled from the Gaussian approximation
    conditioned on hyperparameters, with optional mean correction and skewness
    correction.

    Parameters
    ----------
    n : int
        Number of samples to generate
    result : PyINLAResult
        The fitted INLA model (must have been run with control.compute={'config': True})
    selection : dict, optional
        Select which parts of the sample to return. Keys are component names,
        values are indices (0 = all, positive = only these, negative = not these)
    intern : bool
        If True, return hyperparameters in internal scale (e.g., log-precision)
        If False, return in user scale (e.g., precision)
    use_improved_mean : bool
        If True, use marginal means; if False, use Gaussian approximation means
    skew_corr : bool
        If True, apply skewness correction to samples
    add_names : bool
        If True, add names for each element of each sample
    seed : int
        Random seed (0 for random, >0 for specific seed)
    num_threads : str, optional
        Number of threads in format 'A:B'
    parallel_configs : bool
        If True, sample configurations in parallel (not on Windows)
    verbose : bool
        Print progress information

    Returns
    -------
    samples : list of dict
        Each sample is a dict with keys:
        - 'hyperpar': hyperparameter values (dict with names, or None)
        - 'latent': latent field sample (2D array)
        - 'logdens': dict with 'hyperpar', 'latent', 'joint' log-densities

    Notes
    -----
    Requires the model to be fitted with control.compute={'config': True}.
    """
    from .options import inla_get_option

    # Handle seed
    if seed != 0:
        np.random.seed(seed)
        num_threads = "1:1"
        parallel_configs = False

    if num_threads is None:
        num_threads = inla_get_option("num.threads") or "1:1"

    # Skewness correction requires improved mean
    if not use_improved_mean:
        skew_corr = False

    # Initialize skew-normal cache if needed
    sn_cache = None
    if skew_corr:
        sn_cache = _create_sn_cache()

    # Get misc data from result
    misc = result.results.get('misc') if result.results else None
    if misc is None:
        raise ValueError("Result does not contain 'misc' data. Was the model fitted with config=True?")

    configs = misc.get('configs')
    if configs is None:
        raise ValueError(
            "No configuration data found. You need to fit the model with "
            "control={'compute': {'config': True}}"
        )

    config_list = configs.get('config', [])
    if not config_list:
        raise ValueError("No configurations found in result")

    nconfig = len(config_list)
    contents = configs.get('contents', {})
    constr = configs.get('constr')
    max_log_post = configs.get('max.log.posterior', 0)

    # ---------------------------------------------------------------
    # .preopt branch (matches R-INLA posterior.sample.R lines 388-447)
    # When .preopt is True, the Q matrix covers only core latent (no Predictor).
    # We recursively sample core latent, then compute Predictor = A @ latent + offsets.
    # ---------------------------------------------------------------
    is_preopt = configs.get('.preopt', False)

    if is_preopt:
        # Strip Predictor/APredictor from contents (R lines 398-406)
        ct_tags = list(contents.get('tag', []))
        ct_starts = list(contents.get('start', []))
        ct_lengths = list(contents.get('length', []))

        for pred_name in ['APredictor', 'Predictor']:
            if ct_tags and ct_tags[0] == pred_name:
                ct_tags = ct_tags[1:]
                # Rebase starts so first remaining tag starts at 1
                new_base = ct_starts[1] if len(ct_starts) > 1 else 1
                ct_starts = [s - new_base + 1 for s in ct_starts[1:]]
                ct_lengths = ct_lengths[1:]

        core_contents = {'tag': ct_tags, 'start': ct_starts, 'length': ct_lengths}

        # Build fake configs with .preopt=False and stripped contents (R lines 389-396)
        fake_configs = dict(configs)
        fake_configs['.preopt'] = False
        fake_configs['contents'] = core_contents

        fake_misc = dict(misc)
        fake_misc['configs'] = fake_configs
        fake_result = _FakeResult(result, fake_misc)

        # Recursive call WITHOUT selection (R line 409: selection is not passed)
        xx = inla_posterior_sample(
            n=n, result=fake_result, selection=None,
            intern=intern, use_improved_mean=use_improved_mean,
            skew_corr=skew_corr, add_names=add_names, seed=seed,
            num_threads=num_threads, parallel_configs=parallel_configs,
            verbose=verbose
        )

        # Build full A matrix: [pA @ A; A] (R line 416)
        A_mat = configs.get('A')
        pA_mat = configs.get('pA')

        if pA_mat is not None and hasattr(pA_mat, 'shape') and pA_mat.shape[0] > 0:
            if sparse.issparse(pA_mat) and sparse.issparse(A_mat):
                A_full = sparse.vstack([pA_mat @ A_mat, A_mat])
            else:
                A_full = np.vstack([np.asarray(pA_mat) @ np.asarray(A_mat),
                                    np.asarray(A_mat)])
        else:
            A_full = A_mat

        # Interpret selection on the ORIGINAL result (with Predictor in contents)
        sel_orig, sel_map_orig = _interpret_selection_full(selection, contents)
        if sel_orig is None:
            # None means select all
            sel_orig = np.ones(sum(contents.get('length', [])), dtype=bool)

        # Build predictor names (R lines 420-425)
        if pA_mat is not None and hasattr(pA_mat, 'shape') and pA_mat.shape[0] > 0:
            pnam = ([f"APredictor:{i + 1}" for i in range(pA_mat.shape[0])] +
                    [f"Predictor:{i + 1}" for i in range(A_mat.shape[0])])
        else:
            n_pred = A_mat.shape[0] if A_mat is not None and hasattr(A_mat, 'shape') else 0
            pnam = [f"Predictor:{i + 1}" for i in range(n_pred)]

        off = configs.get('offsets')
        if off is not None:
            off = np.asarray(off).flatten()

        # For each sample: compute Predictor = A @ core_latent + offsets (R lines 438-445)
        for sample_dict in xx:
            core_latent = np.asarray(sample_dict['latent']).flatten()
            nam = sample_dict.get('_latent_names', [])

            # Compute predictor
            if A_full is not None and hasattr(A_full, 'shape'):
                if sparse.issparse(A_full):
                    pred_vals = np.asarray(A_full @ core_latent).flatten()
                else:
                    pred_vals = np.asarray(A_full) @ core_latent
                if off is not None:
                    pred_vals = pred_vals + off
            else:
                pred_vals = np.array([])

            # Full latent = [predictor, core_latent] (R line 440)
            full_latent = np.concatenate([pred_vals, core_latent])
            full_names = pnam + (nam if nam else [])

            # Apply selection (R line 440: [sel])
            selected_latent = full_latent[sel_orig]
            selected_names = [full_names[i] for i in np.where(sel_orig)[0]
                              if i < len(full_names)]

            sample_dict['latent'] = selected_latent.reshape(-1, 1)
            if selected_names:
                sample_dict['_latent_names'] = selected_names

        # Store contents metadata (R line 435)
        # Must describe the structure of the RETURNED latent, not the original
        if xx:
            if selection is None or len(selection) == 0:
                # No selection: store original contents + A/pA
                stored_contents = dict(contents)
                if A_mat is not None:
                    stored_contents['A'] = A_mat
                if pA_mat is not None:
                    stored_contents['pA'] = pA_mat
                xx[0]['_contents'] = stored_contents
            else:
                # Build contents from the selected names
                new_contents = {'tag': [], 'start': [], 'length': []}
                selected_names_first = xx[0].get('_latent_names', [])
                current_start = 1
                current_tag = None
                current_tag_count = 0
                for name in selected_names_first:
                    if name:
                        tag = name.split(':')[0]
                        if tag != current_tag:
                            if current_tag is not None:
                                new_contents['tag'].append(current_tag)
                                new_contents['start'].append(current_start)
                                new_contents['length'].append(current_tag_count)
                                current_start += current_tag_count
                            current_tag = tag
                            current_tag_count = 1
                        else:
                            current_tag_count += 1
                if current_tag is not None:
                    new_contents['tag'].append(current_tag)
                    new_contents['start'].append(current_start)
                    new_contents['length'].append(current_tag_count)
                xx[0]['_contents'] = new_contents

        return xx

    # ---------------------------------------------------------------
    # Post-opt branch (.preopt == False): Q includes Predictor positions
    # ---------------------------------------------------------------

    # Interpret selection
    sel, sel_map = _interpret_selection_full(selection, contents)
    if sel is not None and sel.sum() == 0:
        return []

    # Get selection indices (1-based for qsample)
    if sel_map:
        sel_indices = np.array(sorted(sel_map.keys()))
        sel_names = [sel_map[i] for i in sel_indices]
    else:
        sel_indices = None
        sel_names = None

    # Sample configuration indices weighted by log-posterior probability
    log_posteriors = np.array([c.get('log.posterior', 0) for c in config_list])
    p = np.exp(log_posteriors - np.max(log_posteriors))
    p = p / p.sum()

    # Sample and sort indices (matching R-INLA's sort(sample(...)))
    config_indices = np.sort(np.random.choice(nconfig, size=n, p=p, replace=True))

    # Count samples per configuration
    n_per_config = np.zeros(nconfig, dtype=int)
    for k in range(nconfig):
        n_per_config[k] = np.sum(config_indices == k)

    # Get transform functions and tags
    from_theta = misc.get('from.theta', [])
    theta_tags = misc.get('theta.tags', [])

    # Get mlik for log-density computation
    mlik = result.mlik
    mlik_val = 0
    if mlik is not None:
        try:
            if isinstance(mlik, dict):
                mlik_val = float(mlik.get('log.marginal.likelihood', 0))
            elif hasattr(mlik, 'iloc'):
                # DataFrame or Series
                if hasattr(mlik, 'ndim') and mlik.ndim == 1:
                    # Series
                    mlik_val = float(mlik.iloc[0]) if mlik.size > 0 else 0
                else:
                    # DataFrame
                    mlik_val = float(mlik.iloc[0, 0]) if mlik.size > 0 else 0
            elif hasattr(mlik, 'values'):
                # Array-like
                vals = np.asarray(mlik.values).flatten()
                mlik_val = float(vals[0]) if len(vals) > 0 else 0
            elif np.isscalar(mlik):
                mlik_val = float(mlik)
            else:
                arr = np.asarray(mlik).flatten()
                mlik_val = float(arr[0]) if len(arr) > 0 else 0
        except Exception:
            mlik_val = 0

    # Generate samples
    all_samples = []
    i_sample = 0
    current_seed = seed

    for k in range(nconfig):
        if n_per_config[k] == 0:
            continue

        nk = int(n_per_config[k])
        config = config_list[k]

        # Get Q matrix
        Q_data = config.get('Q')
        if Q_data is None:
            raise ValueError(f"Configuration {k} missing Q matrix")

        # Reconstruct Q matrix
        Q = _reconstruct_Q(Q_data, configs)

        # Get mean
        if use_improved_mean and 'improved.mean' in config:
            mu = np.asarray(config['improved.mean']).flatten()
        else:
            mu = np.asarray(config.get('mean', np.zeros(Q.shape[0]))).flatten()

        # Sample latent field using INLA binary
        # In post-opt branch, pass selection to qsample (matching R-INLA lines 556-561)
        # This makes qsample return only selected rows, which is more efficient
        # and matches R-INLA's exact behavior.
        xx = inla_qsample(
            n=nk,
            Q=Q,
            mu=mu,
            constr=constr,
            logdens=True,
            seed=current_seed,
            num_threads=num_threads,
            selection=sel_indices,  # Pass selection (R: selection = sel.map)
            verbose=verbose
        )

        # Continue RNG stream if seed was set (R line 567: if (seed > 0L) seed <- -1L)
        if current_seed > 0:
            current_seed = -1

        # ---------------------------------------------------------------
        # Skewness correction (matching R-INLA lines 642-704 exactly)
        # Key: dsn_new/psn_new are computed on FULL vectors first,
        # then subset to selection for sample correction.
        # Log-density correction uses the FULL dsn_new/psn_new.
        # ---------------------------------------------------------------
        sample_corr = xx['sample'].copy()
        C_th = xx['logdens'].copy()

        if skew_corr and sn_cache is not None:
            skewness = config.get('skewness')
            if skewness is not None:
                skewness = np.asarray(skewness).flatten()
                skewness = np.where(np.isnan(skewness), 0, skewness)

                # FULL-length vectors (R lines 654-658)
                mean_GA = np.asarray(config.get('mean', mu)).flatten()
                mean_SN = mu.copy()  # improved.mean

                Qinv = config.get('Qinv')
                if Qinv is not None:
                    if sparse.issparse(Qinv):
                        sigma_th = np.sqrt(np.maximum(Qinv.diagonal(), 1e-20))
                    else:
                        sigma_th = np.sqrt(np.maximum(np.diag(np.asarray(Qinv)), 1e-20))
                else:
                    sigma_th = np.ones(Q.shape[0])

                val_max = (mean_GA - mean_SN) / np.maximum(sigma_th, 1e-10)

                # Round skewness values (R lines 662-663)
                dig = sn_cache['dig']
                skew_max = sn_cache['skew_max']
                skew_val = np.round(np.clip(skewness, -skew_max, skew_max), dig)

                # Full-length zero_not (R line 670)
                zero_not = np.where(skew_val != 0)[0]

                # Check if any selected elements have non-zero skewness (R line 671)
                if sel_indices is not None:
                    sel_has_nonzero_skew = np.any(np.isin(sel_indices, zero_not + 1))
                else:
                    sel_has_nonzero_skew = len(zero_not) > 0

                if len(zero_not) == 0 or not sel_has_nonzero_skew:
                    pass  # sample_corr and C_th already set
                else:
                    # Compute dsn_new and psn_new on FULL vectors (R lines 678-684)
                    skew_val_not = skew_val[zero_not]
                    val_max_not = val_max[zero_not]

                    dsn_new = norm.pdf(val_max)
                    psn_new = norm.cdf(val_max)

                    # Overwrite at zero_not positions with SN interpolation
                    dsn_not = _fast_dsn(val_max_not, skew_val_not, sn_cache)
                    psn_not = _fast_psn(val_max_not, skew_val_not, sn_cache)
                    dsn_new[zero_not] = dsn_not
                    psn_new[zero_not] = psn_not

                    # Subset to selection for sample correction (R lines 685-693)
                    if sel_indices is not None and len(sel_indices) < len(skew_val):
                        sel_0based = sel_indices - 1
                        mean_SN_sub = mean_SN[sel_0based]
                        sigma_th_sub = sigma_th[sel_0based]
                        skew_val_sub = skew_val[sel_0based]
                        zero_not_sub = np.where(skew_val_sub != 0)[0]
                    else:
                        mean_SN_sub = mean_SN
                        sigma_th_sub = sigma_th
                        skew_val_sub = skew_val
                        zero_not_sub = zero_not

                    if len(zero_not_sub) > 0:
                        # Vectorized sample correction (R lines 694-702)
                        x_sample_not = xx['sample'][zero_not_sub, :]
                        mean_SN_not = mean_SN_sub[zero_not_sub]
                        sigma_th_not = sigma_th_sub[zero_not_sub]

                        # Standardize
                        x_val_not = (x_sample_not - mean_SN_not[:, None]) / sigma_th_not[:, None]

                        # Vectorized qsn (R line 699: speed.fsn)
                        skew_val_not_sub = skew_val_sub[zero_not_sub]
                        fast_qsn = _fast_qsn_matrix(x_val_not, skew_val_not_sub, sn_cache)

                        sample_corr = xx['sample'].copy()
                        sample_corr[zero_not_sub, :] = sigma_th_not[:, None] * fast_qsn + mean_SN_not[:, None]

                    # Log-density correction uses FULL dsn_new/psn_new (R line 703)
                    psn_clipped = np.clip(psn_new, 1e-10, 1 - 1e-10)
                    log_corr = (np.sum(np.log(np.maximum(dsn_new, 1e-300))) -
                                np.sum(np.log(np.maximum(norm.pdf(norm.ppf(psn_clipped)), 1e-300))))
                    C_th = C_th + log_corr

        # ---------------------------------------------------------------
        # Hyperparameter transformation + Jacobian (R lines 572-641)
        # ---------------------------------------------------------------
        theta = config.get('theta')
        log_J = 0.0

        if theta is not None:
            theta = np.asarray(theta).flatten().copy()

            # Transform to user scale if needed
            if not intern and from_theta:
                theta_user = theta.copy()
                for j in range(min(len(theta), len(from_theta))):
                    t = theta[j]
                    ft = from_theta[j]
                    theta_user[j], dlog_J = _apply_transform_with_jacobian(ft, t)
                    log_J += dlog_J
                theta = theta_user

        # Compute log-density for hyperparameters (R line 569)
        ld_theta = float(max_log_post + config.get('log.posterior', 0))

        # Build row names (R line 570: nm <- names(sel.map))
        if sel_names is not None:
            nm = sel_names
        else:
            nm = []
            for tag, start, length in zip(contents.get('tag', []),
                                          contents.get('start', []),
                                          contents.get('length', [])):
                for i in range(1, length + 1):
                    nm.append(f"{tag}:{i}")

        # ---------------------------------------------------------------
        # Assemble sample dicts (R lines 706-753)
        # Note: selection was already applied by qsample, so sample_corr
        # has the right number of rows. No need for post-filtering.
        # ---------------------------------------------------------------
        for i in range(nk):
            latent = sample_corr[:, i:i + 1]

            ld_latent = float(C_th[i]) if C_th is not None else 0.0

            if theta is None:
                a_sample = {
                    'hyperpar': None,
                    'latent': latent,
                    'logdens': {
                        'hyperpar': None,
                        'latent': ld_latent,
                        'joint': ld_latent
                    }
                }
            else:
                ld_h = float(ld_theta - mlik_val + log_J)
                a_sample = {
                    'hyperpar': theta.copy(),
                    'latent': latent,
                    'logdens': {
                        'hyperpar': ld_h,
                        'latent': ld_latent,
                        'joint': float(ld_h + ld_latent)
                    }
                }

            # Add names (R lines 729-750)
            if add_names or i_sample == 0:
                if nm:
                    a_sample['_latent_names'] = nm[:latent.shape[0]]

                if theta is not None:
                    if not intern:
                        summary_hyper = result.summary_hyperpar
                        if summary_hyper is not None and hasattr(summary_hyper, 'index'):
                            hyper_names = list(summary_hyper.index)[:len(theta)]
                        else:
                            hyper_names = theta_tags[:len(theta)] if theta_tags else [f'theta{j+1}' for j in range(len(theta))]
                    else:
                        hyper_names = theta_tags[:len(theta)] if theta_tags else [f'theta{j+1}' for j in range(len(theta))]

                    a_sample['hyperpar'] = dict(zip(hyper_names, theta))

            all_samples.append(a_sample)
            i_sample += 1

    # ---------------------------------------------------------------
    # Store contents as attribute (R lines 756-771)
    # ---------------------------------------------------------------
    if all_samples:
        if selection is None or len(selection) == 0:
            all_samples[0]['_contents'] = contents
        else:
            # Build new contents for selection (R lines 759-770)
            new_contents = {'tag': [], 'start': [], 'length': []}
            current_start = 1
            current_tag = None
            current_tag_count = 0

            for name in (sel_names or []):
                if name:
                    tag = name.split(':')[0]
                    if tag != current_tag:
                        if current_tag is not None:
                            new_contents['tag'].append(current_tag)
                            new_contents['start'].append(current_start)
                            new_contents['length'].append(current_tag_count)
                            current_start += current_tag_count
                        current_tag = tag
                        current_tag_count = 1
                    else:
                        current_tag_count += 1

            if current_tag is not None:
                new_contents['tag'].append(current_tag)
                new_contents['start'].append(current_start)
                new_contents['length'].append(current_tag_count)

            all_samples[0]['_contents'] = new_contents

    return all_samples


def inla_posterior_sample_eval(
    fun: Union[Callable, List[str], str],
    samples: List[Dict[str, Any]],
    return_matrix: bool = True,
    **kwargs
) -> Union[np.ndarray, List[Any]]:
    """
    Evaluate a function over posterior samples.

    Matches R-INLA's inla.posterior.sample.eval() function.

    Parameters
    ----------
    fun : callable or str or list of str
        Function to evaluate on each sample. The function receives named arguments
        for each latent field component and 'theta' for hyperparameters.
        If a string or list of strings, interpreted as variable names to extract.
    samples : list
        Output from inla_posterior_sample()
    return_matrix : bool
        If True, return results as matrix (samples in columns)
        If False, return as list
    **kwargs
        Additional arguments passed to fun

    Returns
    -------
    results : array or list
        Function evaluations for each sample

    Examples
    --------
    >>> # Extract intercept values
    >>> intercepts = inla_posterior_sample_eval("Intercept", samples)

    >>> # Evaluate custom function
    >>> def my_func(Intercept, x1, theta, **kw):
    ...     return np.exp(Intercept) * theta[0]
    >>> results = inla_posterior_sample_eval(my_func, samples)
    """
    if not samples:
        raise ValueError("samples list is empty")

    # Get contents from first sample
    contents = samples[0].get('_contents', {})

    # Handle string/list of strings shorthand (like R-INLA)
    if isinstance(fun, str):
        fun = [fun]
    if isinstance(fun, list) and all(isinstance(f, str) for f in fun):
        var_names = fun
        def fun(**env):
            result = []
            for name in var_names:
                val = env.get(name)
                if val is None:
                    # Try with parentheses for (Intercept)
                    val = env.get(f'({name})')
                if val is None:
                    val = np.nan
                if isinstance(val, np.ndarray):
                    result.extend(val.flatten())
                else:
                    result.append(val)
            return np.array(result)

    results = []

    for sample in samples:
        # Build environment with named variables
        env = {}

        # Add theta
        theta = sample.get('hyperpar')
        if theta is not None:
            if isinstance(theta, dict):
                env['theta'] = np.array(list(theta.values()))
                env.update(theta)
            else:
                env['theta'] = np.asarray(theta)

        # Add latent field components
        latent = sample.get('latent')
        latent_names = sample.get('_latent_names', [])

        if latent is not None:
            # Handle dict format (for mock samples or direct mapping)
            if isinstance(latent, dict):
                env.update(latent)
                # Add alias for (Intercept)
                if '(Intercept)' in latent:
                    env['Intercept'] = latent['(Intercept)']
            else:
                # Handle array format with contents mapping
                latent_flat = np.asarray(latent).flatten()

                # Use contents to name components
                if contents:
                    tags = contents.get('tag', [])
                    starts = contents.get('start', [])
                    lengths = contents.get('length', [])

                    for tag, start, length in zip(tags, starts, lengths):
                        idx = slice(start - 1, start - 1 + length)  # R is 1-indexed
                        if length == 1:
                            env[tag] = float(latent_flat[idx][0]) if len(latent_flat[idx]) > 0 else np.nan
                        else:
                            env[tag] = latent_flat[idx]

                        # Alias for (Intercept)
                        if tag == '(Intercept)':
                            env['Intercept'] = env[tag]
                elif latent_names:
                    # Use _latent_names for mapping
                    for i, name in enumerate(latent_names):
                        if name and i < len(latent_flat):
                            # Parse "tag:index" format
                            if ':' in name:
                                tag = name.split(':')[0]
                                if tag not in env:
                                    env[tag] = []
                                env[name] = latent_flat[i]
                            else:
                                env[name] = latent_flat[i]

        # Add kwargs
        env.update(kwargs)

        # Evaluate function
        try:
            result = fun(**env)
        except TypeError:
            # Try calling without kwargs
            result = fun()

        results.append(result)

    if return_matrix:
        # Convert to matrix with samples as columns
        if len(results) > 0:
            results = np.column_stack(results)
            if results.ndim == 1:
                results = results.reshape(1, -1)
        else:
            results = np.array([])

    return results


def inla_hyperpar_sample(
    n: int,
    result: Any,
    intern: bool = False,
    improve_marginals: bool = False,
) -> "pd.DataFrame":
    """
    Sample from the joint posterior of hyperparameters.

    Matches R-INLA's inla.hyperpar.sample() function.

    Parameters
    ----------
    n : int
        Number of samples to generate
    result : PyINLAResult
        The fitted INLA model
    intern : bool
        If True, return samples in internal scale (e.g., log-precision)
        If False, return samples in user scale (e.g., precision)
    improve_marginals : bool
        If True, improve samples using marginal quantile matching

    Returns
    -------
    samples : DataFrame
        Matrix where each row is a sample, columns are hyperparameters

    Notes
    -----
    Uses the CCD (Central Composite Design) approach with eigendecomposition
    of the covariance matrix at the mode.
    """
    import pandas as pd

    misc = result.results.get('misc') if result.results else None
    if misc is None:
        raise ValueError("Result does not contain 'misc' data")

    # Get covariance matrix and its eigendecomposition
    cov_intern = misc.get('cov.intern')
    if cov_intern is None or (isinstance(cov_intern, np.ndarray) and cov_intern.size == 0):
        return None  # No hyperparameters

    cov_intern = np.asarray(cov_intern)
    p = cov_intern.shape[0]

    if p == 0:
        return None

    # Get eigenvalues and eigenvectors
    cov_eval = misc.get('cov.intern.eigenvalues')
    cov_evec = misc.get('cov.intern.eigenvectors')

    if cov_eval is None or cov_evec is None:
        # Compute eigendecomposition
        cov_eval, cov_evec = np.linalg.eigh(cov_intern)
    else:
        cov_eval = np.asarray(cov_eval)
        cov_evec = np.asarray(cov_evec)

    # Get mode
    theta_mode = misc.get('theta.mode')
    if theta_mode is None:
        # Try to get from result.mode
        if result.mode is not None:
            theta_mode = result.mode.get('theta', np.zeros(p))
        else:
            theta_mode = np.zeros(p)
    theta_mode = np.asarray(theta_mode).flatten()

    # Get stdev corrections for asymmetric sampling
    sd_plus = misc.get('stdev.corr.positive')
    sd_neg = misc.get('stdev.corr.negative')

    if sd_plus is None:
        sd_plus = np.ones(p)
    else:
        sd_plus = np.asarray(sd_plus).flatten()

    if sd_neg is None:
        sd_neg = np.ones(p)
    else:
        sd_neg = np.asarray(sd_neg).flatten()

    # Number of samples (more if improving marginals)
    ns = max(n, 300) if improve_marginals else n

    # Generate samples with asymmetric sampling (matching R-INLA exactly)
    z = np.zeros((ns, p))
    for i in range(p):
        prob = np.array([sd_plus[i], sd_neg[i]])
        prob = prob / prob.sum()
        direction = np.random.choice([0, 1], size=ns, p=prob)
        s = np.array([sd_plus[i], -sd_neg[i]])
        z[:, i] = s[direction] * np.abs(np.random.randn(ns))

    # Transform: theta = A @ z + mode, where A = V @ sqrt(Lambda)
    A = cov_evec @ np.diag(np.sqrt(np.maximum(cov_eval, 0)))
    theta = z @ A.T + theta_mode

    # Improve marginals using quantile matching
    if improve_marginals:
        int_marginals = result.results.get('internal.marginals.hyperpar', {})
        if int_marginals:
            from .pure.marginal_utils import inla_qmarginal
            for i, (name, marg) in enumerate(int_marginals.items()):
                if i < p:
                    sorted_idx = np.argsort(theta[:, i])
                    ranks = np.empty_like(sorted_idx)
                    ranks[sorted_idx] = np.arange(ns)
                    ecdf = (ranks + 0.5) / ns
                    theta[:, i] = inla_qmarginal(ecdf * (ns / (ns + 1)), marg)

    # Take first n samples
    theta = theta[:n, :]

    # Transform to user scale if needed
    if not intern:
        from_theta = misc.get('from.theta', [])
        if from_theta:
            for i in range(min(p, len(from_theta))):
                ft = from_theta[i]
                theta[:, i] = np.array([_apply_transform(ft, t) for t in theta[:, i]])

    # Set column names
    theta_tags = misc.get('theta.tags', [f'theta{i+1}' for i in range(p)])
    if intern:
        col_names = theta_tags
    else:
        summary_hyper = result.summary_hyperpar
        if summary_hyper is not None and hasattr(summary_hyper, 'index'):
            col_names = list(summary_hyper.index)[:p]
        else:
            col_names = theta_tags

    return pd.DataFrame(
        theta,
        index=[f'sample:{i+1}' for i in range(n)],
        columns=col_names[:p] if len(col_names) >= p else [f'hyper{i+1}' for i in range(p)]
    )


# ============================================================================
# Helper functions
# ============================================================================

def _reconstruct_Q(Q_data: Any, configs: Dict[str, Any]) -> sparse.spmatrix:
    """Reconstruct sparse Q matrix from configuration data."""
    if sparse.issparse(Q_data):
        return Q_data

    if isinstance(Q_data, np.ndarray):
        if Q_data.ndim == 2:
            return sparse.csc_matrix(Q_data)

    n = configs.get('n', 0)
    i_idx = configs.get('i', [])
    j_idx = configs.get('j', [])

    if isinstance(Q_data, (list, np.ndarray)):
        values = np.asarray(Q_data).flatten()
    else:
        values = Q_data

    if len(i_idx) == 0 or n == 0:
        raise ValueError("Cannot reconstruct Q matrix: missing indices or dimension")

    Q = sparse.coo_matrix((values, (i_idx, j_idx)), shape=(n, n))
    Q = Q + Q.T - sparse.diags(Q.diagonal())

    return Q.tocsc()


# ============================================================================
# R expression parser for from.theta / to.theta transforms
# Matches R-INLA's behavior: parses "function (x) <<NEWLINE>>exp(x)" strings
# into Python callables, with analytical derivatives when possible.
# ============================================================================

# Cache for parsed transforms (avoid re-parsing the same string)
_PARSED_TRANSFORMS: Dict[str, tuple] = {}

# Known analytical derivatives for common R-INLA transforms.
# Maps normalised body expression -> derivative expression.
# Matches R-INLA's D(body(from.theta[[i]]), arg) symbolic differentiation.
_KNOWN_DERIVATIVES = {
    'exp(x)': 'exp(x)',
    'x': '1.0',
    '-x': '-1.0',
    'exp(-x)': '-exp(-x)',
    '1/exp(x)': '-exp(-x)',
    'exp(x)/(1+exp(x))': 'exp(x)/(1+exp(x))**2',
    'exp(x)/(1 + exp(x))': 'exp(x)/(1 + exp(x))**2',
    '2*exp(x)/(1+exp(x))-1': '2*exp(x)/(1+exp(x))**2',
    '2*exp(x)/(1 + exp(x))-1': '2*exp(x)/(1 + exp(x))**2',
    '2*exp(x)/(1 + exp(x)) - 1': '2*exp(x)/(1 + exp(x))**2',
    '0.5+0.5*exp(x)/(1+exp(x))': '0.5*exp(x)/(1+exp(x))**2',
    '0.5 + 0.5*exp(x)/(1 + exp(x))': '0.5*exp(x)/(1 + exp(x))**2',
    'sqrt(exp(x))': '0.5*sqrt(exp(x))',
    '1/sqrt(exp(x))': '-0.5/sqrt(exp(x))',
}


def _parse_r_params(params_str: str) -> List[tuple]:
    """
    Parse R function parameter string like "x" or "x, range0 = 1.23, alpha = 2".
    Returns list of (name, default_value_or_None).
    """
    params = []
    for part in params_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            name, default = part.split('=', 1)
            params.append((name.strip(), default.strip()))
        else:
            params.append((part.strip(), None))
    return params


def _r_expr_to_python(body: str, params: List[tuple]) -> callable:
    """
    Translate an R math expression body to a Python callable.

    Uses safe eval with only numpy math functions exposed.
    The expressions come from INLA binary output (theta-from files),
    not user input.

    Parameters
    ----------
    body : str
        R expression body, e.g. "exp(x)", "exp(x)/(1+exp(x))"
    params : list of (name, default_or_None)
        Function parameters extracted from the R signature

    Returns
    -------
    callable
        Python function f(x) -> float
    """
    body = body.strip()
    if not body:
        return lambda x: x

    # Remove R-style braces around multi-line bodies: { expr }
    if body.startswith('{') and body.endswith('}'):
        body = body[1:-1].strip()
    # Handle multi-statement bodies: take the last expression (or the return() value)
    ret_match = re.search(r'return\s*\(([^)]+)\)', body)
    if ret_match:
        body = ret_match.group(1).strip()
    elif ';' in body:
        body = body.split(';')[-1].strip()

    main_var = params[0][0] if params else 'x'

    # Build default values dict from extra params
    defaults = {}
    for name, default in params[1:]:
        if default is not None:
            try:
                defaults[name] = float(default)
            except ValueError:
                defaults[name] = 0.0

    # Safe namespace with only math operations
    safe_ns = {
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
        'abs': np.abs, 'pi': np.pi,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
        'Inf': np.inf, 'inf': np.inf,
        'TRUE': True, 'FALSE': False,
    }
    safe_ns.update(defaults)

    # R-to-Python syntax adjustments
    py_body = body
    py_body = py_body.replace('^', '**')
    # Handle R's is.infinite, is.finite
    py_body = re.sub(r'is\.infinite\(([^)]+)\)', r'(abs(\1) == inf)', py_body)
    py_body = re.sub(r'is\.finite\(([^)]+)\)', r'(abs(\1) != inf)', py_body)

    try:
        code = compile(py_body, '<r_transform>', 'eval')

        def transform(x, _code=code, _ns=safe_ns, _var=main_var):
            ns = dict(_ns)
            ns[_var] = x
            return eval(_code, {"__builtins__": {}}, ns)

        # Quick sanity test
        transform(0.0)
        return transform
    except Exception:
        warnings.warn(
            f"Could not parse R transform expression: '{body}', using identity",
            stacklevel=3,
        )
        return lambda x: x


def _parse_r_function_string(func_str: str) -> tuple:
    """
    Parse an R function string from theta-from / theta-to files.

    Handles forms like:
    - "function (x) <<NEWLINE>>exp(x)"
    - "function (x) <<NEWLINE>>x"
    - "function (x) <<NEWLINE>>exp(x)/(1 + exp(x))"
    - "function (x, range0 = 1.23) <<NEWLINE>>{<<NEWLINE>>range0 * exp(-x)<<NEWLINE>>}"

    Parameters
    ----------
    func_str : str
        R function string, possibly with <<NEWLINE>> markers

    Returns
    -------
    (func, deriv_func_or_None)
        func: callable f(x) -> float
        deriv_func: callable f'(x) -> float, or None if no analytical form known
    """
    if func_str is None or func_str == '(null)':
        return (lambda x: x, lambda x: 1.0)

    # Check cache
    cache_key = str(func_str).strip()
    if cache_key in _PARSED_TRANSFORMS:
        return _PARSED_TRANSFORMS[cache_key]

    # If already callable, wrap it
    if callable(func_str):
        result = (func_str, None)
        _PARSED_TRANSFORMS[cache_key] = result
        return result

    s = str(func_str).strip()

    # Split on <<NEWLINE>> to separate header from body
    parts = s.split('<<NEWLINE>>')
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        result = (lambda x: x, lambda x: 1.0)
        _PARSED_TRANSFORMS[cache_key] = result
        return result

    # Parse function signature
    header = parts[0]
    sig_match = re.match(r'function\s*\(([^)]*)\)', header)

    if sig_match:
        params = _parse_r_params(sig_match.group(1))
        # Body is everything after the header
        body = ' '.join(parts[1:]) if len(parts) > 1 else ''
        # Remove leading/trailing braces
        body = body.strip()
        if body.startswith('{'):
            body = body[1:]
        if body.endswith('}'):
            body = body[:-1]
        body = body.strip()
    else:
        # No "function(...)" wrapper — the whole string is the expression
        params = [('x', None)]
        body = ' '.join(parts)

    if not body:
        body = 'x'

    # Build the callable
    func = _r_expr_to_python(body, params)

    # Try to find analytical derivative
    deriv_func = None
    body_normalised = body.strip()
    if body_normalised in _KNOWN_DERIVATIVES:
        deriv_body = _KNOWN_DERIVATIVES[body_normalised]
        deriv_func = _r_expr_to_python(deriv_body, params)

    result = (func, deriv_func)
    _PARSED_TRANSFORMS[cache_key] = result
    return result


def _apply_transform(transform_str: str, value: float) -> float:
    """
    Apply R-style transform string to a value.

    Uses the R expression parser to handle all transform formats from
    INLA binary output (theta-from files).
    """
    if callable(transform_str):
        return transform_str(value)
    func, _ = _parse_r_function_string(str(transform_str))
    return func(value)


def _apply_transform_with_jacobian(transform_str: str, value: float) -> tuple:
    """
    Apply transform and compute log-Jacobian for density correction.

    Matches R-INLA's approach: tries analytical derivative first (like R's D()),
    falls back to numerical differentiation with h = eps^0.25 (matching R's
    .Machine$double.eps^0.25).
    """
    func, deriv_func = _parse_r_function_string(str(transform_str))

    transformed = func(value)

    # Try analytical derivative first (matching R's D(body(from.theta), arg))
    if deriv_func is not None:
        try:
            deriv_val = deriv_func(value)
            if np.isfinite(deriv_val) and abs(deriv_val) > 0:
                log_J = -np.log(np.abs(deriv_val))
                return transformed, log_J
        except Exception:
            pass

    # Fall back to numerical differentiation
    # Step size matches R's .Machine$double.eps^0.25 (optimal for central differences)
    h = np.finfo(float).eps ** 0.25

    t1 = func(value - h)
    t2 = func(value + h)

    # Log-Jacobian (negative because we're transforming from internal to user scale)
    log_J = -np.log(np.abs((t2 - t1) / (2 * h)))

    return transformed, log_J
