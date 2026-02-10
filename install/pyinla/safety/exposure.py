"""Exposure, trials, scale, and weights validation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

from .errors import SafetyError
from .family import _normalize_family_spec
from .response import _extract_response_array
from .utils import _as_float_list, _infer_observation_count, _normalize_positive_sequence


def enforce_scale_usage(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_gaussian = any(f == "gaussian" for f in normalized)
    has_xbinomial = any(f == "xbinomial" for f in normalized)
    has_gamma = any(f == "gamma" for f in normalized)
    has_beta = any(f == "beta" for f in normalized)
    has_logistic = any(f == "logistic" for f in normalized)
    has_t = any(f == "t" for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)

    if has_gaussian or has_xbinomial or has_gamma or has_beta or has_logistic or has_t or has_nbinomial:
        scale = kwargs.get("scale")
        if scale is None:
            return
        values = _normalize_positive_sequence("scale", scale)
        if has_gaussian or has_logistic or has_t or has_nbinomial:
            bad = [val for val in values if not (val > 0.0) or math.isnan(val)]
            if bad:
                raise SafetyError(
                    "pyinla safety check: all entries in 'scale' must be strictly positive for gaussian/logistic/t/nbinomial; "
                    f"found invalid values {bad[:5]}."
                )
            return
        bad = [val for val in values if val < 0.0 or math.isnan(val)]
        if bad:
            raise SafetyError(
                "pyinla safety check: 'scale' must be non-negative for xbinomial/gamma/beta; "
                f"found invalid values {bad[:5]}."
            )
        if has_xbinomial:
            too_large = [val for val in values if val > 1.0 + 1e-12]
            if too_large:
                raise SafetyError(
                    "pyinla safety check: 'scale' entries for xbinomial must be <= 1; "
                    f"found values above 1: {too_large[:5]}."
                )
        return

    if kwargs.get("scale") is not None:
        raise SafetyError(
            "pyinla safety check: 'scale' is only permitted for gaussian, nbinomial, xbinomial, gamma, beta, logistic, or t likelihoods."
        )


def enforce_exposure_usage(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_poisson = any(f == "poisson" for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    if has_poisson or has_nbinomial:
        return
    if "E" in kwargs and kwargs["E"] is not None:
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) is only permitted for poisson or nbinomial likelihoods."
        )


def enforce_poisson_exposure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_poisson = any(f == "poisson" for f in normalized)
    if not has_poisson:
        return

    exposures = kwargs.get("E")
    if exposures is None:
        return

    exp_list = list(_as_float_list("E", exposures))
    if len(exp_list) == 0:
        raise SafetyError("pyinla safety check: 'E' (exposure) may not be empty.")

    import numpy as np
    exp_arr = np.asarray(exp_list, dtype=float)
    invalid_mask = ~(exp_arr > 0)
    if invalid_mask.any():
        bad_vals = exp_arr[invalid_mask][:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) must be strictly positive; found invalid values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(exp_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'E' length ({}) must match number of observations ({}).".format(
                len(exp_list), n_obs
            )
        )


def enforce_nbinomial_exposure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate exposure (E) for nbinomial likelihood: must be strictly positive."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    if not has_nbinomial:
        return

    exposures = kwargs.get("E")
    if exposures is None:
        return

    exp_list = list(_as_float_list("E", exposures))
    if len(exp_list) == 0:
        raise SafetyError("pyinla safety check: 'E' (exposure) may not be empty for nbinomial.")

    import numpy as np
    exp_arr = np.asarray(exp_list, dtype=float)
    invalid_mask = ~(exp_arr > 0)
    if invalid_mask.any():
        bad_vals = exp_arr[invalid_mask][:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'E' (exposure) must be strictly positive for nbinomial; found invalid values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(exp_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'E' length ({}) must match number of observations ({}) for nbinomial.".format(
                len(exp_list), n_obs
            )
        )


def enforce_binomial_trials(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    requires_trials = any(f in ("binomial", "xbinomial", "betabinomial", "nbinomial2") for f in normalized)
    ntrials = kwargs.get("Ntrials")

    if not requires_trials:
        if ntrials is not None:
            raise SafetyError(
                "pyinla safety check: 'Ntrials' is only permitted for binomial/xbinomial/betabinomial/nbinomial2 likelihoods."
            )
        return

    if ntrials is None:
        data = _extract_response_array(kwargs)
        if data is None:
            raise SafetyError(
                "pyinla safety check: provide 'Ntrials' for aggregated binomial responses (currently missing)."
            )
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(data, dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            values = set(np.unique(finite))
        except Exception:
            try:
                values = set(float(v) for v in data)
            except Exception as exc:
                raise SafetyError(
                    "pyinla safety check: could not interpret response when validating missing 'Ntrials'."
                ) from exc
        import numpy as np
        values = {v for v in values if np.isfinite(v)}
        if not values.issubset({0.0, 1.0}):
            raise SafetyError(
                "pyinla safety check: aggregated binomial data requires 'Ntrials'; response contains values other than 0/1."
            )
        return

    trials_list = list(_as_float_list("Ntrials", ntrials))
    if len(trials_list) == 0:
        raise SafetyError("pyinla safety check: 'Ntrials' may not be empty for binomial models.")

    import numpy as np
    trials_arr = np.asarray(trials_list, dtype=float)
    non_positive = trials_arr[trials_arr <= 0]
    if len(non_positive) > 0:
        bad_vals = non_positive[:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'Ntrials' must be positive; found invalid values: {}.".format(bad_vals)
        )
    non_integer = trials_arr[trials_arr != np.floor(trials_arr)]
    if len(non_integer) > 0:
        bad_vals = non_integer[:5].tolist()
        raise SafetyError(
            "pyinla safety check: 'Ntrials' must be integers; found non-integer values: {}.".format(bad_vals)
        )

    n_obs = _infer_observation_count(kwargs)
    if n_obs is not None and len(trials_list) != n_obs:
        raise SafetyError(
            "pyinla safety check: 'Ntrials' length ({}) must match number of observations ({}).".format(
                len(trials_list), n_obs
            )
        )


def enforce_offset_values(kwargs: Dict[str, Any]) -> None:
    """Validate that offset values are finite (no NaN/Inf)."""
    offset = kwargs.get("offset")
    if offset is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(offset, dtype=float).reshape(-1)
        mask_invalid = ~np.isfinite(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: 'offset' must contain finite values (no NaN/Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in offset]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret 'offset' as numeric values."
        ) from exc

    bad = [val for val in values if not math.isfinite(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: 'offset' must contain finite values (no NaN/Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_weights_values(kwargs: Dict[str, Any]) -> None:
    """Validate that weights values are strictly positive and finite."""
    weights = kwargs.get("weights")
    if weights is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(weights, dtype=float).reshape(-1)
        mask_nonfinite = ~np.isfinite(arr)
        mask_nonpositive = arr <= 0.0
        mask_invalid = mask_nonfinite | mask_nonpositive
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: 'weights' must be strictly positive (> 0) and finite; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in weights]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret 'weights' as numeric values."
        ) from exc

    bad = [val for val in values if not math.isfinite(val) or val <= 0.0]
    if bad:
        raise SafetyError(
            "pyinla safety check: 'weights' must be strictly positive (> 0) and finite; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_random_weights_values(weights_vec, label: str) -> None:
    """Validate random effect weights: must be finite (NaN allowed, replaced with 0 upstream)."""
    try:
        import numpy as _np
        arr = _np.asarray(weights_vec, dtype=float).ravel()
        mask_inf = _np.isinf(arr)
        if mask_inf.any():
            bad = arr[mask_inf]
            raise SafetyError(
                f"pyinla safety check: random effect '{label}' weights must be finite; "
                f"found Inf/-Inf entries (showing up to 5): {bad[:5].tolist()}"
            )
    except SafetyError:
        raise
    except Exception:
        pass
