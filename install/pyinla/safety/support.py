"""Response support validation for different likelihood families."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

from .errors import SafetyError
from .family import _normalize_family_spec
from .response import _extract_response_array


def enforce_gamma_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that gamma response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "gamma" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: gamma likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5]}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for gamma likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: gamma likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_beta_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that beta response values are in valid range (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "beta" not in normalized:
        return

    control = kwargs.get("control") or {}
    beta_censor = None
    if isinstance(control, dict):
        family_block = control.get("family")
        if isinstance(family_block, dict):
            beta_censor = family_block.get("beta.censor.value")
    censoring_enabled = beta_censor is not None

    response = _extract_response_array(kwargs)
    if response is None:
        return
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    try:
        arr = np.asarray(response, dtype=float).reshape(-1) if np is not None else [float(v) for v in response]
    except Exception as exc:
        raise SafetyError("pyinla safety check: could not interpret response values for beta likelihood.") from exc

    if np is not None:
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr)
        if not censoring_enabled:
            mask_invalid |= (valid_arr <= 0.0) | (valid_arr >= 1.0)
        else:
            mask_invalid |= (valid_arr < 0.0) | (valid_arr > 1.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: beta likelihood requires response values in {} when censoring {}; "
                f"found invalid values (showing up to 5): {invalid[:5]}".format(
                    "[0,1]" if censoring_enabled else "(0,1)",
                    "enabled" if censoring_enabled else "disabled",
                )
            )
    else:
        if censoring_enabled:
            bad = [val for val in arr if not math.isnan(val) and (math.isinf(val) or not (0.0 <= float(val) <= 1.0))]
        else:
            bad = [val for val in arr if not math.isnan(val) and (math.isinf(val) or not (0.0 < float(val) < 1.0))]
        if bad:
            raise SafetyError(
                "pyinla safety check: beta likelihood requires response values in {} when censoring {}; "
                f"found invalid values (showing up to 5): {bad[:5]}".format(
                    "[0,1]" if censoring_enabled else "(0,1)",
                    "enabled" if censoring_enabled else "disabled",
                )
            )


def enforce_poisson_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that Poisson response values are non-negative integers (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "poisson" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_inf = np.isinf(valid_arr)
        mask_negative = valid_arr < 0.0
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: poisson likelihood requires non-negative integer response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for poisson likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: poisson likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_nbinomial_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that negative binomial response values are non-negative integers (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    nbinom_families = {"nbinomial", "nbinomial2"}
    if not any(f in nbinom_families for f in normalized):
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_inf = np.isinf(valid_arr)
        mask_negative = valid_arr < 0.0
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: nbinomial likelihood requires non-negative integer response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for nbinomial likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: nbinomial likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_binomial_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that binomial response values are in valid range [0, Ntrials] (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    binomial_families = {"binomial", "xbinomial", "betabinomial"}
    if not any(f in binomial_families for f in normalized):
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    ntrials = kwargs.get("Ntrials")

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_inf = np.isinf(valid_arr)
        mask_negative = valid_arr < 0.0
        mask_noninteger = np.abs(valid_arr - np.round(valid_arr)) > 1e-10
        mask_invalid = mask_inf | mask_negative | mask_noninteger

        if ntrials is not None:
            ntrials_arr = np.asarray(ntrials, dtype=float).reshape(-1)
            if ntrials_arr.size == 1:
                ntrials_check = np.full(valid_arr.size, ntrials_arr[0])
            else:
                ntrials_check = ntrials_arr[valid_mask]
            mask_exceeds = valid_arr > ntrials_check
            mask_invalid = mask_invalid | mask_exceeds

        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: binomial likelihood requires non-negative integer response values "
                f"not exceeding Ntrials; found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for binomial likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val < 0.0 or abs(val - round(val)) > 1e-10)]
    if bad:
        raise SafetyError(
            "pyinla safety check: binomial likelihood requires non-negative integer response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_exponential_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that exponential response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "exponential" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: exponential likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for exponential likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: exponential likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_lognormal_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that lognormal response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "lognormal" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: lognormal likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for lognormal likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: lognormal likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_weibull_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that weibull response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "weibull" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: weibull likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for weibull likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: weibull likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_loglogistic_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that loglogistic response values are strictly positive (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "loglogistic" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]
        if valid_arr.size == 0:
            return
        mask_invalid = np.isinf(valid_arr) | (valid_arr <= 0.0)
        invalid = valid_arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: loglogistic likelihood requires strictly positive response values; "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for loglogistic likelihood."
        ) from exc

    bad = [val for val in values if not math.isnan(val) and (math.isinf(val) or val <= 0.0)]
    if bad:
        raise SafetyError(
            "pyinla safety check: loglogistic likelihood requires strictly positive response values; "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_gaussian_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that gaussian response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "gaussian" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: gaussian likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for gaussian likelihood."
        ) from exc

    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: gaussian likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_logistic_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that logistic response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "logistic" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: logistic likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for logistic likelihood."
        ) from exc

    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: logistic likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_t_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that student-t response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "t" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: student-t likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for student-t likelihood."
        ) from exc

    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: student-t likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )


def enforce_sn_support(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that skew-normal response values have no Inf (NaN allowed as missing data)."""
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    if "sn" not in normalized:
        return

    response = _extract_response_array(kwargs)
    if response is None:
        return

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(response, dtype=float).reshape(-1)
        mask_invalid = np.isinf(arr)
        invalid = arr[mask_invalid]
        if invalid.size > 0:
            raise SafetyError(
                "pyinla safety check: skew-normal likelihood requires finite response values (no Inf); "
                f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
            )
        return
    except SafetyError:
        raise
    except Exception:
        pass

    try:
        values = [float(v) for v in response]
    except Exception as exc:
        raise SafetyError(
            "pyinla safety check: could not interpret response values for skew-normal likelihood."
        ) from exc

    bad = [val for val in values if math.isinf(val)]
    if bad:
        raise SafetyError(
            "pyinla safety check: skew-normal likelihood requires finite response values (no Inf); "
            f"found invalid entries (showing up to 5): {bad[:5]}"
        )
