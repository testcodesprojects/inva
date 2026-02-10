"""Response validation and extraction utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .errors import SafetyError
from .family import _normalize_family_spec

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


def _get_response_object(kwargs: Dict[str, Any]) -> Optional[Any]:
    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return None
    response = model.get("response")
    if response is None:
        return None
    data = kwargs.get("data")

    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    def lookup_by_name(name: str) -> Optional[Any]:
        if data is None:
            return None
        if pd is not None and isinstance(data, pd.DataFrame):
            if name in data.columns:
                return data[name]
            return None
        if isinstance(data, dict):
            return data.get(name)
        return None

    if isinstance(response, str):
        return lookup_by_name(response)
    if isinstance(response, list) and all(isinstance(r, str) for r in response):
        return None
    if isinstance(response, tuple) and len(response) == 2:
        _, values = response
        return values
    return response


def _extract_response_array(kwargs: Dict[str, Any]) -> Optional[Any]:
    obj = _get_response_object(kwargs)
    if obj is None:
        return None
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(obj)
        if arr.size == 0:
            return []
        return arr.reshape(-1)
    except Exception:
        pass
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return obj


def enforce_multiple_likelihood_response(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    """Validate that multiple likelihood responses have proper NaN separation."""
    import numpy as np

    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return

    response = model.get("response")
    if response is None:
        return

    if not (isinstance(response, list) and len(response) > 1 and all(isinstance(r, str) for r in response)):
        return

    data = kwargs.get("data")
    if data is None:
        return

    try:
        import pandas as pd
    except Exception:
        return

    if not isinstance(data, pd.DataFrame):
        return

    missing_cols = [r for r in response if r not in data.columns]
    if missing_cols:
        raise SafetyError(
            f"pyinla safety check: response columns not found in data: {missing_cols}"
        )

    n_rows = len(data)
    response_arrays = []
    for col in response:
        arr = data[col].to_numpy()
        response_arrays.append(arr)

    non_nan_counts = np.zeros(n_rows, dtype=int)
    for arr in response_arrays:
        non_nan_counts += ~np.isnan(arr.astype(float))

    overlapping_rows = np.where(non_nan_counts > 1)[0]
    if len(overlapping_rows) > 0:
        show_rows = overlapping_rows[:5].tolist()
        raise SafetyError(
            f"pyinla safety check: for multiple likelihoods, each row should have data in only one response column. "
            f"Found {len(overlapping_rows)} rows with values in multiple response columns (showing first 5): {show_rows}. "
            f"Use NaN to mark which rows belong to which likelihood."
        )


def enforce_survival_response(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    from ..surv import is_inla_surv

    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))
    survival_fams = {
        "exponentialsurv", "gammasurv", "lognormalsurv", "weibullsurv", "loglogisticsurv",
        "mgammasurv", "qloglogisticsurv", "gompertzsurv", "dgompertzsurv", "fmrisurv", "coxph"
    }
    if not any(f in survival_fams for f in normalized):
        return

    response = _get_response_object(kwargs)
    if response is None:
        raise SafetyError(
            "pyinla safety check: survival likelihoods require the response to be built via pyinla.surv.inla_surv(...)."
        )
    if not is_inla_surv(response):
        raise SafetyError(
            "pyinla safety check: survival likelihoods require the response to be an inla_surv(...) object."
        )

    try:
        import numpy as np  # type: ignore

        if not isinstance(response, dict):
            return

        time_vals = response.get("time")
        lower_vals = response.get("lower")
        upper_vals = response.get("upper")
        event_vals = response.get("event")

        for name, vals in [("time", time_vals), ("lower", lower_vals), ("upper", upper_vals)]:
            if vals is not None:
                arr = np.asarray(vals, dtype=float).reshape(-1)
                mask_invalid = (~np.isfinite(arr)) | (arr < 0.0)
                invalid = arr[mask_invalid]
                if invalid.size > 0:
                    raise SafetyError(
                        f"pyinla safety check: survival '{name}' values must be non-negative and finite; "
                        f"found invalid entries (showing up to 5): {invalid[:5].tolist()}"
                    )

        if event_vals is not None:
            event_arr = np.asarray(event_vals, dtype=float).reshape(-1)
            valid_events = {0.0, 1.0, 2.0, 3.0, 4.0}
            mask_invalid_event = (~np.isfinite(event_arr)) | (~np.isin(event_arr, list(valid_events)))
            invalid_events = event_arr[mask_invalid_event]
            if invalid_events.size > 0:
                raise SafetyError(
                    "pyinla safety check: survival event indicators must be in {0, 1, 2, 3, 4}; "
                    f"found invalid entries (showing up to 5): {invalid_events[:5].tolist()}"
                )
    except SafetyError:
        raise
    except Exception:
        pass


def enforce_untested_arguments(kwargs: Dict[str, Any]) -> None:
    """Block top-level arguments that have not been tested for input file parity with R-INLA."""

    untested_observation = {
        "strata": "stratification specification",
        "lp_scale": "linear predictor scale",
        "link_covariates": "link function covariates",
    }

    for key, desc in untested_observation.items():
        if key in kwargs and kwargs[key] is not None:
            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    untested_output = {
        "selection": "selection specification",
    }

    for key, desc in untested_output.items():
        if key in kwargs and kwargs[key] is not None:
            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    untested_execution = {
        "silent": "silent mode",
        "inla_arg": "custom INLA arguments",
        "safe": "safe mode retry behavior",
        "debug": "debug mode",
        "dry_run": "dry run mode",
        "collect": "result collection flag",
    }

    for key, desc in untested_execution.items():
        if key in kwargs and kwargs[key] is not None:
            if key == "safe" and kwargs[key] is True:
                continue
            if key == "collect" and kwargs[key] is True:
                continue

            raise SafetyError(
                f"pyinla safety check: '{key}' ({desc}) is not yet tested for input file parity with R-INLA. "
                "This feature will be enabled once testing coverage is complete."
            )

    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        default_quantiles = (0.025, 0.5, 0.975)
        try:
            user_q = tuple(float(q) for q in quantiles)
        except (TypeError, ValueError):
            user_q = None
        if user_q is not None and user_q != default_quantiles:
            raise SafetyError(
                "pyinla safety check: custom 'quantiles' values are not yet tested for input file parity with R-INLA. "
                f"Please use the default quantiles {default_quantiles} or omit the argument."
            )
