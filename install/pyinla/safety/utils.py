"""Utility functions for safety validation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .errors import SafetyError

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import scipy.sparse as sp  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAVE_SCIPY = False


def _is_truthy(value: Any) -> bool:
    """Best-effort coercion of user-supplied flags to booleans."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "t"}
    return bool(value)


def _normalize_positive_sequence(name: str, values: Any) -> Iterable[float]:
    """Return a flat list of floats, raising SafetyError on conversion issues."""
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values)
        if arr.size == 0:
            return []
        arr = arr.astype(float)
        return arr.reshape(-1).tolist()
    except Exception:
        pass

    iterable_types = (list, tuple, set)
    if isinstance(values, iterable_types):
        try:
            return [float(v) for v in values]
        except Exception as exc:
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)):
        try:
            return [float(v) for v in values]
        except Exception as exc:
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    try:
        return [float(values)]
    except Exception as exc:
        raise SafetyError(
            f"pyinla safety check: could not interpret '{name}' as numeric values."
        ) from exc


def _as_float_list(name: str, values: Any) -> Iterable[float]:
    """Coerce values to a flat list of floats, mirroring INLA expectations."""
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values)
        if arr.size == 0:
            return []
        return arr.astype(float).reshape(-1).tolist()
    except Exception:
        pass

    iterable_types = (list, tuple, set)
    if isinstance(values, iterable_types):
        try:
            return [float(v) for v in values]
        except Exception as exc:
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    if hasattr(values, "__iter__") and not isinstance(values, (str, bytes)):
        try:
            return [float(v) for v in values]
        except Exception as exc:
            raise SafetyError(
                f"pyinla safety check: could not interpret '{name}' as numeric values."
            ) from exc

    try:
        return [float(values)]
    except Exception as exc:
        raise SafetyError(
            f"pyinla safety check: could not interpret '{name}' as numeric values."
        ) from exc


def _coerce_length(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return len(_as_float_list("_probe", value))
    except SafetyError:
        pass
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    except Exception:
        pass
    try:
        return len(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _infer_observation_count(kwargs: Dict[str, Any]) -> Optional[int]:
    data = kwargs.get("data")
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    if pd is not None and isinstance(data, pd.DataFrame):
        return len(data)

    if isinstance(data, dict):
        for value in data.values():
            length = _coerce_length(value)
            if length is not None:
                return length

    length = _coerce_length(data)
    if length is not None:
        return length

    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return None

    response = model.get("response")
    if isinstance(response, str) and isinstance(data, dict) and response in data:
        return _coerce_length(data.get(response))

    if response is not None and not isinstance(response, str):
        return _coerce_length(response)

    return None


def _normalize_hyper_entries(
    family_block: Dict[str, Any], *, allowed_names: Optional[Iterable[str]] = None
) -> list[tuple[Optional[str], Dict[str, Any]]]:
    hyper = family_block.get("hyper")
    if hyper is None:
        return []

    def ensure_dict(entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            raise SafetyError(
                "pyinla safety check: hyperparameter entries must be dicts (with 'prior', 'param', etc.)."
            )
        return dict(entry)

    if isinstance(hyper, list):
        normalized = [ensure_dict(entry) for entry in hyper]
        family_block["hyper"] = normalized
        return [(None, entry) for entry in normalized]

    if isinstance(hyper, dict):
        base_fields = {"prior", "param", "initial", "fixed", "to_theta", "from_theta"}
        if base_fields & set(hyper.keys()):
            entry = ensure_dict(hyper)
            family_block["hyper"] = [entry]
            name = None
            if allowed_names:
                name = next(iter(allowed_names), None)
            return [(name, entry)]

        if not allowed_names:
            raise SafetyError(
                "pyinla safety check: provide hyperparameters as a list or map by known keys for this likelihood."
            )

        allowed_order = list(allowed_names)
        unknown = set(hyper.keys()) - set(allowed_order)
        if unknown:
            raise SafetyError(
                "pyinla safety check: unsupported hyperparameter keys {} for this likelihood.".format(
                    ", ".join(sorted(unknown))
                )
            )
        normalized: list[tuple[Optional[str], Dict[str, Any]]] = []
        normalized_dict: Dict[str, Dict[str, Any]] = {}
        for name in allowed_order:
            if name in hyper:
                entry = ensure_dict(hyper[name])
                normalized.append((name, entry))
                normalized_dict[name] = entry
        family_block["hyper"] = normalized_dict
        return normalized

    raise SafetyError(
        "pyinla safety check: control['family']['hyper'] must be a list or dict when overriding hyperparameters."
    )
