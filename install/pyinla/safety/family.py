"""Family normalization and validation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from .errors import SafetyError


def _normalize_family_spec(family: Any) -> Iterable[str]:
    if family is None:
        return []
    if isinstance(family, (list, tuple, set)):
        items = family
    else:
        items = [family]
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        normalized.append(str(item).strip().lower())
    return normalized


def _normalize_link_value(link_val: Any) -> str | None:
    """Normalize a link value to a lowercase string.

    Handles both formats:
    - String format: 'neglog' -> 'neglog'
    - Dict format: {'model': 'neglog'} -> 'neglog'
    """
    if link_val is None:
        return None
    if isinstance(link_val, dict):
        # Extract 'model' key if present
        model_val = link_val.get("model")
        if model_val is not None:
            return str(model_val).strip().lower()
        return None
    return str(link_val).strip().lower()


def enforce_allowed_family(
    kwargs: Dict[str, Any], *, allowed: Iterable[str] = (
        "gaussian", "poisson", "binomial", "xbinomial", "gamma", "nbinomial", "nbinomial2", "beta",
        "betabinomial", "exponential", "exponentialsurv", "gammasurv", "lognormal",
        "lognormalsurv", "logistic", "loglogistic", "loglogisticsurv", "sn", "t",
        "weibull", "weibullsurv"
    )
) -> Tuple[str, ...]:
    """Ensure the caller supplied an explicit family drawn from the allow-list."""

    allowed_set = {str(a).strip().lower() for a in allowed}

    if "family" not in kwargs:
        raise SafetyError(
            "pyinla safety check: please pass the 'family' argument explicitly."
        )

    normalized = list(_normalize_family_spec(kwargs.get("family")))
    if not normalized:
        raise SafetyError(
            "pyinla safety check: 'family' may not be empty."
        )

    invalid = [fam for fam in normalized if fam not in allowed_set]
    if invalid:
        raise SafetyError(
            "pyinla safety check: unsupported family requested: {}. Only {} allowed.".format(
                ", ".join(sorted(set(invalid))), ", ".join(sorted(allowed_set))
            )
        )

    return tuple(normalized)
