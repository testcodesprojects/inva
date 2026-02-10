"""Hyperparameter structure validation for different families."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

from .errors import SafetyError
from .expression import _is_expression_prior, _validate_expression_prior
from .family import _normalize_family_spec
from .utils import _as_float_list, _is_truthy, _normalize_hyper_entries


def _validate_gaussian_family_block(family_block: Dict[str, Any]) -> None:
    allowed_top_keys = {"hyper"}
    extra_keys = set(family_block.keys()) - allowed_top_keys
    if extra_keys:
        raise SafetyError(
            "pyinla safety check: unsupported keys in control['family'] for gaussian: {}.".format(
                ", ".join(sorted(extra_keys))
            )
        )

    hyper_list = family_block.get("hyper", []) or []
    if not isinstance(hyper_list, (list, tuple)):
        raise SafetyError("pyinla safety check: control['family']['hyper'] must be a list of dicts.")

    allowed_hyper_keys = {"prior", "param", "initial", "fixed", "id"}
    allowed_priors = {None, "loggamma", "pc.prec"}

    for entry in hyper_list:
        if not isinstance(entry, dict):
            raise SafetyError("pyinla safety check: entries within control['family']['hyper'] must be dicts.")
        extra = set(entry.keys()) - allowed_hyper_keys
        if extra:
            raise SafetyError(
                "pyinla safety check: unsupported keys in gaussian hyper specification: {}.".format(
                    ", ".join(sorted(extra))
                )
            )
        prior = entry.get("prior")
        if prior is not None:
            if _is_expression_prior(prior):
                _validate_expression_prior(str(prior))
            else:
                prior_norm = str(prior).strip().lower()
                if prior_norm not in allowed_priors:
                    raise SafetyError(
                        "pyinla safety check: unsupported prior '{}' for gaussian; allowed: {}, expression:.".format(
                            prior, ", ".join(sorted(p for p in allowed_priors if p))
                        )
                    )
        entry_id = str(entry.get("id", "")).strip().lower()
        if entry_id == "precoffset":
            if prior is not None and str(prior).strip().lower() != "none":
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' prior is read-only (must remain 'none')."
                )
            if "param" in entry and entry["param"]:
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' does not accept custom 'param'."
                )
            if "fixed" in entry and not bool(entry["fixed"]):
                raise SafetyError(
                    "pyinla safety check: gaussian 'precoffset' must stay 'fixed = True'."
                )
        if prior is not None and str(prior).strip().lower() == "loggamma":
            params = entry.get("param")
            vals = _as_float_list("param", params) if params is not None else []
            if len(vals) != 2 or not all(v > 0.0 and math.isfinite(v) for v in vals):
                raise SafetyError(
                    "pyinla safety check: loggamma prior requires two positive parameters (shape, rate)."
                )


def enforce_gaussian_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "gaussian" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return
    _validate_gaussian_family_block(family_block)


def enforce_gamma_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "gamma" not in fams and "gammasurv" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: gamma/gammasurv hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError("pyinla safety check: gamma/gammasurv hyperparameters currently support only 'loggamma', 'pc.prec', or 'expression:' prior.")


def enforce_beta_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "beta" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block)
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: beta hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError("pyinla safety check: beta hyperparameters currently support only 'loggamma', 'pc.prec', or 'expression:' prior.")


def enforce_logistic_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "logistic" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: logistic hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError("pyinla safety check: logistic hyperparameters currently support only 'loggamma', 'pc.prec', or 'expression:' prior.")


def enforce_loglogistic_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if not any(f in ("loglogistic", "loglogisticsurv") for f in fams):
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["alpha"])
    if not entries:
        return

    for _, entry in entries:
        prior = entry.get("prior")
        if prior is None:
            raise SafetyError("pyinla safety check: loglogistic hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if prior_norm not in ("loggamma", "pc.prec"):
                raise SafetyError("pyinla safety check: loglogistic hyperparameters currently support only 'loggamma', 'pc.prec', or 'expression:' prior.")


def enforce_sn_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "sn" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec", "skew"])
    if not entries:
        return

    for name, entry in entries:
        is_fixed = _is_truthy(entry.get("fixed")) if entry.get("fixed") is not None else False
        prior = entry.get("prior")
        if prior is None:
            if is_fixed:
                continue
            raise SafetyError("pyinla safety check: skew-normal hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if name == "skew":
                if prior_norm not in ("pc.sn",):
                    raise SafetyError("pyinla safety check: skew parameter for skew-normal supports only 'pc.sn' or 'expression:' prior.")
            else:
                if prior_norm not in ("loggamma", "pc.prec"):
                    raise SafetyError("pyinla safety check: precision parameter for skew-normal supports only 'loggamma', 'pc.prec', or 'expression:' prior.")


def enforce_t_hyperstructure(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    fams = tuple(families) if families is not None else tuple(_normalize_family_spec(kwargs.get("family")))
    if "t" not in fams:
        return

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if not isinstance(family_block, dict):
        return

    entries = _normalize_hyper_entries(family_block, allowed_names=["prec", "dof"])
    if not entries:
        return

    for name, entry in entries:
        is_fixed = _is_truthy(entry.get("fixed")) if entry.get("fixed") is not None else False
        prior = entry.get("prior")
        if prior is None:
            if is_fixed:
                continue
            raise SafetyError("pyinla safety check: student-t hyperparameters require explicit 'prior'.")
        if _is_expression_prior(prior):
            _validate_expression_prior(str(prior))
        else:
            prior_norm = str(prior).strip().lower()
            if name == "prec":
                if prior_norm not in ("loggamma", "pc.prec"):
                    raise SafetyError(
                        "pyinla safety check: precision parameter for student-t supports only 'loggamma', 'pc.prec', or 'expression:' prior."
                    )
            else:  # dof
                if prior_norm not in ("pc.dof", "pcdof"):
                    raise SafetyError(
                        "pyinla safety check: degrees-of-freedom parameter for student-t supports only 'pc.dof' or 'expression:' prior."
                    )
