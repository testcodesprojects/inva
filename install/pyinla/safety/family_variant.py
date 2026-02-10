"""Family variant and link validation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .errors import SafetyError
from .family import _normalize_family_spec, _normalize_link_value


def _validate_single_family_control(family_name: str, family_block: Dict[str, Any], idx: Optional[int] = None) -> None:
    """Validate a single control.family dict against a specific likelihood."""
    prefix = f"control['family'][{idx}]" if idx is not None else "control['family']"

    is_binomial = family_name in ("binomial", "xbinomial")
    is_nbinomial = family_name == "nbinomial"
    is_nbinomial2 = family_name == "nbinomial2"
    is_betabinomial = family_name == "betabinomial"
    is_beta = family_name == "beta"
    is_logistic = family_name == "logistic"
    is_loglogistic = family_name in ("loglogistic", "loglogisticsurv")
    is_weibull = family_name == "weibull"
    is_weibullsurv = family_name == "weibullsurv"
    is_exponential = family_name == "exponential"
    is_exponentialsurv = family_name == "exponentialsurv"
    is_lognormal = family_name == "lognormal"
    is_lognormalsurv = family_name == "lognormalsurv"
    is_sn = family_name == "sn"
    is_t = family_name == "t"
    is_gaussian = family_name == "gaussian"
    is_poisson = family_name == "poisson"
    is_gamma = family_name == "gamma"
    is_gammasurv = family_name == "gammasurv"

    allowed_keys = set()
    if is_binomial or is_betabinomial or is_nbinomial or is_nbinomial2 or is_loglogistic or is_weibull or is_weibullsurv:
        allowed_keys.add("variant")
    if is_nbinomial or is_nbinomial2 or is_betabinomial or is_beta or is_logistic or is_weibull or is_weibullsurv or is_lognormal or is_lognormalsurv or is_loglogistic or is_sn or is_t or is_gaussian or is_gamma or is_gammasurv or is_exponentialsurv:
        allowed_keys.add("hyper")
    if is_beta:
        allowed_keys.add("beta.censor.value")
    if is_binomial or is_betabinomial or is_nbinomial or is_nbinomial2 or is_beta or is_logistic or is_exponential or is_exponentialsurv or is_lognormal or is_lognormalsurv or is_loglogistic or is_weibull or is_weibullsurv or is_sn or is_t or is_gaussian or is_poisson or is_gamma or is_gammasurv:
        allowed_keys.add("link")
    if is_gamma or is_gammasurv or is_weibull or is_weibullsurv or is_nbinomial:
        allowed_keys.add("control.link")

    extra_keys = set(family_block.keys()) - allowed_keys
    if extra_keys:
        raise SafetyError(
            f"pyinla safety check: {prefix} only accepts {{{', '.join(sorted(allowed_keys))}}} for '{family_name}' likelihood; "
            f"got invalid keys: {{{', '.join(sorted(extra_keys))}}}."
        )

    if "variant" in family_block:
        if not (is_binomial or is_nbinomial or is_nbinomial2 or is_betabinomial or is_loglogistic or is_weibull or is_weibullsurv):
            raise SafetyError(
                f"pyinla safety check: {prefix}['variant'] is not valid for '{family_name}' likelihood."
            )
        try:
            variant_val = int(family_block.get("variant"))
        except Exception as exc:
            raise SafetyError(
                f"pyinla safety check: {prefix}['variant'] must be an integer (0 or 1)."
            ) from exc
        allowed_variant_values = {0, 1}
        if is_nbinomial:
            allowed_variant_values.add(2)
        if is_nbinomial2:
            allowed_variant_values = {0}
        if variant_val not in allowed_variant_values:
            raise SafetyError(
                f"pyinla safety check: {prefix}['variant'] must be one of {sorted(allowed_variant_values)} for '{family_name}'."
            )

    hyper_block = family_block.get("hyper")
    if hyper_block is not None:
        if not (is_nbinomial or is_betabinomial or is_beta or is_logistic or is_lognormal or is_lognormalsurv or is_weibull or is_weibullsurv or is_loglogistic or is_sn or is_t or is_gaussian or is_gamma or is_gammasurv or is_exponentialsurv):
            raise SafetyError(
                f"pyinla safety check: {prefix}['hyper'] is not valid for '{family_name}' likelihood."
            )
        if not isinstance(hyper_block, (dict, list)):
            raise SafetyError(
                f"pyinla safety check: {prefix}['hyper'] must be a dict or list of dicts."
            )

    if "beta.censor.value" in family_block:
        if not is_beta:
            raise SafetyError(
                f"pyinla safety check: {prefix}['beta.censor.value'] is only valid for 'beta' likelihood."
            )
        censor_val = family_block.get("beta.censor.value")
        if censor_val is not None:
            try:
                censor_float = float(censor_val)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    f"pyinla safety check: {prefix}['beta.censor.value'] must be a number."
                ) from exc
            if censor_float < 0.0 or censor_float >= 0.5:
                raise SafetyError(
                    f"pyinla safety check: {prefix}['beta.censor.value'] must be in [0, 0.5); got {censor_float}."
                )


def enforce_binomial_family_variant(
    kwargs: Dict[str, Any], *, families: Iterable[str] | None = None
) -> None:
    normalized = list(families) if families is not None else list(_normalize_family_spec(kwargs.get("family")))

    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    family_block = control.get("family")
    if family_block is None:
        return

    if isinstance(family_block, list):
        for idx, fb in enumerate(family_block):
            if not isinstance(fb, dict):
                raise SafetyError(
                    f"pyinla safety check: control['family'][{idx}] must be a dict."
                )
        if len(family_block) != len(normalized):
            raise SafetyError(
                f"pyinla safety check: control['family'] has {len(family_block)} entries but there are {len(normalized)} families."
            )
        for idx, (fam, fb) in enumerate(zip(normalized, family_block)):
            if fb:
                _validate_single_family_control(fam, fb, idx)
        return

    if not isinstance(family_block, dict):
        raise SafetyError("pyinla safety check: control['family'] must be a dict (or list of dicts for multiple families).")

    has_binomial = any(f in ("binomial", "xbinomial") for f in normalized)
    has_nbinomial = any(f == "nbinomial" for f in normalized)
    has_nbinomial2 = any(f == "nbinomial2" for f in normalized)
    has_betabinomial = any(f == "betabinomial" for f in normalized)
    has_beta = any(f == "beta" for f in normalized)
    has_logistic = any(f == "logistic" for f in normalized)
    has_loglogistic = any(f in ("loglogistic", "loglogisticsurv") for f in normalized)
    has_loglogistic_only = any(f == "loglogistic" for f in normalized)
    has_loglogisticsurv = any(f == "loglogisticsurv" for f in normalized)
    has_weibull = any(f == "weibull" for f in normalized)
    has_weibullsurv = any(f == "weibullsurv" for f in normalized)
    has_exponential = any(f == "exponential" for f in normalized)
    has_exponentialsurv = any(f == "exponentialsurv" for f in normalized)
    has_lognormal = any(f == "lognormal" for f in normalized)
    has_lognormalsurv = any(f == "lognormalsurv" for f in normalized)
    has_sn = any(f == "sn" for f in normalized)
    has_t = any(f == "t" for f in normalized)
    has_gaussian = any(f == "gaussian" for f in normalized)
    has_poisson = any(f == "poisson" for f in normalized)
    has_gamma = any(f == "gamma" for f in normalized)
    has_gammasurv = any(f == "gammasurv" for f in normalized)

    if not (
        has_binomial or has_betabinomial or has_nbinomial or has_nbinomial2 or has_beta or
        has_logistic or has_loglogistic or has_weibull or has_weibullsurv or
        has_exponential or has_exponentialsurv or has_lognormal or has_lognormalsurv or
        has_sn or has_t or has_gaussian or has_poisson or has_gamma or has_gammasurv
    ):
        return

    allowed_keys = set()
    if has_binomial or has_betabinomial or has_nbinomial or has_nbinomial2 or has_loglogistic or has_weibull or has_weibullsurv:
        allowed_keys.add("variant")
    if has_nbinomial or has_nbinomial2 or has_betabinomial or has_beta or has_logistic or has_weibull or has_weibullsurv or has_lognormal or has_lognormalsurv or has_loglogistic or has_sn or has_t or has_gaussian or has_gamma or has_gammasurv or has_exponentialsurv:
        allowed_keys.add("hyper")
    if has_beta:
        allowed_keys.add("beta.censor.value")
    if has_binomial or has_betabinomial or has_nbinomial or has_nbinomial2 or has_beta or has_logistic or has_exponential or has_exponentialsurv or has_lognormal or has_lognormalsurv or has_loglogistic or has_weibull or has_weibullsurv or has_sn or has_t or has_gaussian or has_poisson or has_gamma or has_gammasurv:
        allowed_keys.add("link")
    if has_gamma or has_gammasurv or has_weibull or has_weibullsurv or has_nbinomial:
        allowed_keys.add("control.link")
    extra_keys = set(family_block.keys()) - allowed_keys
    if extra_keys:
        raise SafetyError(
            "pyinla safety check: control['family'] only accepts {} for these likelihoods: {}.".format(
                ", ".join(sorted(allowed_keys)),
                ", ".join(sorted(extra_keys))
            )
        )

    if "variant" in family_block:
        if not (has_binomial or has_nbinomial or has_nbinomial2 or has_betabinomial or has_loglogistic or has_weibull or has_weibullsurv):
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] is only valid for binomial/xbinomial/nbinomial/nbinomial2/betabinomial/loglogistic/loglogisticsurv/weibull/weibullsurv families."
            )
        try:
            variant_val = int(family_block.get("variant"))
        except Exception as exc:
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] must be an integer (0 or 1)."
            ) from exc
        allowed_variant_values = {0, 1}
        if has_nbinomial:
            allowed_variant_values.add(2)
        if has_nbinomial2:
            allowed_variant_values = {0}
        if variant_val not in allowed_variant_values:
            raise SafetyError(
                "pyinla safety check: control['family']['variant'] must be one of {} for the requested family.".format(
                    sorted(allowed_variant_values)
                )
            )

    hyper_block = family_block.get("hyper")
    if hyper_block is not None:
        if not (has_nbinomial or has_betabinomial or has_beta or has_logistic or has_lognormal or has_lognormalsurv or has_weibull or has_weibullsurv or has_loglogistic or has_sn or has_t or has_gaussian or has_gamma or has_gammasurv or has_exponentialsurv):
            raise SafetyError(
                "pyinla safety check: control['family']['hyper'] is only allowed for gaussian/gamma/gammasurv/exponentialsurv/nbinomial/betabinomial/beta/logistic/lognormal/lognormalsurv/weibull/weibullsurv/loglogistic/loglogisticsurv/sn/t."
            )
        if not isinstance(hyper_block, (dict, list)):
            raise SafetyError(
                "pyinla safety check: control['family']['hyper'] must be a dict or list of dicts."
            )

        if "beta.censor.value" in family_block and not has_beta:
            raise SafetyError(
                "pyinla safety check: control['family']['beta.censor.value'] is only valid for the beta likelihood."
            )

    if has_beta and "beta.censor.value" in family_block:
        censor_val = family_block.get("beta.censor.value")
        if censor_val is not None:
            try:
                censor_float = float(censor_val)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: control['family']['beta.censor.value'] must be a number."
                ) from exc
            if censor_float < 0.0 or censor_float >= 0.5:
                raise SafetyError(
                    f"pyinla safety check: control['family']['beta.censor.value'] must be in [0, 0.5); got {censor_float}."
                )

    # Link validation for each family
    _validate_link_for_family(family_block, "beta", has_beta, {"default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog"})
    _validate_link_for_family(family_block, "exponential", has_exponential, {"default", "log"})
    _validate_link_for_family(family_block, "exponentialsurv", has_exponentialsurv, {"default", "log", "neglog"})
    _validate_link_for_family(family_block, "lognormal", has_lognormal, {"default", "identity"})
    _validate_link_for_family(family_block, "lognormalsurv", has_lognormalsurv, {"default", "identity"})
    _validate_link_for_family(family_block, "logistic", has_logistic, {"default", "identity"})
    _validate_link_for_family(family_block, "sn", has_sn, {"default", "identity"})
    _validate_link_for_family(family_block, "loglogistic", has_loglogistic_only, {"default", "log", "neglog"})
    _validate_link_for_family(family_block, "loglogisticsurv", has_loglogisticsurv, {"default", "log", "neglog"})
    _validate_link_for_family(family_block, "weibull", has_weibull, {"default", "log", "neglog", "quantile"})
    _validate_link_for_family(family_block, "weibullsurv", has_weibullsurv, {"default", "log", "neglog", "quantile"})
    _validate_link_for_family(family_block, "t", has_t, {"default", "identity"})
    _validate_link_for_family(family_block, "gaussian", has_gaussian, {"default", "identity", "logit", "loga", "cauchit", "log", "logoffset"})
    _validate_link_for_family(family_block, "poisson", has_poisson, {"default", "log", "logoffset", "quantile"})
    _validate_link_for_family(family_block, "binomial", has_binomial, {
        "default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog",
        "log", "sslogit", "logitoffset", "quantile", "pquantile", "robit", "sn", "powerlogit", "gevit", "cgevit"
    })
    _validate_link_for_family(family_block, "gamma", has_gamma, {"default", "log", "quantile"})
    _validate_link_for_family(family_block, "nbinomial", has_nbinomial, {"default", "log", "logoffset", "quantile"})
    _validate_link_for_family(family_block, "betabinomial", has_betabinomial, {"default", "logit", "loga", "cauchit", "probit", "cloglog", "ccloglog", "loglog", "robit", "sn"})
    _validate_link_for_family(family_block, "gammasurv", has_gammasurv, {"default", "log", "neglog", "quantile"})


def _validate_link_for_family(family_block: Dict[str, Any], family_name: str, has_family: bool, allowed_links: set[str]) -> None:
    """Helper to validate link function for a specific family."""
    if has_family and "link" in family_block:
        link_val = family_block.get("link")
        if link_val is not None:
            link_norm = _normalize_link_value(link_val)
            if link_norm is not None and link_norm not in allowed_links:
                raise SafetyError(
                    f"pyinla safety check: control['family']['link'] for {family_name} must be one of {sorted(allowed_links)}; got '{link_val}'."
                )
