"""Control structure validation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .errors import SafetyError
from .family import _normalize_family_spec, _normalize_link_value


def enforce_control_structure(kwargs: Dict[str, Any], families: Tuple[str, ...] = ()) -> None:
    """Allow only ``family`` and ``predictor`` keys in control, and constrain predictor."""

    control = kwargs.get("control")
    if control is None:
        return
    if not isinstance(control, dict):
        raise SafetyError("pyinla safety check: 'control' must be a dict when provided.")

    allowed_top = {
        "family",
        "predictor",
        "compute",
        "inla",
        "fixed",
        "mode",
        "expert",
        "hazard",
        "lincomb",
        "update",
        "lp_scale",
        "pardiso",
        "stiles",
        "taucs",
        "numa",
        "only_hyperparam",
    }
    extra = set(control.keys()) - allowed_top
    if extra:
        raise SafetyError(
            "pyinla safety check: unsupported keys in 'control': {}.".format(
                ", ".join(sorted(extra))
            )
        )

    family_block = control.get("family")
    if family_block is not None and not isinstance(family_block, (dict, list)):
        raise SafetyError("pyinla safety check: control['family'] must be a dict (or list of dicts for multiple families).")

    def _validate_control_link(block: Dict[str, Any]) -> None:
        if not any(f in ("gamma", "gammasurv") for f in families):
            raise SafetyError("pyinla safety check: control.link is only allowed for gamma likelihoods.")
        if not isinstance(block, dict):
            raise SafetyError("pyinla safety check: control.link configuration must be a dict.")
        model = block.get("model")
        if model is None:
            raise SafetyError("pyinla safety check: control.link['model'] must be specified.")
        model_norm = str(model).strip().lower()
        is_survival = any(f == "gammasurv" for f in families)
        if is_survival:
            if model_norm not in ("log", "neglog", "quantile"):
                raise SafetyError("pyinla safety check: only 'log', 'neglog', or 'quantile' control.link model is allowed for gammasurv.")
            if model_norm in ("log", "neglog"):
                return
        else:
            if model_norm != "quantile":
                raise SafetyError("pyinla safety check: only 'quantile' control.link model is allowed for gamma.")
        quantile = block.get("quantile")
        if quantile is None:
            raise SafetyError("pyinla safety check: control.link['quantile'] must be provided when model='quantile'.")
        try:
            q = float(quantile)
        except Exception as exc:
            raise SafetyError("pyinla safety check: quantile must be numeric.") from exc
        if not (0.0 < q < 1.0):
            raise SafetyError("pyinla safety check: quantile must be in (0,1).")

    if isinstance(family_block, dict):
        link_conf = family_block.get("control.link") or family_block.get("control_link")
        if link_conf is not None:
            _validate_control_link(link_conf)

    predictor_block = control.get("predictor")
    if predictor_block is None:
        pass
    elif not isinstance(predictor_block, dict):
        raise SafetyError("pyinla safety check: control['predictor'] must be a dict.")
    else:
        allowed_predictor = {"compute", "link"}
        if "control.link" in predictor_block or "control_link" in predictor_block:
            _validate_control_link(predictor_block.get("control.link") or predictor_block.get("control_link"))
            allowed_predictor.add("control.link")
        extra_pred = set(predictor_block.keys()) - allowed_predictor
        if extra_pred:
            raise SafetyError(
                "pyinla safety check: unsupported keys in control['predictor']: {}.".format(
                    ", ".join(sorted(extra_pred))
                )
            )

        if "compute" in predictor_block:
            compute = predictor_block["compute"]
            if not isinstance(compute, bool):
                raise SafetyError(
                    "pyinla safety check: control['predictor']['compute'] must be True/False."
                )

    inla_block = control.get("inla")
    if inla_block is None:
        return
    if not isinstance(inla_block, dict):
        raise SafetyError("pyinla safety check: control['inla'] must be a dict when provided.")

    allowed_inla_keys = {"strategy", "int.strategy", "int_strategy", "control_vb", "adaptive_max"}
    extra_inla = set(inla_block.keys()) - allowed_inla_keys
    if extra_inla:
        raise SafetyError(
            "pyinla safety check: control['inla'] currently supports only {} overrides; found {}.".format(
                ", ".join(sorted(allowed_inla_keys)),
                ", ".join(sorted(extra_inla))
            )
        )

    control_vb = inla_block.get("control_vb")
    if control_vb is not None:
        if not isinstance(control_vb, dict):
            raise SafetyError("pyinla safety check: control['inla']['control_vb'] must be a dict.")
        allowed_vb_keys = {"enable", "strategy", "iter_max", "hessian_update", "hessian_strategy",
                          "verbose", "f_enable_limit", "emergency"}
        extra_vb = set(control_vb.keys()) - allowed_vb_keys
        if extra_vb:
            raise SafetyError(
                "pyinla safety check: control['inla']['control_vb'] supports only {} keys; found {}.".format(
                    ", ".join(sorted(allowed_vb_keys)),
                    ", ".join(sorted(extra_vb))
                )
            )
        vb_strategy = control_vb.get("strategy")
        if vb_strategy is not None:
            vb_strategy_str = str(vb_strategy).strip().lower()
            if vb_strategy_str not in {"mean", "variance"}:
                raise SafetyError(
                    "pyinla safety check: control['inla']['control_vb']['strategy'] must be 'mean' or 'variance'; got '{}'.".format(vb_strategy)
                )
        vb_hessian = control_vb.get("hessian_strategy")
        if vb_hessian is not None:
            vb_hessian_str = str(vb_hessian).strip().lower()
            if vb_hessian_str not in {"default", "full", "partial", "diagonal"}:
                raise SafetyError(
                    "pyinla safety check: control['inla']['control_vb']['hessian_strategy'] must be one of 'default', 'full', 'partial', 'diagonal'; got '{}'.".format(vb_hessian)
                )

    def _normalize_choice(name: str, value: Any, allowed: set[str]) -> None:
        try:
            normalized = str(value).strip().lower().replace("_", ".")
        except Exception as exc:
            raise SafetyError(
                "pyinla safety check: control['inla']['{}'] must be a string.".format(name)
            ) from exc
        if normalized not in allowed:
            raise SafetyError(
                "pyinla safety check: control['inla']['{}'] must be one of {}; got '{}'.".format(
                    name,
                    ", ".join(sorted(allowed)),
                    value,
                )
            )

    strategy_val = inla_block.get("strategy")
    if strategy_val is not None:
        _normalize_choice("strategy", strategy_val, {"auto", "gaussian", "simplified.laplace", "laplace", "adaptive"})

    int_strategy_val = inla_block.get("int.strategy")
    if int_strategy_val is None and "int_strategy" in inla_block:
        int_strategy_val = inla_block.get("int_strategy")
    if int_strategy_val is not None:
        _normalize_choice(
            "int.strategy",
            int_strategy_val,
            {"auto", "ccd", "grid", "eb", "user", "user.std"},
        )


def enforce_compute_section(kwargs: Dict[str, Any]) -> None:
    control = kwargs.get("control") or {}
    if not isinstance(control, dict):
        return
    compute = control.get("compute")
    if compute is None:
        return
    if not isinstance(compute, dict):
        raise SafetyError("pyinla safety check: control['compute'] must be a dict if provided.")

    _allowed_compute_keys = {
        "config",
        "dic",
        "cpo",
        "mlik",
        "return_marginals",
        "return_marginals_predictor",
        "waic",
        "po",
    }

    allowed_keys = _allowed_compute_keys
    extra = set(compute.keys()) - allowed_keys
    if extra:
        raise SafetyError(
            "pyinla safety check: unsupported keys in control['compute']: {}. "
            "Allowed: {}.".format(
                ", ".join(sorted(extra)),
                ", ".join(sorted(allowed_keys))
            )
        )

    for key in allowed_keys.intersection(compute.keys()):
        val = compute[key]
        if isinstance(val, bool):
            continue
        if isinstance(val, int) and val in (0, 1):
            continue
        raise SafetyError(
            "pyinla safety check: control['compute']['{}'] must be boolean.".format(key)
        )
