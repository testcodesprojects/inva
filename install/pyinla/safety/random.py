"""Random effect structure validation."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .errors import SafetyError

try:
    import scipy.sparse as sp  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    sp = None  # type: ignore
    _HAVE_SCIPY = False


def enforce_random_structure(kwargs: Dict[str, Any]) -> None:
    model = kwargs.get("model") or kwargs.get("formula")
    if not isinstance(model, dict):
        return
    random_block = model.get("random")
    if random_block is None:
        return

    entries: List[Dict[str, Any]] = []

    def ensure_entry(entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            raise SafetyError("pyinla safety check: random effect specifications must be dicts.")
        return dict(entry)

    if isinstance(random_block, dict) and "model" not in random_block:
        for label, spec in random_block.items():
            spec_dict = ensure_entry(spec)
            if spec_dict.get("id") is None and spec_dict.get("covariate") is None:
                spec_dict["id"] = label
            entries.append(spec_dict)
    else:
        iterable = random_block if isinstance(random_block, (list, tuple)) else [random_block]
        for spec in iterable:
            entries.append(ensure_entry(spec))

    if not entries:
        raise SafetyError("pyinla safety check: empty random-effect specification is not allowed.")

    def _is_spde_object(obj):
        """Check if object is an SPDE model from fmesher."""
        return type(obj).__name__ == "SPDE2PcMatern"

    allowed_spde_keys = {"id", "model", "A.local", "__spde_object__"}

    for spec in entries:
        model_val = spec.get("model")
        if _is_spde_object(model_val) or spec.get("__spde_object__"):
            unknown_keys = set(spec.keys()) - allowed_spde_keys
            if unknown_keys:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects only allow keys {{id, model, A.local}}; "
                    "found unexpected: {}.".format(", ".join(sorted(unknown_keys)))
                )

            if "id" not in spec:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects require an 'id' key."
                )

            if "A.local" not in spec:
                raise SafetyError(
                    "pyinla safety check: SPDE random effects require an 'A.local' projection matrix."
                )

            A_local = spec.get("A.local")
            is_valid_A = False
            try:
                import numpy as _np_local
                if isinstance(A_local, _np_local.ndarray) and A_local.ndim == 2:
                    is_valid_A = True
            except ImportError:
                pass
            if not is_valid_A and _HAVE_SCIPY and sp.issparse(A_local):
                is_valid_A = True

            if not is_valid_A:
                raise SafetyError(
                    "pyinla safety check: SPDE 'A.local' must be a 2D array or sparse matrix."
                )

            spec["__spde_object__"] = True

    allowed_models = {"iid", "linear", "clinear", "z", "generic0", "generic1", "generic2", "iidkd", "rw1", "rw2", "seasonal", "ar1", "ar", "besag", "bym", "bym2"}

    _common_optional = {"id_names", "weights"}
    allowed_keys_per_model = {
        "iid": {"id", "model", "hyper", "n", "constr", "initial", "fixed", "diagonal"} | _common_optional,
        "iidkd": {"id", "model", "hyper", "order", "n", "constr", "diagonal"} | _common_optional,
        "rw1": {"id", "model", "hyper", "n", "constr", "cyclic", "scale.model", "values", "diagonal"} | _common_optional,
        "rw2": {"id", "model", "hyper", "n", "constr", "cyclic", "scale.model", "values", "diagonal"} | _common_optional,
        "linear": {"id", "model", "mean.linear", "prec.linear", "diagonal", "covariate"} | _common_optional,
        "clinear": {"id", "model", "hyper", "range", "diagonal", "covariate"} | _common_optional,
        "z": {"id", "model", "hyper", "Z", "Cmatrix", "precision", "constr", "extraconstr", "diagonal"} | _common_optional,
        "generic0": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"} | _common_optional,
        "generic1": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"} | _common_optional,
        "generic2": {"id", "model", "hyper", "Cmatrix", "n", "constr", "extraconstr", "diagonal"} | _common_optional,
        "seasonal": {"id", "model", "hyper", "n", "constr", "scale.model", "values", "extraconstr", "diagonal", "season.length"} | _common_optional,
        "ar1": {"id", "model", "hyper", "constr", "cyclic", "values", "extraconstr", "diagonal"} | _common_optional,
        "ar": {"id", "model", "hyper", "order", "constr", "values", "extraconstr", "diagonal"} | _common_optional,
        "besag": {"id", "model", "hyper", "graph", "n", "constr", "scale.model", "adjust.for.con.comp", "diagonal", "rankdef", "values", "nrep", "replicate", "ngroup", "group", "control.group", "compute", "vb.correct"} | _common_optional,
        "bym": {"id", "model", "hyper", "graph", "n", "constr", "scale.model", "adjust.for.con.comp", "extraconstr", "diagonal", "rankdef", "values", "nrep", "replicate", "ngroup", "group", "control.group", "compute", "vb.correct"} | _common_optional,
        "bym2": {"id", "model", "hyper", "graph", "n", "constr", "scale.model", "extraconstr", "diagonal", "rankdef", "values", "nrep", "replicate", "ngroup", "group", "control.group", "compute", "vb.correct"} | _common_optional,
    }

    for spec in entries:
        if spec.get("__spde_object__"):
            continue

        model_name = str(spec.get("model", "")).strip().lower()
        if model_name not in allowed_models:
            raise SafetyError(
                "pyinla safety check: random effects currently support only model in {}.".format(
                    ", ".join(sorted(allowed_models))
                )
            )

        allowed_keys = allowed_keys_per_model.get(model_name, set())
        unknown_keys = set(spec.keys()) - allowed_keys
        if unknown_keys:
            raise SafetyError(
                "pyinla safety check: unknown key(s) {} for '{}' random effect. Allowed keys: {}.".format(
                    ", ".join(sorted(unknown_keys)),
                    model_name,
                    ", ".join(sorted(allowed_keys))
                )
            )

        ident = spec.get("id") or spec.get("covariate")
        if ident is None:
            raise SafetyError(
                "pyinla safety check: random effects must include an 'id' (or 'covariate' for linear/clinear models)."
            )
        spec.setdefault("id", ident)

        if model_name == "linear":
            allowed_dot_fields = {"mean.linear", "prec.linear"}
            dot_fields = [key for key in spec.keys() if "." in key]
            extra = [key for key in dot_fields if key not in allowed_dot_fields]
            if extra:
                raise SafetyError(
                    "pyinla safety check: linear random effects only allow {} overrides; invalid fields: {}."
                    .format(
                        ", ".join(sorted(allowed_dot_fields)),
                        ", ".join(sorted(extra))
                    )
                )
        elif model_name == "clinear":
            dot_fields = [key for key in spec.keys() if "." in key]
            if dot_fields:
                raise SafetyError(
                    "pyinla safety check: clinear does not support dot-style options like mean.linear/prec.linear; "
                    "use 'hyper' to configure the prior on theta. Found: {}.".format(
                        ", ".join(sorted(dot_fields))
                    )
                )

        if model_name == "clinear":
            rng = spec.get("range")
            if rng is not None:
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' must be a two-value sequence (low, high)."
                    )
                low, high = rng
                try:
                    low = float(low)
                    high = float(high)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' entries must be numeric."
                    ) from exc
                if not (low < high) or math.isnan(low) or math.isnan(high):
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' requires low < high (NaN not allowed)."
                    )
                low_inf = math.isinf(low) and low < 0
                high_finite = math.isfinite(high)
                if low_inf and high_finite:
                    raise SafetyError(
                        "pyinla safety check: clinear 'range' with (-Inf, high) where high is finite "
                        "is not supported by R-INLA. Use (low, Inf), (low, high) both finite, or (-Inf, Inf)."
                    )

        if model_name == "z":
            Z = spec.get("Z")
            if Z is None:
                raise SafetyError(
                    "pyinla safety check: z model requires 'Z' design matrix."
                )
            if _HAVE_SCIPY and sp.issparse(Z):
                Z_arr = Z
            else:
                try:
                    import numpy as np  # type: ignore
                    Z_arr = np.asarray(Z)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'Z' must be convertible to a numpy array."
                    ) from exc
            if Z_arr.ndim != 2:
                raise SafetyError(
                    "pyinla safety check: z model 'Z' must be a 2D matrix (n_obs × n_random_effects)."
                )

            Cmatrix = spec.get("Cmatrix")
            if Cmatrix is not None:
                if _HAVE_SCIPY and sp.issparse(Cmatrix):
                    C_arr = Cmatrix
                else:
                    try:
                        import numpy as np  # type: ignore
                        C_arr = np.asarray(Cmatrix)
                    except Exception as exc:
                        raise SafetyError(
                            "pyinla safety check: z model 'Cmatrix' must be convertible to a numpy array."
                        ) from exc
                if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: z model 'Cmatrix' must be square (m × m)."
                    )
                if C_arr.shape[0] != Z_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: z model 'Cmatrix' dimension must match number of columns in 'Z'."
                    )

            precision_val = spec.get("precision")
            if precision_val is not None:
                try:
                    precision_float = float(precision_val)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'precision' must be numeric."
                    ) from exc
                if not math.isfinite(precision_float) or precision_float <= 0.0:
                    raise SafetyError(
                        "pyinla safety check: z model 'precision' must be a positive, finite number."
                    )

            extraconstr = spec.get("extraconstr")
            if extraconstr is not None:
                if not isinstance(extraconstr, dict):
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' must be a dict with keys 'A' and 'e'."
                    )
                allowed_constr_keys = {"A", "e"}
                extra_keys = set(extraconstr.keys()) - allowed_constr_keys
                if extra_keys:
                    raise SafetyError(
                        f"pyinla safety check: z model 'extraconstr' only allows keys 'A' and 'e'; "
                        f"found {', '.join(sorted(extra_keys))}."
                    )
                A_constr = extraconstr.get("A")
                e_constr = extraconstr.get("e")
                if A_constr is None:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' requires 'A' matrix."
                    )
                try:
                    import numpy as np  # type: ignore
                    A_arr = np.asarray(A_constr)
                except Exception as exc:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' 'A' must be convertible to a numpy array."
                    ) from exc
                if A_arr.ndim == 1:
                    A_arr = A_arr.reshape(1, -1)
                if A_arr.ndim != 2:
                    raise SafetyError(
                        "pyinla safety check: z model 'extraconstr' 'A' must be a 1D or 2D array."
                    )
                n_obs, m_effects = Z_arr.shape
                expected_cols = n_obs + m_effects
                if A_arr.shape[1] != expected_cols:
                    raise SafetyError(
                        f"pyinla safety check: z model 'extraconstr' 'A' must have {expected_cols} columns "
                        f"(n={n_obs} + m={m_effects}) to match the augmented latent field; got {A_arr.shape[1]}."
                    )
                if e_constr is not None:
                    try:
                        import numpy as np  # type: ignore
                        e_arr = np.asarray(e_constr)
                    except Exception as exc:
                        raise SafetyError(
                            "pyinla safety check: z model 'extraconstr' 'e' must be convertible to a numpy array."
                        ) from exc
                    e_arr = np.atleast_1d(e_arr)
                    if e_arr.shape[0] != A_arr.shape[0]:
                        raise SafetyError(
                            f"pyinla safety check: z model 'extraconstr' 'e' length ({e_arr.shape[0]}) "
                            f"must match number of rows in 'A' ({A_arr.shape[0]})."
                        )

        if model_name == "seasonal":
            season_length = spec.get("season.length")
            if season_length is None:
                raise SafetyError(
                    "pyinla safety check: seasonal model requires 'season.length' parameter "
                    "(the periodicity m of the seasonal component)."
                )
            try:
                season_length_int = int(season_length)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: seasonal 'season.length' must be a positive integer."
                ) from exc
            if season_length_int < 2:
                raise SafetyError(
                    "pyinla safety check: seasonal 'season.length' must be at least 2."
                )

        if model_name == "ar":
            order = spec.get("order")
            if order is None:
                raise SafetyError(
                    "pyinla safety check: ar model requires 'order' parameter "
                    "(the autoregressive order p, must be 1-10)."
                )
            try:
                order_int = int(order)
            except (TypeError, ValueError) as exc:
                raise SafetyError(
                    "pyinla safety check: ar 'order' must be a positive integer (1-10)."
                ) from exc
            if order_int < 1 or order_int > 10:
                raise SafetyError(
                    "pyinla safety check: ar 'order' must be between 1 and 10 (inclusive); got {}.".format(order_int)
                )

        hyper = spec.get("hyper")
        if model_name in {"generic", "generic0", "generic1", "generic2"}:
            Cmatrix = spec.get("Cmatrix")
            if Cmatrix is None:
                raise SafetyError(
                    "pyinla safety check: generic/generic0/1/2 models require 'Cmatrix'."
                )
            try:
                import scipy.sparse as sp_check  # type: ignore

                if sp_check.issparse(Cmatrix):
                    shape = Cmatrix.shape
                    if len(shape) != 2 or shape[0] != shape[1]:
                        raise SafetyError(
                            "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                        )
                else:
                    import numpy as np  # type: ignore

                    C_arr = np.asarray(Cmatrix)
                    if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                        raise SafetyError(
                            "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                        )
            except ImportError:
                import numpy as np  # type: ignore

                C_arr = np.asarray(Cmatrix)
                if C_arr.ndim != 2 or C_arr.shape[0] != C_arr.shape[1]:
                    raise SafetyError(
                        "pyinla safety check: generic* model 'Cmatrix' must be a square matrix."
                    )

        if hyper is not None:
            if model_name == "linear":
                raise SafetyError(
                    "pyinla safety check: linear model has no hyperparameters. "
                    "Use 'mean.linear' and 'prec.linear' to specify the prior on the slope."
                )
            expected_hyper_count = {
                "iid": 1,
                "clinear": 1,
                "z": 1,
                "generic": 1,
                "generic0": 1,
                "generic1": 2,
                "generic2": 2,
                "iidkd": None,
                "rw1": 1,
                "rw2": 1,
                "seasonal": 1,
                "ar1": (1, 3),
                "ar": (1, 11),
                "besag": 1,
                "bym": 2,
                "bym2": (1, 2),
            }
            expected = expected_hyper_count.get(model_name, 1)

            hyper_name_mapping = {
                "iid": {"prec": 1, "precision": 1},
                "clinear": {"beta": 1, "b": 1},
                "z": {"prec": 1, "precision": 1},
                "generic": {"prec": 1, "precision": 1},
                "generic0": {"prec": 1, "precision": 1},
                "generic1": {"prec": 1, "precision": 1, "beta": 2, "b": 2},
                "generic2": {"prec": 1, "precision": 1, "prec.random": 2},
                "rw1": {"prec": 1, "precision": 1},
                "rw2": {"prec": 1, "precision": 1},
                "seasonal": {"prec": 1, "precision": 1},
                "ar1": {"prec": 1, "precision": 1, "rho": 2, "mean": 3},
                "ar": {
                    "prec": 1, "precision": 1,
                    "pacf1": 2, "pacf2": 3, "pacf3": 4, "pacf4": 5, "pacf5": 6,
                    "pacf6": 7, "pacf7": 8, "pacf8": 9, "pacf9": 10, "pacf10": 11,
                },
                "besag": {"prec": 1, "precision": 1},
                "bym": {"prec": 1, "precision": 1, "prec.iid": 2},
                "bym2": {"prec": 1, "precision": 1, "phi": 2},
            }

            if isinstance(hyper, dict):
                name_map = hyper_name_mapping.get(model_name, {})
                max_theta = 0
                for key in hyper.keys():
                    key_lower = key.lower()
                    if key_lower in name_map:
                        max_theta = max(max_theta, name_map[key_lower])
                    elif key_lower.startswith("theta"):
                        try:
                            idx = int(key_lower[5:]) if len(key_lower) > 5 else 1
                            max_theta = max(max_theta, idx)
                        except ValueError:
                            pass

                hyper_list = [{} for _ in range(max_theta)]

                for key, value in hyper.items():
                    key_lower = key.lower()
                    if key_lower in name_map:
                        idx = name_map[key_lower] - 1
                    elif key_lower.startswith("theta"):
                        try:
                            idx = int(key_lower[5:]) - 1 if len(key_lower) > 5 else 0
                        except ValueError:
                            raise SafetyError(
                                "pyinla safety check: {} model 'hyper' has unknown key '{}'. "
                                "Use semantic names like {} or positional names like 'theta1', 'theta2', etc.".format(
                                    model_name,
                                    key,
                                    ", ".join(f"'{k}'" for k in name_map.keys()) if name_map else "'theta1'"
                                )
                            )
                    else:
                        raise SafetyError(
                            "pyinla safety check: {} model 'hyper' has unknown key '{}'. "
                            "Use semantic names like {} or positional names like 'theta1', 'theta2', etc.".format(
                                model_name,
                                key,
                                ", ".join(f"'{k}'" for k in name_map.keys()) if name_map else "'theta1'"
                            )
                        )

                    if idx < 0 or idx >= len(hyper_list):
                        raise SafetyError(
                            "pyinla safety check: {} model 'hyper' key '{}' maps to invalid position {}.".format(
                                model_name, key, idx + 1
                            )
                        )
                    hyper_list[idx] = value

                hyper = hyper_list

            if not isinstance(hyper, list):
                raise SafetyError(
                    "pyinla safety check: {} model 'hyper' must be a list of dicts or a dict with semantic names.".format(model_name)
                )
            if expected is None:
                min_hyper = 1
                max_hyper = 299
            elif isinstance(expected, tuple):
                min_hyper, max_hyper = expected
            else:
                min_hyper = max_hyper = expected
            if len(hyper) < min_hyper or len(hyper) > max_hyper:
                if min_hyper == max_hyper:
                    raise SafetyError(
                        "pyinla safety check: {} model expects exactly {} hyperparameter entr{}; got {}.".format(
                            model_name,
                            min_hyper,
                            "y" if min_hyper == 1 else "ies",
                            len(hyper),
                        )
                    )
                else:
                    raise SafetyError(
                        "pyinla safety check: {} model expects between {} and {} hyperparameter entries; got {}.".format(
                            model_name,
                            min_hyper,
                            max_hyper,
                            len(hyper),
                        )
                    )
            allowed = {"prior", "param", "initial", "fixed"}
            validated_entries = []
            for entry in hyper:
                entry_dict = dict(entry)
                extra = set(entry_dict.keys()) - allowed
                if extra:
                    raise SafetyError(
                        "pyinla safety check: {} model 'hyper' only allows {}; found {}.".format(
                            model_name,
                            ", ".join(sorted(allowed)),
                            ", ".join(sorted(extra))
                        )
                    )
                validated_entries.append(entry_dict)
            spec["hyper"] = validated_entries

    model["random"] = entries
