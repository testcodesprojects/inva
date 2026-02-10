"""User call capture and orchestration of all safety checks."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .control import enforce_compute_section, enforce_control_structure
from .exposure import (
    enforce_binomial_trials,
    enforce_exposure_usage,
    enforce_nbinomial_exposure,
    enforce_offset_values,
    enforce_poisson_exposure,
    enforce_scale_usage,
    enforce_weights_values,
)
from .family import enforce_allowed_family
from .family_variant import enforce_binomial_family_variant
from .hyperstructure import (
    enforce_beta_hyperstructure,
    enforce_gamma_hyperstructure,
    enforce_gaussian_hyperstructure,
    enforce_logistic_hyperstructure,
    enforce_loglogistic_hyperstructure,
    enforce_sn_hyperstructure,
    enforce_t_hyperstructure,
)
from .random import enforce_random_structure
from .response import (
    enforce_multiple_likelihood_response,
    enforce_survival_response,
    enforce_untested_arguments,
)
from .support import (
    enforce_beta_support,
    enforce_binomial_support,
    enforce_exponential_support,
    enforce_gamma_support,
    enforce_gaussian_support,
    enforce_lognormal_support,
    enforce_logistic_support,
    enforce_loglogistic_support,
    enforce_nbinomial_support,
    enforce_poisson_support,
    enforce_sn_support,
    enforce_t_support,
    enforce_weibull_support,
)


def capture_user_call(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot the raw arguments supplied by the user.

    Parameters
    ----------
    args
        Positional arguments as received by the public entry point.
    kwargs
        Keyword arguments as received by the public entry point.

    Returns
    -------
    dict
        A shallow copy of the keyword arguments for downstream use.
    """

    snapshot = dict(kwargs or {})
    snapshot["__args__"] = tuple(args or ())

    families = enforce_allowed_family(snapshot)
    snapshot["__families__"] = tuple(families)
    enforce_gaussian_hyperstructure(snapshot, families=families)
    enforce_gamma_hyperstructure(snapshot, families=families)
    enforce_gamma_support(snapshot, families=families)
    enforce_logistic_hyperstructure(snapshot, families=families)
    enforce_loglogistic_hyperstructure(snapshot, families=families)
    enforce_sn_hyperstructure(snapshot, families=families)
    enforce_t_hyperstructure(snapshot, families=families)
    enforce_beta_hyperstructure(snapshot, families=families)
    enforce_scale_usage(snapshot, families=families)
    enforce_control_structure(snapshot, families=families)
    enforce_random_structure(snapshot)
    enforce_compute_section(snapshot)
    enforce_exposure_usage(snapshot, families=families)
    enforce_poisson_exposure(snapshot, families=families)
    enforce_nbinomial_exposure(snapshot, families=families)
    enforce_binomial_trials(snapshot, families=families)
    enforce_binomial_family_variant(snapshot, families=families)
    enforce_multiple_likelihood_response(snapshot, families=families)
    enforce_beta_support(snapshot, families=families)
    enforce_survival_response(snapshot, families=families)
    enforce_poisson_support(snapshot, families=families)
    enforce_nbinomial_support(snapshot, families=families)
    enforce_binomial_support(snapshot, families=families)
    enforce_exponential_support(snapshot, families=families)
    enforce_lognormal_support(snapshot, families=families)
    enforce_weibull_support(snapshot, families=families)
    enforce_loglogistic_support(snapshot, families=families)
    enforce_gaussian_support(snapshot, families=families)
    enforce_logistic_support(snapshot, families=families)
    enforce_t_support(snapshot, families=families)
    enforce_sn_support(snapshot, families=families)
    enforce_offset_values(snapshot)
    enforce_weights_values(snapshot)
    enforce_untested_arguments(snapshot)

    return snapshot
