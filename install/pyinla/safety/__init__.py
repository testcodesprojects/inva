"""Safety gate helpers for pyinla public entry points.

This package provides input validation and safety checks organized into modules:

- errors: SafetyError exception class
- expression: Expression prior validation (muparser mini-language)
- utils: Utility functions for validation
- family: Family normalization and validation
- hyperstructure: Hyperparameter structure validation
- support: Response support validation for different likelihoods
- control: Control structure validation
- random: Random effect structure validation
- response: Response extraction and validation
- exposure: Exposure, trials, scale, and weights validation
- family_variant: Family variant and link validation
- capture: User call capture and orchestration
"""

from .errors import SafetyError

from .capture import capture_user_call

from .family import (
    enforce_allowed_family,
    _normalize_family_spec,
    _normalize_link_value,
)

from .hyperstructure import (
    enforce_gaussian_hyperstructure,
    enforce_gamma_hyperstructure,
    enforce_beta_hyperstructure,
    enforce_logistic_hyperstructure,
    enforce_loglogistic_hyperstructure,
    enforce_sn_hyperstructure,
    enforce_t_hyperstructure,
)

from .support import (
    enforce_gamma_support,
    enforce_beta_support,
    enforce_poisson_support,
    enforce_nbinomial_support,
    enforce_binomial_support,
    enforce_exponential_support,
    enforce_lognormal_support,
    enforce_weibull_support,
    enforce_loglogistic_support,
    enforce_gaussian_support,
    enforce_logistic_support,
    enforce_t_support,
    enforce_sn_support,
)

from .control import (
    enforce_control_structure,
    enforce_compute_section,
)

from .random import enforce_random_structure

from .response import (
    enforce_multiple_likelihood_response,
    enforce_survival_response,
    enforce_untested_arguments,
)

from .exposure import (
    enforce_scale_usage,
    enforce_exposure_usage,
    enforce_poisson_exposure,
    enforce_nbinomial_exposure,
    enforce_binomial_trials,
    enforce_offset_values,
    enforce_weights_values,
    enforce_random_weights_values,
)

from .family_variant import enforce_binomial_family_variant


__all__ = [
    "SafetyError",
    "capture_user_call",
    "enforce_allowed_family",
    "enforce_gaussian_hyperstructure",
    "enforce_gamma_hyperstructure",
    "enforce_gamma_support",
    "enforce_logistic_hyperstructure",
    "enforce_sn_hyperstructure",
    "enforce_beta_hyperstructure",
    "enforce_t_hyperstructure",
    "enforce_scale_usage",
    "enforce_compute_section",
    "enforce_exposure_usage",
    "enforce_control_structure",
    "enforce_random_structure",
    "enforce_poisson_exposure",
    "enforce_nbinomial_exposure",
    "enforce_binomial_trials",
    "enforce_binomial_family_variant",
    "enforce_multiple_likelihood_response",
    "enforce_beta_support",
    "enforce_survival_response",
    "enforce_untested_arguments",
    "enforce_poisson_support",
    "enforce_nbinomial_support",
    "enforce_binomial_support",
    "enforce_exponential_support",
    "enforce_lognormal_support",
    "enforce_weibull_support",
    "enforce_loglogistic_support",
    "enforce_gaussian_support",
    "enforce_logistic_support",
    "enforce_t_support",
    "enforce_sn_support",
    "enforce_offset_values",
    "enforce_weights_values",
    "enforce_random_weights_values",
]
