"""
Copy models: copy, scopy, clinear

Models for copying or scaling other model components.
"""

import numpy as np

# R-style transform functions for bounded parameters
_COPY_TO_THETA = (
    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
    "{<<NEWLINE>>"
    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
    "        stopifnot(low < high)<<NEWLINE>>"
    "        return(x)<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
    "        stopifnot(low < high)<<NEWLINE>>"
    "        return(log(-(low - x)/(high - x)))<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
    "        return(log(x - low))<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else {<<NEWLINE>>"
    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "}"
)

_COPY_FROM_THETA = (
    "function (x, REPLACE.ME.low, REPLACE.ME.high) <<NEWLINE>>"
    "{<<NEWLINE>>"
    "    if (all(is.infinite(c(low, high))) || low == high) {<<NEWLINE>>"
    "        stopifnot(low < high)<<NEWLINE>>"
    "        return(x)<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else if (all(is.finite(c(low, high)))) {<<NEWLINE>>"
    "        stopifnot(low < high)<<NEWLINE>>"
    "        return(low + exp(x)/(1 + exp(x)) * (high - low))<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else if (is.finite(low) && is.infinite(high) && high > low) {<<NEWLINE>>"
    "        return(low + exp(x))<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "    else {<<NEWLINE>>"
    "        stop(\"Condition not yet implemented\")<<NEWLINE>>"
    "    }<<NEWLINE>>"
    "}"
)

MODELS = {
    'copy': {
        'doc': "Create a copy of a model component",
        'hyper': {
            'theta': {
                'hyperid': 36001,
                'name': "beta",
                'short_name': "b",
                'initial': 0.0,
                'fixed': True,
                'prior': "normal",
                'param': [1, 10],
                'to.theta': _COPY_TO_THETA,
                'from.theta': _COPY_FROM_THETA
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "copy"
    },
    'scopy': {
        'doc': "Create a scaled copy of a model component",
        'hyper': {
            'theta1': {
                'hyperid': 36101,
                'name': "mean",
                'short_name': "mean",
                'initial': 1.0,
                'fixed': False,
                'prior': "normal",
                'param': [1, 10],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 36102,
                'name': "slope",
                'short_name': "slope",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 10],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            **{f'theta{i+2}': {
                'hyperid': 36102 + i,
                'name': f"spline.theta{i}",
                'short_name': f"spline{i if i > 1 else ''}",
                'initial': 0,
                'fixed': False,
                'prior': 'laplace' if i == 1 else 'none',
                'param': [0, 10] if i == 1 else [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(1, 14)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "scopy"
    },
    'clinear': {
        'doc': "Constrained linear effect",
        'hyper': {
            'theta': {
                'hyperid': 37001,
                'name': "beta",
                'short_name': "b",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [1, 10],
                'to.theta': _COPY_TO_THETA,
                'from.theta': _COPY_FROM_THETA
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "clinear"
    },
}


# Python transform functions for bounded parameters
def copy_clinear_to_theta(x, low, high):
    """Transform for 'copy'/'clinear' betas with optional bounds."""
    eps = 1e-15
    low_inf = np.isinf(low)
    high_inf = np.isinf(high)

    if (low_inf and high_inf) or (low == high):
        return x
    if np.isfinite(low) and np.isfinite(high):
        if not (low < high):
            raise ValueError("low must be less than high")
        return np.log((x - low + eps) / (high - x + eps))
    if np.isfinite(low) and high_inf:
        return np.log(x - low + eps)
    if low_inf and np.isfinite(high):
        return np.log(high - x + eps)
    raise NotImplementedError("Unhandled bound configuration")


def copy_clinear_from_theta(z, low, high):
    """Inverse transform for 'copy'/'clinear' betas with optional bounds."""
    low_inf = np.isinf(low)
    high_inf = np.isinf(high)

    if (low_inf and high_inf) or (low == high):
        return z
    if np.isfinite(low) and np.isfinite(high):
        if not (low < high):
            raise ValueError("low must be less than high")
        ez = np.exp(z)
        return low + ez / (1.0 + ez) * (high - low)
    if np.isfinite(low) and high_inf:
        return low + np.exp(z)
    if low_inf and np.isfinite(high):
        return high - np.exp(z)
    raise NotImplementedError("Unhandled bound configuration")
