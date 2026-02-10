"""
Basic latent models: linear, iid, z

These are the fundamental building blocks for random effects in INLA.
"""

import numpy as np

MODELS = {
    'linear': {
        'doc': "Alternative interface to a fixed effect",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "linear"
    },
    'iid': {
        'doc': "Gaussian random effects in dim=1",
        'hyper': {
            'theta': {
                'hyperid': 1001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
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
        'pdf': "indep"
    },
    'z': {
        'doc': "The z-model in a classical mixed model formulation",
        'hyper': {
            'theta': {
                'hyperid': 31001,
                'name': "log precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "z"
    },
}
