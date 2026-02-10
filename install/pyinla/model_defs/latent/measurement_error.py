"""
Measurement error models: mec, meb

Classical and Berkson measurement error models.
"""

import numpy as np

MODELS = {
    'mec': {
        'doc': "Classical measurement error model",
        'hyper': {
            'theta1': {
                'hyperid': 2001,
                'name': "beta",
                'short_name': "b",
                'prior': "gaussian",
                'param': [1, 0.001],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 2002,
                'name': "prec.u",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0001],
                'initial': np.log(1 / 0.0001),
                'fixed': True,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 2003,
                'name': "mean.x",
                'short_name': "mu.x",
                'prior': "gaussian",
                'param': [0, 0.0001],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta4': {
                'hyperid': 2004,
                'name': "prec.x",
                'short_name': "prec.x",
                'prior': "loggamma",
                'param': [1, 10000],
                'initial': np.log(1 / 10000),
                'fixed': True,
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
        'pdf': "mec"
    },
    'meb': {
        'doc': "Berkson measurement error model",
        'hyper': {
            'theta1': {
                'hyperid': 3001,
                'name': "beta",
                'short_name': "b",
                'prior': "gaussian",
                'param': [1, 0.001],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 3002,
                'name': "prec.u",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0001],
                'initial': np.log(1000),
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
        'pdf': "meb"
    },
}
