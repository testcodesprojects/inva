"""
Random walk models: rw1, rw2, crw2, rw2d, rw2diid

Temporal smoothing models based on random walks.
"""

import numpy as np

MODELS = {
    'rw1': {
        'doc': "Random walk of order 1",
        'hyper': {
            'theta': {
                'hyperid': 4001,
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
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-06,
        'pdf': "rw1"
    },
    'rw2': {
        'doc': "Random walk of order 2",
        'hyper': {
            'theta': {
                'hyperid': 5001,
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
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-04,
        'pdf': "rw2"
    },
    'crw2': {
        'doc': "Exact solution to the random walk of order 2",
        'hyper': {
            'theta': {
                'hyperid': 6001,
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
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 2,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'min_diff': 1e-04,
        'pdf': "crw2"
    },
    'rw2d': {
        'doc': "Thin-plate spline model",
        'hyper': {
            'theta': {
                'hyperid': 32001,
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
        'constr': True,
        'nrow_ncol': True,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "rw2d"
    },
    'rw2diid': {
        'doc': "Thin-plate spline with iid noise",
        'hyper': {
            'theta1': {
                'hyperid': 33001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [1, .01],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 33002,
                'name': "logit phi",
                'short_name': "phi",
                'prior': "pc",
                'param': [0.5, 0.5],
                'initial': 3,
                'fixed': False,
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': True,
        'nrow_ncol': True,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "rw2diid"
    },
}
