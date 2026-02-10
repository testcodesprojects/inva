"""
SPDE models: spde, spde2, spde3, matern2d, dmatern

Stochastic Partial Differential Equation models for spatial/spatio-temporal data.
"""

import numpy as np

MODELS = {
    'spde': {
        'doc': "A SPDE model",
        'hyper': {
            'theta1': {
                'hyperid': 22001,
                'name': "theta.T",
                'short_name': "T",
                'initial': 2,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 22002,
                'name': "theta.K",
                'short_name': "K",
                'initial': -2,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 22003,
                'name': "theta.KT",
                'short_name': "KT",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta4': {
                'hyperid': 22004,
                'name': "theta.OC",
                'short_name': "OC",
                'initial': -20,
                'fixed': True,
                'prior': "normal",
                'param': [0, 0.2],
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
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
        'pdf': "spde"
    },
    'spde2': {
        'doc': "A SPDE2 model",
        'hyper': {
            'theta1': {
                'hyperid': 23001,
                'name': "theta1",
                'short_name': "t1",
                'initial': 0,
                'fixed': False,
                'prior': "mvnorm",
                'param': [1, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 23000 + i,
                'name': f"theta{i}",
                'short_name': f"t{i}",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 101)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "spde2"
    },
    'spde3': {
        'doc': "A SPDE3 model",
        'hyper': {
            'theta1': {
                'hyperid': 24001,
                'name': "theta1",
                'short_name': "t1",
                'initial': 0,
                'fixed': False,
                'prior': "mvnorm",
                'param': [1, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 24000 + i,
                'name': f"theta{i}",
                'short_name': f"t{i}",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 101)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "spde3"
    },
    'matern2d': {
        'doc': "Matern covariance function on a regular grid",
        'hyper': {
            'theta1': {
                'hyperid': 35001,
                'name': "log precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 35002,
                'name': "log range",
                'short_name': "range",
                'initial': 2,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.01],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': True,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "matern2d"
    },
    'dmatern': {
        'doc': "Dense Matern field",
        'hyper': {
            'theta1': {
                'hyperid': 35101,
                'name': "log precision",
                'short_name': "prec",
                'initial': 3,
                'fixed': False,
                'prior': "pc.prec",
                'param': [1, 0.01],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 35102,
                'name': "log range",
                'short_name': "range",
                'initial': 0,
                'fixed': False,
                'prior': "pc.range",
                'param': [1, 0.5],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 35103,
                'name': "log nu",
                'short_name': "nu",
                'initial': np.log(0.5),
                'fixed': True,
                'prior': "loggamma",
                'param': [0.5, 1],
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
        'pdf': "dmatern"
    },
}
