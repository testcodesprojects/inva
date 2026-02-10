"""
Autoregressive models: ar1, ar1c, ar, ou

Time series models with autoregressive structure.
"""

import numpy as np

MODELS = {
    'ar1': {
        'doc': "Auto-regressive model of order 1 (AR(1))",
        'hyper': {
            'theta1': {
                'hyperid': 14001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 14002,
                'name': "logit lag one correlation",
                'short_name': "rho",
                'prior': "normal",
                'param': [0, 0.15],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta3': {
                'hyperid': 14003,
                'name': "mean",
                'short_name': "mean",
                'prior': "normal",
                'param': [0, 1],
                'initial': 0,
                'fixed': True,
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
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
        'pdf': "ar1"
    },
    'ar1c': {
        'doc': "Auto-regressive model of order 1 w/covariates",
        'hyper': {
            'theta1': {
                'hyperid': 14101,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [1, 0.01],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 14102,
                'name': "logit lag one correlation",
                'short_name': "rho",
                'prior': "pc.cor0",
                'param': [0.5, 0.5],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "ar1c"
    },
    'ar': {
        'doc': "Auto-regressive model of order p (AR(p))",
        'hyper': {
            'theta1': {
                'hyperid': 15001,
                'name': "log precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "pc.prec",
                'param': [3, 0.01],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 15002, 'name': "pacf1", 'short_name': "pacf1", 'initial': 1, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.5],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta3': {
                'hyperid': 15003, 'name': "pacf2", 'short_name': "pacf2", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.4],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta4': {
                'hyperid': 15004, 'name': "pacf3", 'short_name': "pacf3", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.3],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta5': {
                'hyperid': 15005, 'name': "pacf4", 'short_name': "pacf4", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.2],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta6': {
                'hyperid': 15006, 'name': "pacf5", 'short_name': "pacf5", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta7': {
                'hyperid': 15007, 'name': "pacf6", 'short_name': "pacf6", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta8': {
                'hyperid': 15008, 'name': "pacf7", 'short_name': "pacf7", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta9': {
                'hyperid': 15009, 'name': "pacf8", 'short_name': "pacf8", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta10': {
                'hyperid': 15010, 'name': "pacf9", 'short_name': "pacf9", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta11': {
                'hyperid': 15011, 'name': "pacf10", 'short_name': "pacf10", 'initial': 0, 'fixed': False,
                'prior': "pc.cor0", 'param': [0.5, 0.1],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': False,
        'pdf': "ar"
    },
    'ou': {
        'doc': "The Ornstein-Uhlenbeck process",
        'hyper': {
            'theta1': {
                'hyperid': 16001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.00005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 16002,
                'name': "log phi",
                'short_name': "phi",
                'prior': "normal",
                'param': [0, 0.2],
                'initial': -1,
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
        'pdf': "ou"
    },
}
