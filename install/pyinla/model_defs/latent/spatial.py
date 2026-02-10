"""
Spatial models: besag, besag2, bym, bym2, besagproper, besagproper2, slm

Conditional autoregressive (CAR) models for areal data.
"""

import numpy as np

MODELS = {
    'besag': {
        'doc': "The Besag area model (CAR-model)",
        'hyper': {
            'theta': {
                'hyperid': 8001,
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
        'n_required': True,
        'set_default_values': True,
        'pdf': "besag"
    },
    'besag2': {
        'doc': "The shared Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 9001,
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
                'hyperid': 9002,
                'name': "scaling parameter",
                'short_name': "a",
                'prior': "loggamma",
                'param': [10, 10],
                'initial': 0,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': [1, 2],
        'n_div_by': 2,
        'n_required': True,
        'set_default_values': True,
        'pdf': "besag2"
    },
    'bym': {
        'doc': "The BYM-model (Besag-York-Mollier model)",
        'hyper': {
            'theta1': {
                'hyperid': 10001,
                'name': "log unstructured precision",
                'short_name': "prec.unstruct",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 10002,
                'name': "log spatial precision",
                'short_name': "prec.spatial",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 4,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "bym"
    },
    'bym2': {
        'doc': "The BYM-model with the PC priors",
        'hyper': {
            'theta1': {
                'hyperid': 11001,
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
                'hyperid': 11002,
                'name': "logit phi",
                'short_name': "phi",
                'prior': "pc",
                'param': [0.5, 0.5],
                'initial': -3,
                'fixed': False,
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': True,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "bym2"
    },
    'besagproper': {
        'doc': "A proper version of the Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 12001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 12002,
                'name': "log diagonal",
                'short_name': "diag",
                'prior': "loggamma",
                'param': [1, 1],
                'initial': 1,
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
        'n_required': True,
        'set_default_values': True,
        'pdf': "besagproper"
    },
    'besagproper2': {
        'doc': "An alternative proper version of the Besag model",
        'hyper': {
            'theta1': {
                'hyperid': 13001,
                'name': "log precision",
                'short_name': "prec",
                'prior': "loggamma",
                'param': [1, 0.0005],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13002,
                'name': "logit lambda",
                'short_name': "lambda",
                'prior': "gaussian",
                'param': [0, 0.45],
                'initial': 3,
                'fixed': False,
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
        'pdf': "besagproper2"
    },
    'slm': {
        'doc': "Spatial lag model",
        'hyper': {
            'theta1': {
                'hyperid': 34001,
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
                'hyperid': 34002,
                'name': "rho",
                'short_name': "rho",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 10],
                'to_theta': lambda x: np.log(x / (1 - x)),
                'from_theta': lambda x: 1 / (1 + np.exp(-x))
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
        'pdf': "slm"
    },
}
