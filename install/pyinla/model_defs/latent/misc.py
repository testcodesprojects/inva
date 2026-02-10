"""
Miscellaneous models: seasonal, fgn, fgn2, intslope, sigm, revsigm, log1exp, logdist

Various specialized latent models.
"""

import numpy as np

MODELS = {
    'seasonal': {
        'doc': "Seasonal model for time series",
        'hyper': {
            'theta': {
                'hyperid': 7001,
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
        'pdf': "seasonal"
    },
    'fgn': {
        'doc': "Fractional Gaussian noise model",
        'hyper': {
            'theta1': {
                'hyperid': 13101,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [3, 0.01],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13102,
                'name': "logit H",
                'short_name': "H",
                'prior': "pcfgnh",
                'param': [0.9, 0.1],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((2 * x - 1) / (2 * (1 - x))),
                'from_theta': lambda x: 0.5 + 0.5 * np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 5,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'order_default': 4,
        'order_defined': list(range(3, 5)),
        'pdf': "fgn"
    },
    'fgn2': {
        'doc': "Fractional Gaussian noise model (alt 2)",
        'hyper': {
            'theta1': {
                'hyperid': 13111,
                'name': "log precision",
                'short_name': "prec",
                'prior': "pc.prec",
                'param': [3, 0.01],
                'initial': 1,
                'fixed': False,
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 13112,
                'name': "logit H",
                'short_name': "H",
                'prior': "pcfgnh",
                'param': [0.9, 0.1],
                'initial': 2,
                'fixed': False,
                'to_theta': lambda x: np.log((2 * x - 1) / (2 * (1 - x))),
                'from_theta': lambda x: 0.5 + 0.5 * np.exp(x) / (1 + np.exp(x))
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 4,
        'aug_constr': 1,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'order_default': 4,
        'order_defined': list(range(3, 5)),
        'pdf': "fgn"
    },
    'intslope': {
        'doc': "Intercept-slope model with Wishart-prior",
        'hyper': {
            'theta1': {
                'hyperid': 16101,
                'name': "log precision1",
                'short_name': "prec1",
                'initial': 4,
                'fixed': False,
                'prior': "wishart2d",
                'param': [4, 1, 1, 0],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 16102,
                'name': "log precision2",
                'short_name': "prec2",
                'initial': 4,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 16103,
                'name': "logit correlation",
                'short_name': "cor",
                'initial': 4,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            **{f'theta{i+4}': {
                'hyperid': 16101 + i + 3,
                'name': f"gamma{i+1}",
                'short_name': f"g{i+1}",
                'initial': 1,
                'fixed': True,
                'prior': "normal",
                'param': [1, 36],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(50)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': False,
        'set_default_values': True,
        'pdf': "intslope"
    },
    'sigm': {
        'doc': "Sigmoidal effect of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 38001,
                'name': "beta",
                'short_name': "b",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [1, 10],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 38002,
                'name': "loghalflife",
                'short_name': "halflife",
                'initial': 3,
                'fixed': False,
                'prior': "loggamma",
                'param': [3, 1],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 38003,
                'name': "logshape",
                'short_name': "shape",
                'initial': 0,
                'fixed': False,
                'prior': "loggamma",
                'param': [10, 10],
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
        'pdf': "sigm"
    },
    'revsigm': {
        'doc': "Reverse sigmoidal effect of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39001,
                'name': "beta",
                'short_name': "b",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [1, 10],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39002,
                'name': "loghalflife",
                'short_name': "halflife",
                'initial': 3,
                'fixed': False,
                'prior': "loggamma",
                'param': [3, 1],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 39003,
                'name': "logshape",
                'short_name': "shape",
                'initial': 0,
                'fixed': False,
                'prior': "loggamma",
                'param': [10, 10],
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
        'pdf': "sigm"
    },
    'log1exp': {
        'doc': "A nonlinear model of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39011,
                'name': "beta",
                'short_name': "b",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39012,
                'name': "alpha",
                'short_name': "a",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta3': {
                'hyperid': 39013,
                'name': "gamma",
                'short_name': "g",
                'initial': 0,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
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
        'pdf': "log1exp"
    },
    'logdist': {
        'doc': "A nonlinear model of a covariate",
        'hyper': {
            'theta1': {
                'hyperid': 39021,
                'name': "beta",
                'short_name': "b",
                'initial': 1,
                'fixed': False,
                'prior': "normal",
                'param': [0, 1],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            'theta2': {
                'hyperid': 39022,
                'name': "alpha1",
                'short_name': "a1",
                'initial': 0,
                'fixed': False,
                'prior': "loggamma",
                'param': [0.1, 1],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 39023,
                'name': "alpha2",
                'short_name': "a2",
                'initial': 0,
                'fixed': False,
                'prior': "loggamma",
                'param': [0.1, 1],
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
        'pdf': "logdist"
    },
}
