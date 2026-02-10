"""
Multivariate IID models: iid1d, iid2d, iid3d, iid4d, iid5d, iidkd, 2diid

Multivariate Gaussian random effects with Wishart priors.
"""

import numpy as np

_SPECIAL_NUMBER = 1048576.0

MODELS = {
    'iid1d': {
        'doc': "Gaussian random effect in dim=1 with Wishart prior",
        'hyper': {
            'theta': {
                'hyperid': 25001,
                'name': "precision",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "wishart1d",
                'param': [2.0, 2.0 * 0.00005],
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
        'set_default_values': True,
        'pdf': "iid123d"
    },
    'iid2d': {
        'doc': "Gaussian random effect in dim=2 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 26001,
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
                'hyperid': 26002,
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
                'hyperid': 26003,
                'name': "logit correlation",
                'short_name': "cor",
                'initial': 4,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 1,
        'aug_constr': [1, 2],
        'n_div_by': 2,
        'n_required': True,
        'set_default_values': True,
        'pdf': "iid123d"
    },
    'iid3d': {
        'doc': "Gaussian random effect in dim=3 with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 27001,
                'name': "log precision1",
                'short_name': "prec1",
                'initial': 4,
                'fixed': False,
                'prior': "wishart3d",
                'param': [7, 1, 1, 1, 0, 0, 0],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 27002,
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
                'hyperid': 27003,
                'name': "log precision3",
                'short_name': "prec3",
                'initial': 4,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta4': {
                'hyperid': 27004,
                'name': "logit correlation12",
                'short_name': "cor12",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta5': {
                'hyperid': 27005,
                'name': "logit correlation13",
                'short_name': "cor13",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            },
            'theta6': {
                'hyperid': 27006,
                'name': "logit correlation23",
                'short_name': "cor23",
                'initial': 0,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 1,
        'aug_constr': [1, 2, 3],
        'n_div_by': 3,
        'n_required': True,
        'set_default_values': True,
        'pdf': "iid123d"
    },
    'iid4d': {
        'doc': "Gaussian random effect in dim=4 with Wishart prior",
        'hyper': {
            'theta1': {'hyperid': 28001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False, 'prior': "wishart4d", 'param': [11, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta2': {'hyperid': 28002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta3': {'hyperid': 28003, 'name': "log precision3", 'short_name': "prec3", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta4': {'hyperid': 28004, 'name': "log precision4", 'short_name': "prec4", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta5': {'hyperid': 28005, 'name': "logit correlation12", 'short_name': "cor12", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta6': {'hyperid': 28006, 'name': "logit correlation13", 'short_name': "cor13", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta7': {'hyperid': 28007, 'name': "logit correlation14", 'short_name': "cor14", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta8': {'hyperid': 28008, 'name': "logit correlation23", 'short_name': "cor23", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta9': {'hyperid': 28009, 'name': "logit correlation24", 'short_name': "cor24", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta10': {'hyperid': 28010, 'name': "logit correlation34", 'short_name': "cor34", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1, 'aug_constr': [1, 2, 3, 4], 'n_div_by': 4, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iid5d': {
        'doc': "Gaussian random effect in dim=5 with Wishart prior",
        'hyper': {
            'theta1': {'hyperid': 29001, 'name': "log precision1", 'short_name': "prec1", 'initial': 4, 'fixed': False, 'prior': "wishart5d", 'param': [16, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta2': {'hyperid': 29002, 'name': "log precision2", 'short_name': "prec2", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta3': {'hyperid': 29003, 'name': "log precision3", 'short_name': "prec3", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta4': {'hyperid': 29004, 'name': "log precision4", 'short_name': "prec4", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta5': {'hyperid': 29005, 'name': "log precision5", 'short_name': "prec5", 'initial': 4, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log(x), 'from_theta': lambda x: np.exp(x)},
            'theta6': {'hyperid': 29006, 'name': "logit correlation12", 'short_name': "cor12", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta7': {'hyperid': 29007, 'name': "logit correlation13", 'short_name': "cor13", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta8': {'hyperid': 29008, 'name': "logit correlation14", 'short_name': "cor14", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta9': {'hyperid': 29009, 'name': "logit correlation15", 'short_name': "cor15", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta10': {'hyperid': 29010, 'name': "logit correlation23", 'short_name': "cor23", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta11': {'hyperid': 29011, 'name': "logit correlation24", 'short_name': "cor24", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta12': {'hyperid': 29012, 'name': "logit correlation25", 'short_name': "cor25", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta13': {'hyperid': 29013, 'name': "logit correlation34", 'short_name': "cor34", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta14': {'hyperid': 29014, 'name': "logit correlation35", 'short_name': "cor35", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
            'theta15': {'hyperid': 29015, 'name': "logit correlation45", 'short_name': "cor45", 'initial': 0, 'fixed': False, 'prior': "none", 'param': [], 'to_theta': lambda x: np.log((1 + x) / (1 - x)), 'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1},
        },
        'constr': False, 'nrow_ncol': False, 'augmented': True, 'aug_factor': 1, 'aug_constr': [1, 2, 3, 4, 5], 'n_div_by': 5, 'n_required': True, 'set_default_values': True, 'pdf': "iid123d"
    },
    'iidkd': {
        'doc': "Gaussian random effect in dim=k with Wishart prior",
        'hyper': {
            'theta1': {
                'hyperid': 29101,
                'name': "theta1",
                'short_name': "theta1",
                'initial': _SPECIAL_NUMBER,
                'fixed': False,
                'prior': "wishartkd",
                'param': [30] + [_SPECIAL_NUMBER] * int((24*25)/2),
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            },
            **{f'theta{i}': {
                'hyperid': 29100 + i,
                'name': f"theta{i}",
                'short_name': f"theta{i}",
                'initial': _SPECIAL_NUMBER,
                'fixed': False,
                'prior': "none",
                'param': [],
                'to_theta': lambda x: x,
                'from_theta': lambda x: x
            } for i in range(2, 301)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': True,
        'aug_factor': 1,
        'aug_constr': list(range(1, 25)),
        'n_div_by': -1,
        'n_required': True,
        'set_default_values': True,
        'pdf': "iidkd"
    },
    '2diid': {
        'doc': "(This model is obsolete)",
        'hyper': {
            'theta1': {
                'hyperid': 30001,
                'name': "log precision1",
                'short_name': "prec1",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 30002,
                'name': "log precision2",
                'short_name': "prec2",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta3': {
                'hyperid': 30003,
                'name': "correlation",
                'short_name': "cor",
                'initial': 4,
                'fixed': False,
                'prior': "normal",
                'param': [0, 0.15],
                'to_theta': lambda x: np.log((1 + x) / (1 - x)),
                'from_theta': lambda x: 2 * np.exp(x) / (1 + np.exp(x)) - 1
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': [1, 2],
        'n_div_by': 2,
        'n_required': True,
        'set_default_values': True,
        'pdf': "iid123d"
    },
}
