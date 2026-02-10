"""
Generic models: generic, generic0, generic1, generic2, generic3, rgeneric, cgeneric

User-defined models with custom precision matrices.
"""

import numpy as np

MODELS = {
    'rgeneric': {
        'doc': "Generic latent model specified using R",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "rgeneric"
    },
    'cgeneric': {
        'doc': "Generic latent model specified using C",
        'hyper': {},
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "rgeneric"
    },
    'generic': {
        'doc': "A generic model",
        'hyper': {
            'theta': {
                'hyperid': 17001,
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
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic0"
    },
    'generic0': {
        'doc': "A generic model (type 0)",
        'hyper': {
            'theta': {
                'hyperid': 18001,
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
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic0"
    },
    'generic1': {
        'doc': "A generic model (type 1)",
        'hyper': {
            'theta1': {
                'hyperid': 19001,
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
                'hyperid': 19002,
                'name': "beta",
                'short_name': "beta",
                'initial': 2,
                'fixed': False,
                'prior': "gaussian",
                'param': [0, 0.1],
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
        'pdf': "generic1"
    },
    'generic2': {
        'doc': "A generic model (type 2)",
        'hyper': {
            'theta1': {
                'hyperid': 20001,
                'name': "log precision cmatrix",
                'short_name': "prec",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            },
            'theta2': {
                'hyperid': 20002,
                'name': "log precision random",
                'short_name': "prec.random",
                'initial': 4,
                'fixed': False,
                'prior': "loggamma",
                'param': [1, 0.001],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            }
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 2,
        'aug_constr': 2,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic2"
    },
    'generic3': {
        'doc': "A generic model (type 3)",
        'hyper': {
            **{f'theta{i}': {
                'hyperid': 21000 + i,
                'name': f"log precision{i if i <= 10 else ' common'}",
                'short_name': f"prec{i if i <= 10 else '.common'}",
                'initial': 4 if i <= 10 else 0,
                'fixed': i > 10,
                'prior': "loggamma",
                'param': [1, 0.00005],
                'to_theta': lambda x: np.log(x),
                'from_theta': lambda x: np.exp(x)
            } for i in range(1, 12)}
        },
        'constr': False,
        'nrow_ncol': False,
        'augmented': False,
        'aug_factor': 1,
        'aug_constr': None,
        'n_div_by': None,
        'n_required': True,
        'set_default_values': True,
        'pdf': "generic3"
    },
}
