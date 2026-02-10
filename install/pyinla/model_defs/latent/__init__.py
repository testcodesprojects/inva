"""
Latent model definitions for INLA, organized by category.

Categories:
- basic: linear, iid, z
- measurement_error: mec, meb
- random_walk: rw1, rw2, crw2, rw2d, rw2diid
- autoregressive: ar1, ar1c, ar, ou
- spatial: besag, besag2, bym, bym2, besagproper, besagproper2, slm
- spde: spde, spde2, spde3, matern2d, dmatern
- multidim: iid1d, iid2d, iid3d, iid4d, iid5d, iidkd, 2diid
- copy: copy, scopy, clinear
- generic: generic, generic0-3, rgeneric, cgeneric
- misc: seasonal, fgn, fgn2, intslope, sigm, revsigm, log1exp, logdist
"""

from .basic import MODELS as BASIC_MODELS
from .measurement_error import MODELS as MEASUREMENT_ERROR_MODELS
from .random_walk import MODELS as RANDOM_WALK_MODELS
from .autoregressive import MODELS as AUTOREGRESSIVE_MODELS
from .spatial import MODELS as SPATIAL_MODELS
from .spde import MODELS as SPDE_MODELS
from .multidim import MODELS as MULTIDIM_MODELS, _SPECIAL_NUMBER
from .copy import MODELS as COPY_MODELS, copy_clinear_to_theta, copy_clinear_from_theta
from .generic import MODELS as GENERIC_MODELS
from .misc import MODELS as MISC_MODELS

# Combine all models into a single dictionary
LATENT_MODELS = {
    **BASIC_MODELS,
    **MEASUREMENT_ERROR_MODELS,
    **RANDOM_WALK_MODELS,
    **AUTOREGRESSIVE_MODELS,
    **SPATIAL_MODELS,
    **SPDE_MODELS,
    **MULTIDIM_MODELS,
    **COPY_MODELS,
    **GENERIC_MODELS,
    **MISC_MODELS,
}


def get_latent_models() -> dict:
    """Return all latent model definitions."""
    return LATENT_MODELS


# Export for backwards compatibility
__all__ = [
    'LATENT_MODELS',
    'get_latent_models',
    'copy_clinear_to_theta',
    'copy_clinear_from_theta',
    '_SPECIAL_NUMBER',
    # Individual category exports
    'BASIC_MODELS',
    'MEASUREMENT_ERROR_MODELS',
    'RANDOM_WALK_MODELS',
    'AUTOREGRESSIVE_MODELS',
    'SPATIAL_MODELS',
    'SPDE_MODELS',
    'MULTIDIM_MODELS',
    'COPY_MODELS',
    'GENERIC_MODELS',
    'MISC_MODELS',
]
