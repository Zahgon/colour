"""
Blackbody - Planck (1900) - Correlated Colour Temperature
=========================================================

Define the *Planck (1900)* correlated colour temperature :math:`T_{cp}`
computation objects based on the spectral radiance of a planckian
radiator:

-   :func:`colour.temperature.uv_to_CCT_Planck1900`
-   :func:`colour.temperature.CCT_to_uv_Planck1900`

References
----------
-   :cite:`CIETC1-482004i` : CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON
    THE USE OF PLANCK'S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
    3rd Edition (pp. 77-82). ISBN:978-3-901906-33-6
"""

from __future__ import annotations

import typing

import numpy as np

from colour.colorimetry import (
    MultiSpectralDistributions,
    handle_spectral_arguments,
    msds_to_XYZ_integration,
    planck_law,
)

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, DTypeFloat, NDArrayFloat

from colour.models import UCS_to_uv, XYZ_to_UCS
from colour.utilities import (
    array_namespace,
    as_float,
    as_float_array,
    required,
    xp_asarray,
    xp_atleast_1d,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "uv_to_CCT_Planck1900",
    "CCT_to_uv_Planck1900",
]


@required("SciPy")
def uv_to_CCT_Planck1900(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    optimisation_kwargs: dict | None = None,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` of a blackbody
    from specified *CIE UCS* colourspace *uv* chromaticity coordinates using
    colour matching functions.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    The current implementation relies on optimisation using
    :func:`scipy.optimize.minimize` definition and thus has reduced
    precision and poor performance.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> uv_to_CCT_Planck1900(np.array([0.20042808, 0.31033343]))
    ... # doctest: +ELLIPSIS
    np.float64(6504.0000617...)
    """

    from scipy.optimize import minimize  # noqa: PLC0415

    uv = as_float_array(uv)

    xp = array_namespace(uv)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    shape = uv.shape
    uv = xp_atleast_1d(xp_reshape(uv, (-1, 2), xp=xp), xp=xp)

    def objective_function(CCT: NDArrayFloat, uv: NDArrayFloat) -> DTypeFloat:
        """Objective function."""

        objective = np.linalg.norm(CCT_to_uv_Planck1900(CCT, cmfs) - np.asarray(uv))

        return as_float(objective)

    optimisation_settings = {
        "method": "Nelder-Mead",
        "options": {
            "fatol": 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    CCT = as_float_array(
        [
            minimize(
                objective_function,
                x0=[6500],
                args=(uv_i,),
                **optimisation_settings,
            ).x
            for uv_i in uv
        ]
    )

    CCT = xp_asarray(CCT, xp=xp, like=uv)
    return as_float(xp_reshape(CCT, shape[:-1], xp=xp))


def CCT_to_uv_Planck1900(
    CCT: ArrayLike, cmfs: MultiSpectralDistributions | None = None
) -> NDArrayFloat:
    """
    Compute the *CIE UCS* colourspace *uv* chromaticity coordinates from the
    specified correlated colour temperature :math:`T_{cp}` and colour
    matching functions using the spectral radiance of a blackbody at the
    specified thermodynamic temperature.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> CCT_to_uv_Planck1900(6504)  # doctest: +ELLIPSIS
    array([0.2004280..., 0.3103334...])
    """

    CCT = as_float_array(CCT)

    xp = array_namespace(CCT)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    spd = (
        planck_law(
            cmfs.wavelengths * 1e-9,
            xp_reshape(CCT, (-1,), xp=xp),
        )
        * 1e-9
    )
    if spd.ndim >= 2:
        spd = xp.matrix_transpose(spd)

    XYZ = msds_to_XYZ_integration(
        spd,
        cmfs,
        shape=cmfs.shape,
    )

    UVW = XYZ_to_UCS(XYZ)
    uv = UCS_to_uv(UVW)

    return xp_reshape(uv, [*list(CCT.shape), 2], xp=xp)
