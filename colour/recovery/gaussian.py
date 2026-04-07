"""
Gaussian - Reflectance Recovery
===============================

Define objects for reflectance recovery using Gaussian basis spectra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ_integration,
    reshape_msds,
    reshape_sd,
    sd_constant,
    sd_gaussian_super_clamped,
    sd_to_XYZ,
)
from colour.difference import delta_E
from colour.models import RGB_Colourspace, RGB_COLOURSPACE_sRGB, XYZ_to_Lab, XYZ_to_RGB
from colour.recovery.smits1999 import RGB_to_msds_Smits1999, RGB_to_sd_Smits1999
from colour.utilities import as_float, optional, required

if TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain1, DTypeFloat, NDArrayFloat, Range1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_GAUSSIAN",
    "WHITEPOINT_NAME_GAUSSIAN",
    "CCS_WHITEPOINT_GAUSSIAN",
    "RGB_COLOURSPACE_GAUSSIAN",
    "XYZ_to_RGB_Gaussian",
    "optimise_gaussian_basis_parameters",
    "generate_gaussian_basis",
    "MSDS_GAUSSIAN_BASIS",
    "PEAK_WAVELENGTHS_GAUSSIAN_BASIS",
    "FWHM_GAUSSIAN_BASIS",
    "EXPONENT_GAUSSIAN_BASIS",
    "RGB_to_msds_Gaussian",
    "RGB_to_sd_Gaussian",
]

PRIMARIES_GAUSSIAN: NDArrayFloat = RGB_COLOURSPACE_sRGB.primaries
"""*Gaussian* method implementation colourspace primaries."""

WHITEPOINT_NAME_GAUSSIAN: str = "E"
"""*Gaussian* method implementation colourspace whitepoint name."""

CCS_WHITEPOINT_GAUSSIAN: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_GAUSSIAN]
"""*Gaussian* method implementation colourspace whitepoint."""

RGB_COLOURSPACE_GAUSSIAN: RGB_Colourspace = RGB_Colourspace(
    "Gaussian",
    PRIMARIES_GAUSSIAN,
    CCS_WHITEPOINT_GAUSSIAN,
    WHITEPOINT_NAME_GAUSSIAN,
)
"""*Gaussian* colourspace."""


def XYZ_to_RGB_Gaussian(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace using
    the conditions required by the *Gaussian* method implementation.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colour array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Gaussian(XYZ)  # doctest: +ELLIPSIS
    array([0.4063959..., 0.0275289..., 0.0398219...])
    """

    return XYZ_to_RGB(XYZ, RGB_COLOURSPACE_GAUSSIAN)


@required("SciPy")
def optimise_gaussian_basis_parameters(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    colourspace: RGB_Colourspace | None = None,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    initial_peak_wavelengths: dict | None = None,
    initial_fwhm: dict | None = None,
    initial_exponent: dict | None = None,
    smoothness_penalty_weight: float = 1.0,
    optimisation_kwargs: dict | None = None,
) -> tuple[dict, dict, dict]:
    """
    Optimise Gaussian basis parameters for colorimetric accuracy.

    This function finds the peak wavelengths, FWHM values, and exponents that
    minimize the colorimetric error between the basis spectra tristimulus
    values and the target *RGB* colourspace primaries, as well as the
    :math:`\\Delta E^*_{ab}` error on *ColorChecker* patches.

    The secondary colours are modeled based on their spectral characteristics:

    -   **Cyan**: Peak in blue-green region (absorbs red), modeled as a
        Gaussian peak clamped left, similar to blue.
    -   **Yellow**: Peak in red-green region (absorbs blue), modeled as a
        Gaussian peak clamped right, similar to red.
    -   **Magenta**: High at red and blue, low at green (absorbs green),
        modeled as an inverted Gaussian (valley at green wavelengths).

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    colourspace
        *RGB* colourspace for the optimization. Default is
        :attr:`RGB_COLOURSPACE_GAUSSIAN` which uses *sRGB* primaries and
        illuminant *E* whitepoint.
    cmfs
        Standard observer colour matching functions used for computing the
        spectral tristimulus values. Default is
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution for *XYZ* integration. If not
        provided, defaults to the illuminant matching the colourspace's
        whitepoint name, or illuminant *E* if not found.
    initial_peak_wavelengths
        Initial peak wavelengths for optimization. Default includes RGB peaks
        plus cyan, magenta, and yellow peak positions.
    initial_fwhm
        Initial FWHM values for optimization. Default includes RGB FWHM
        plus cyan, magenta, and yellow FWHM values.
    initial_exponent
        Initial exponents for super-Gaussian per basis function. Default 2.0
        gives standard Gaussian. Values > 2 give flatter peaks.
    smoothness_penalty_weight
        Weight for the smoothness penalty that penalizes deviation from
        standard Gaussian (exponent=2). Higher values produce smoother curves
        closer to standard Gaussians.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`tuple`
        Tuple of (peak_wavelengths, fwhm, exponent) with optimized values.
        All three are dictionaries with entries for red, green, blue, cyan,
        magenta, and yellow.

    References
    ----------
    :cite:`Smits1999a`
    """
    pass


def generate_gaussian_basis(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    peak_wavelengths: dict | None = None,
    fwhm: dict | None = None,
    exponent: dict | None = None,
) -> MultiSpectralDistributions:
    """
    Generate a set of Gaussian basis multi-spectral distributions.

    The secondary colours are modeled based on their spectral characteristics:

    -   **Cyan**: Peak in blue-green region (absorbs red), modeled as a
        Gaussian peak clamped left, similar to blue.
    -   **Yellow**: Peak in red-green region (absorbs blue), modeled as a
        Gaussian peak clamped right, similar to red.
    -   **Magenta**: High at red and blue, low at green (absorbs green),
        modeled as an inverted Gaussian (valley at green wavelengths).

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    peak_wavelengths
        Dictionary with peak wavelengths for red, green, blue, cyan, magenta,
        and yellow.
    fwhm
        Dictionary with FWHM values for red, green, blue, cyan, magenta, and
        yellow.
    exponent
        Dictionary with exponent values for red, green, blue, cyan, magenta,
        and yellow. Default 2.0 gives a standard Gaussian. Values > 2 give a
        flatter peak (super-Gaussian).

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Gaussian basis multi-spectral distributions with signals: white, cyan,
        magenta, yellow, red, green, blue.

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> basis = generate_gaussian_basis()
    >>> sorted(basis.labels)
    ['blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow']
    >>> float(basis.signals["yellow"].values.max())  # doctest: +ELLIPSIS
    1.0...
    """

    peak_wavelengths = optional(peak_wavelengths, PEAK_WAVELENGTHS_GAUSSIAN_BASIS)
    fwhm = optional(fwhm, FWHM_GAUSSIAN_BASIS)
    exponent = optional(exponent, EXPONENT_GAUSSIAN_BASIS)

    # Primary colours with clamping
    sd_red = sd_gaussian_super_clamped(
        peak_wavelengths["red"],
        fwhm["red"],
        shape,
        clamp="right",
        exponent=exponent["red"],
        name="red",
    )
    sd_green = sd_gaussian_super_clamped(
        peak_wavelengths["green"],
        fwhm["green"],
        shape,
        clamp="none",
        exponent=exponent["green"],
        name="green",
    )
    sd_blue = sd_gaussian_super_clamped(
        peak_wavelengths["blue"],
        fwhm["blue"],
        shape,
        clamp="left",
        exponent=exponent["blue"],
        name="blue",
    )

    # Cyan: peak in blue-green region, clamped left
    sd_cyan = sd_gaussian_super_clamped(
        peak_wavelengths["cyan"],
        fwhm["cyan"],
        shape,
        clamp="left",
        exponent=exponent["cyan"],
        name="cyan",
    )

    # Yellow: peak in red-green region, clamped right
    sd_yellow = sd_gaussian_super_clamped(
        peak_wavelengths["yellow"],
        fwhm["yellow"],
        shape,
        clamp="right",
        exponent=exponent["yellow"],
        name="yellow",
    )

    # Magenta: inverted Gaussian (valley) with independent peak and fwhm
    sd_M_gaussian = sd_gaussian_super_clamped(
        peak_wavelengths["magenta"],
        fwhm["magenta"],
        shape,
        clamp="none",
        exponent=exponent["magenta"],
    )
    sd_magenta = sd_constant(1, shape) - sd_M_gaussian
    sd_magenta.name = "magenta"

    # White as constant
    sd_white = sd_constant(1, shape)
    sd_white.name = "white"

    sds = [sd_white, sd_cyan, sd_magenta, sd_yellow, sd_red, sd_green, sd_blue]

    return MultiSpectralDistributions(
        sds,
        labels=[sd.name for sd in sds],
        name="Gaussian Basis",
    )


PEAK_WAVELENGTHS_GAUSSIAN_BASIS: dict = {
    "red": 620.280,
    "green": 538.554,
    "blue": 443.979,
    "cyan": 568.405,
    "magenta": 540.500,
    "yellow": 534.422,
}
"""
Default peak wavelengths for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

FWHM_GAUSSIAN_BASIS: dict = {
    "red": 75.000,
    "green": 118.702,
    "blue": 112.619,
    "cyan": 55.000,
    "magenta": 128.926,
    "yellow": 123.124,
}
"""
Default full width at half maximum for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

EXPONENT_GAUSSIAN_BASIS: dict = {
    "red": 2.10,
    "green": 2.13,
    "blue": 2.00,
    "cyan": 2.02,
    "magenta": 2.10,
    "yellow": 2.00,
}
"""
Default exponents for Gaussian basis spectra.

A value of 2.0 gives a standard Gaussian. Values > 2 give a flatter peak
(super-Gaussian). These values are optimized for round-trip colorimetric
accuracy using :func:`optimise_gaussian_basis_parameters`.
"""

MSDS_GAUSSIAN_BASIS: MultiSpectralDistributions = generate_gaussian_basis()
MSDS_GAUSSIAN_BASIS.__doc__ = """
Gaussian basis multi-spectral distributions for spectral upsampling.

The basis spectra use clamped Gaussians with parameters optimized for round-trip
colorimetric accuracy with *sRGB* primaries:

- Red: Gaussian clamped flat toward long wavelengths
- Green: Symmetric Gaussian
- Blue: Gaussian clamped flat toward short wavelengths
- Cyan: Independent Gaussian clamped flat toward short wavelengths
- Yellow: Independent Gaussian clamped flat toward long wavelengths
- Magenta: Inverted Gaussian (valley at green wavelengths)
- White: Constant value of 1

Parameters can be recomputed using :func:`optimise_gaussian_basis_parameters`.
"""


def RGB_to_msds_Gaussian(RGB: ArrayLike) -> NDArrayFloat:
    """
    Recover spectral values from *RGB* colourspace array using *Gaussian*
    basis spectra and the *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover spectral values from. The last
        dimension must be size 3.

    Returns
    -------
    :class:`numpy.ndarray`
        Recovered spectral values with shape ``(*RGB.shape[:-1], wavelengths)``.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array(
    ...     [
    ...         [0.45623196, 0.03080455, 0.04093343],
    ...         [0.05438271, 0.29877169, 0.07188444],
    ...         [0.01863137, 0.05139773, 0.28887675],
    ...     ]
    ... )
    >>> RGB_to_msds_Gaussian(RGB).shape
    (3, 421)
    >>> float(RGB_to_msds_Gaussian(RGB)[0, 300])  # doctest: +ELLIPSIS
    0.4561...
    """
    pass


def RGB_to_sd_Gaussian(RGB: Domain1) -> SpectralDistribution:
    """
    Recover the spectral distribution of the specified *RGB* colourspace array
    using Gaussian basis spectra and the *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover the spectral distribution from.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Recovered spectral distribution.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_RGB_Gaussian(XYZ)
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["E"].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Gaussian(RGB)
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([0.19324..., 0.11592..., 0.04332...])
    """

    return RGB_to_sd_Smits1999(RGB, MSDS_GAUSSIAN_BASIS, f"Gaussian - {RGB!r}")
