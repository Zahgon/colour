"""
Spectral Generation
===================

Define objects for generating spectral distributions and multi-spectral
distributions with the specified characteristics.

-   :func:`colour.sd_constant`
-   :func:`colour.sd_zeros`
-   :func:`colour.sd_ones`
-   :func:`colour.msds_constant`
-   :func:`colour.msds_zeros`
-   :func:`colour.msds_ones`
-   :func:`colour.colorimetry.sd_gaussian_normal`
-   :func:`colour.colorimetry.sd_gaussian_fwhm`
-   :func:`colour.colorimetry.sd_gaussian_super_clamped`
-   :attr:`colour.SD_GAUSSIAN_METHODS`
-   :func:`colour.sd_gaussian`
-   :func:`colour.colorimetry.sd_single_led_Ohno2005`
-   :attr:`colour.SD_SINGLE_LED_METHODS`
-   :func:`colour.sd_single_led`
-   :func:`colour.colorimetry.sd_multi_leds_Ohno2005`
-   :attr:`colour.SD_MULTI_LEDS_METHODS`
-   :func:`colour.sd_multi_leds`

References
----------
-   :cite:`Ohno2005` : Ohno, Yoshi. (2005). Spectral design considerations for
    white LED color rendering. Optical Engineering, 44(11), 111302.
    doi:10.1117/1.2130694
-   :cite:`Ohno2008a` : Ohno, Yoshiro, & Davis, W. (2008). NIST CQS simulation
    (Version 7.4) [Computer software].
    https://drive.google.com/file/d/1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?\
usp=sharing
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra.interpolation import LinearInterpolator
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Literal,
        NDArrayFloat,
        Sequence,
    )

from colour.utilities import (
    CanonicalMapping,
    as_float_array,
    full,
    ones,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "sd_constant",
    "sd_zeros",
    "sd_ones",
    "msds_constant",
    "msds_zeros",
    "msds_ones",
    "sd_gaussian_normal",
    "sd_gaussian_fwhm",
    "sd_gaussian_super_clamped",
    "SD_GAUSSIAN_METHODS",
    "sd_gaussian",
    "sd_single_led_Ohno2005",
    "SD_SINGLE_LED_METHODS",
    "sd_single_led",
    "sd_multi_leds_Ohno2005",
    "SD_MULTI_LEDS_METHODS",
    "sd_multi_leds",
]


def sd_constant(
    k: float, shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT, **kwargs: Any
) -> SpectralDistribution:
    """
    Generate a spectral distribution of the specified spectral shape filled with
    constant :math:`k` values.

    Parameters
    ----------
    k
        Constant :math:`k` to fill the spectral distribution with.
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.SpectralDistribution`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Constant :math:`k` filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator` class.

    Examples
    --------
    >>> sd = sd_constant(100)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    np.float64(100.0)
    """

    settings = {"name": f"{k} Constant", "interpolator": LinearInterpolator}
    settings.update(kwargs)

    values = full(len(shape.wavelengths), k)

    return SpectralDistribution(values, shape.wavelengths, **settings)


def sd_zeros(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT, **kwargs: Any
) -> SpectralDistribution:
    """
    Generate a spectral distribution of the specified spectral shape filled with
    zeros.

    Parameters
    ----------
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.sd_constant`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Zeros-filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator` class.

    Examples
    --------
    >>> sd = sd_zeros()
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    np.float64(0.0)
    """

    return sd_constant(0, shape, **kwargs)


def sd_ones(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT, **kwargs: Any
) -> SpectralDistribution:
    """
    Generate a spectral distribution of the specified spectral shape filled
    with ones.

    Parameters
    ----------
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.sd_constant`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Ones-filled spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator` class.

    Examples
    --------
    >>> sd = sd_ones()
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[400]
    np.float64(1.0)
    """

    return sd_constant(1, shape, **kwargs)


def msds_constant(
    k: float,
    labels: Sequence,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> MultiSpectralDistributions:
    """
    Generate multi-spectral distributions with the specified labels and spectral
    shape filled with constant :math:`k` values.

    Parameters
    ----------
    k
        Constant :math:`k` to fill the multi-spectral distributions with.
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape
        Spectral shape used to create the multi-spectral distributions.

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.MultiSpectralDistributions`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Constant :math:`k` filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape
        specified by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator` class.

    Examples
    --------
    >>> msds = msds_constant(100, labels=["a", "b", "c"])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([100., 100., 100.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """
    pass


def msds_zeros(
    labels: Sequence,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> MultiSpectralDistributions:
    """
    Generate multi-spectral distributions with the specified labels and spectral
    shape filled with zeros.

    Parameters
    ----------
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape
        Spectral shape used to create the multi-spectral distributions.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.msds_constant`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Zero-filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape
        specified by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator` class.

    Examples
    --------
    >>> msds = msds_zeros(labels=["a", "b", "c"])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([0., 0., 0.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """
    pass


def msds_ones(
    labels: Sequence,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> MultiSpectralDistributions:
    """
    Generate multi-spectral distributions with the specified labels and
    spectral shape filled with ones.

    Parameters
    ----------
    labels
        Names to use for the :class:`colour.SpectralDistribution` class
        instances.
    shape
        Spectral shape used to create the multi-spectral distributions.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.msds_constant`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Ones-filled multi-spectral distributions.

    Notes
    -----
    -   By default, the multi-spectral distributions will use the shape
        specified by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.
    -   The interpolator is set to :class:`colour.LinearInterpolator`
        class.

    Examples
    --------
    >>> msds = msds_ones(labels=["a", "b", "c"])
    >>> msds.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> msds[400]
    array([1., 1., 1.])
    >>> msds.labels  # doctest: +SKIP
    ['a', 'b', 'c']
    """
    pass


def sd_gaussian_normal(
    mu: float,
    sigma: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a Gaussian spectral distribution of the specified spectral shape at
    specified mean wavelength :math:`\\mu` and standard deviation
    :math:`\\sigma`.

    Parameters
    ----------
    mu
        Mean wavelength :math:`\\mu` at which the Gaussian spectral
        distribution will peak.
    sigma
        Standard deviation :math:`\\sigma` of the Gaussian spectral
        distribution.
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.SpectralDistribution`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian_normal(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +SKIP
    np.float64(1.0)
    >>> sd[530]  # doctest: +ELLIPSIS
    np.float64(0.6065306...)
    """
    pass


def sd_gaussian_fwhm(
    peak_wavelength: float,
    fwhm: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a Gaussian spectral distribution of the specified spectral shape at
    specified peak wavelength and full width at half maximum (FWHM).

    Parameters
    ----------
    peak_wavelength
        Wavelength at which the Gaussian spectral distribution peaks.
    fwhm
        Full width at half maximum, i.e., width of the Gaussian spectral
        distribution measured between those points on the *y* axis which are
        half the maximum amplitude.
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.SpectralDistribution`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian_fwhm(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +SKIP
    np.float64(1.0)
    >>> sd[530]  # doctest: +ELLIPSIS
    np.float64(0.062...)
    """
    pass


def sd_gaussian_super_clamped(
    peak_wavelength: float,
    fwhm: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    clamp: Literal["none", "left", "right"] | str = "none",
    exponent: float = 2.0,
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a super-Gaussian spectral distribution, optionally clamped flat on
    one side of the peak, with the peak normalized to 1.

    A super-Gaussian (exponent > 2) has a flatter peak than a regular Gaussian,
    which can better model real-world reflectance spectra.

    Parameters
    ----------
    peak_wavelength
        Peak wavelength of the Gaussian.
    fwhm
        Full width at half maximum.
    shape
        Spectral shape for the distribution.
    clamp
        Clamping mode: ``"none"`` for symmetric Gaussian, ``"left"`` for flat
        from start to peak, ``"right"`` for flat from peak to end.
    exponent
        Exponent for the Gaussian function. Default 2.0 gives a standard
        Gaussian. Values > 2 give a flatter peak (super-Gaussian).

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.SpectralDistribution`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Clamped super-Gaussian spectral distribution with peak normalized to 1.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian_super_clamped(600, 50, clamp="right")
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> round(sd[600], 5)
    np.float64(1.0)
    >>> round(sd[700], 5)
    np.float64(1.0)
    >>> sd = sd_gaussian_super_clamped(450, 40, clamp="left", exponent=4.0)
    >>> round(sd[450], 5)
    np.float64(1.0)
    >>> round(sd[350], 5)  # doctest: +SKIP
    1.0
    """

    settings = {"name": f"{peak_wavelength}nm - {fwhm} FWHM - Super-Gaussian Clamped"}
    settings.update(kwargs)

    wavelengths = shape.wavelengths
    # Convert FWHM to sigma: FWHM = 2 * sigma * (2 * ln(2))^(1/exponent)
    sigma = fwhm / (2 * (2 * np.log(2)) ** (1 / exponent))
    # Super-Gaussian: exp(-|x/sigma|^exponent)
    values = np.exp(-(np.abs((wavelengths - peak_wavelength) / sigma) ** exponent))

    sd = SpectralDistribution(values, wavelengths, **settings)
    sd.range = sd.range / sd.range.max()  # Normalize peak to 1

    if clamp == "left":
        sd[sd.wavelengths[sd.wavelengths <= peak_wavelength]] = 1.0
    elif clamp == "right":
        sd[sd.wavelengths[sd.wavelengths >= peak_wavelength]] = 1.0

    return sd


SD_GAUSSIAN_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Normal": sd_gaussian_normal,
        "FWHM": sd_gaussian_fwhm,
        "Super-Gaussian Clamped": sd_gaussian_super_clamped,
    }
)
SD_GAUSSIAN_METHODS.__doc__ = """
Supported gaussian spectral distribution generation methods.
"""


def sd_gaussian(
    mu_peak_wavelength: float,
    sigma_fwhm: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    method: Literal["Normal", "FWHM", "Super-Gaussian Clamped"] | str = "Normal",
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a Gaussian spectral distribution with the specified spectral
    shape using the specified method.

    Parameters
    ----------
    mu_peak_wavelength
        Mean wavelength :math:`\\mu` at which the Gaussian spectral
        distribution will peak.
    sigma_fwhm
        Standard deviation :math:`\\sigma` of the Gaussian spectral
        distribution or full width at half maximum (FWHM), i.e., the width
        of the Gaussian spectral distribution measured between those points
        on the *y* axis which are half the maximum amplitude.
    shape
        Spectral shape used to create the spectral distribution.
    method
        Computation method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.colorimetry.sd_gaussian_normal`,
        :func:`colour.colorimetry.sd_gaussian_fwhm`,
        :func:`colour.colorimetry.sd_gaussian_super_clamped`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Gaussian spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    Examples
    --------
    >>> sd = sd_gaussian(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +SKIP
    np.float64(1.0)
    >>> sd[530]  # doctest: +ELLIPSIS
    np.float64(0.6065306...)
    >>> sd = sd_gaussian(555, 25, method="FWHM")
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +SKIP
    np.float64(1.0)
    >>> sd[530]  # doctest: +ELLIPSIS
    np.float64(0.062...)
    >>> sd = sd_gaussian(600, 50, method="Super-Gaussian Clamped", clamp="right")
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> round(sd[600], 5)
    np.float64(1.0)
    >>> round(sd[700], 5)
    np.float64(1.0)
    """
    pass


def sd_single_led_Ohno2005(
    peak_wavelength: float,
    half_spectral_width: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a single *LED* spectral distribution with the specified spectral
    shape at specified peak wavelength and half spectral width
    :math:`\\Delta\\lambda_{0.5}` using *Ohno (2005)* method.

    Parameters
    ----------
    peak_wavelength
        Wavelength at which the single *LED* spectral distribution peaks.
    half_spectral_width
        Half spectral width :math:`\\Delta\\lambda_{0.5}`.
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:class:`colour.SpectralDistribution`},
        See the documentation of the previously listed class.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Single *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_single_led_Ohno2005(555, 25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    np.float64(1...)
    """
    pass


SD_SINGLE_LED_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Ohno 2005": sd_single_led_Ohno2005,
    }
)
SD_SINGLE_LED_METHODS.__doc__ = """
Supported single *LED* spectral distribution computation methods.
"""


def sd_single_led(
    peak_wavelength: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    method: Literal["Ohno 2005"] | str = "Ohno 2005",
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a single *LED* spectral distribution with the specified spectral
    shape at the specified peak wavelength using the specified method.

    Parameters
    ----------
    peak_wavelength
        Wavelength the single *LED* spectral distribution will peak at.
    shape
        Spectral shape used to create the spectral distribution.
    method
        Computation method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.colorimetry.sd_single_led_Ohno2005`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Single *LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified by
        :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_single_led(555, half_spectral_width=25)
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[555]  # doctest: +ELLIPSIS
    np.float64(1...)
    """
    pass


def sd_multi_leds_Ohno2005(
    peak_wavelengths: ArrayLike,
    half_spectral_widths: ArrayLike,
    peak_power_ratios: ArrayLike | None = None,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a multi-*LED* spectral distribution with the specified spectral
    shape at specified peak wavelengths, half spectral widths
    :math:`\\Delta\\lambda_{0.5}`, and peak power ratios according to the
    *Ohno (2005)* method.

    The multi-*LED* spectral distribution is computed by summing multiple
    single *LED* spectral distributions generated with the
    :func:`colour.sd_single_led_Ohno2005` function.

    Parameters
    ----------
    peak_wavelengths
        Wavelengths at which the multi-*LED* spectral distribution will
        peak, i.e., the peak wavelengths for each constituent single *LED*
        spectral distribution.
    half_spectral_widths
        Half spectral widths :math:`\\Delta\\lambda_{0.5}` for each
        constituent single *LED* spectral distribution.
    peak_power_ratios
        Peak power ratios for each constituent single *LED* spectral
        distribution. If not specified, defaults to unity for all *LEDs*.
    shape
        Spectral shape used to create the spectral distribution.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.colorimetry.sd_single_led_Ohno2005`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Multi-*LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_multi_leds_Ohno2005(
    ...     np.array([457, 530, 615]),
    ...     np.array([20, 30, 20]),
    ...     np.array([0.731, 1.000, 1.660]),
    ... )
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[500]  # doctest: +ELLIPSIS
    np.float64(0.1295132...)
    """
    pass


SD_MULTI_LEDS_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Ohno 2005": sd_multi_leds_Ohno2005,
    }
)
SD_MULTI_LEDS_METHODS.__doc__ = """
Supported multi-*LED* spectral distribution computation methods.
"""


def sd_multi_leds(
    peak_wavelengths: ArrayLike,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    method: Literal["Ohno 2005"] | str = "Ohno 2005",
    **kwargs: Any,
) -> SpectralDistribution:
    """
    Generate a multi-*LED* spectral distribution with the specified spectral
    shape at specified peak wavelengths.

    Parameters
    ----------
    peak_wavelengths
        Wavelengths at which the multi-*LED* spectral distribution will
        peak, i.e., the peak wavelengths for each generated single *LED*
        spectral distribution.
    shape
        Spectral shape used to create the spectral distribution.
    method
        Computation method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.colorimetry.sd_multi_leds_Ohno2005`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Multi-*LED* spectral distribution.

    Notes
    -----
    -   By default, the spectral distribution will use the shape specified
        by :attr:`colour.SPECTRAL_SHAPE_DEFAULT` attribute.

    References
    ----------
    :cite:`Ohno2005`, :cite:`Ohno2008a`

    Examples
    --------
    >>> sd = sd_multi_leds(
    ...     np.array([457, 530, 615]),
    ...     half_spectral_widths=np.array([20, 30, 20]),
    ...     peak_power_ratios=np.array([0.731, 1.000, 1.660]),
    ... )
    >>> sd.shape
    SpectralShape(360.0, 780.0, 1.0)
    >>> sd[500]  # doctest: +ELLIPSIS
    np.float64(0.1295132...)
    """
    pass
