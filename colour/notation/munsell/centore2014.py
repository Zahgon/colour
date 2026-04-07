"""
Munsell Renotation System - Centore (2014)
==========================================

Define *Centore (2014)* objects for *Munsell Renotation System* computations:

-   :func:`colour.notation.munsell.centore2014.munsell_specification_to_xyY`
-   :func:`colour.notation.munsell.centore2014.munsell_colour_to_xyY`
-   :func:`colour.notation.munsell.centore2014.xyY_to_munsell_specification`
-   :func:`colour.notation.munsell.centore2014.xyY_to_munsell_colour`

Notes
-----
-   The Munsell Renotation data commonly available within the *all.dat*,
    *experimental.dat* and *real.dat* files features *CIE xyY* colourspace
    values that are scaled by a :math:`1 / 0.975 \\simeq 1.02568` factor. If
    you are performing conversions using *Munsell* *Colorlab* specification,
    e.g., *2.5R 9/2*, according to *ASTM D1535-08e1* method, you should not
    scale the output :math:`Y` Luminance. However, if you use directly the
    *CIE xyY* colourspace values from the Munsell Renotation data, you should
    scale the :math:`Y` Luminance before conversions by a :math:`0.975`
    factor.

    *ASTM D1535-08e1* states that::

        The coefficients of this equation are obtained from the 1943 equation
        by multiplying each coefficient by 0.975, the reflectance factor of
        magnesium oxide with respect to the perfect reflecting diffuser, and
        rounding to ve digits of precision.

References
----------
-   :cite:`ASTMInternational1989a` : ASTM International. (1989). ASTM D1535-89
    - Standard Practice for Specifying Color by the Munsell System (pp. 1-29).
    Retrieved September 25, 2014, from
    http://www.astm.org/DATABASE.CART/HISTORICAL/D1535-89.htm
-   :cite:`Centore2012a` : Centore, P. (2012). An open-source inversion
    algorithm for the Munsell renotation. Color Research & Application, 37(6),
    455-464. doi:10.1002/col.20715
-   :cite:`Centore2014k` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellHueToASTMHue.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014l` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellSystemRoutines/LinearVsRadialInterpOnRenotationOvoid.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014m` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellToxyY.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014n` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/FindHueOnRenotationOvoid.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014o` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellSystemRoutines/BoundingRenotationHues.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014p` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/xyYtoMunsell.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014q` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellToxyForIntegerMunsellValue.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014r` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MaxChromaForExtrapolatedRenotation.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014s` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellHueToChromDiagHueAngle.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014t` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/ChromDiagHueAngleToMunsellHue.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014u` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    GeneralRoutines/CIELABtoApproxMunsellSpec.m.
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centorea` : Centore, P. (n.d.). The Munsell and Kubelka-Munk
    Toolbox. Retrieved January 23, 2018, from
    http://www.munsellcolourscienceforpainters.com/\
MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html
-   :cite:`Wikipedia2007c` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
"""

from __future__ import annotations

import re
import typing

import numpy as np

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    cartesian_to_cylindrical,
    euclidean_distance,
    polar_to_cartesian,
    sdiv_mode,
)
from colour.colorimetry import CCS_ILLUMINANTS, luminance_ASTMD1535
from colour.constants import (
    PATTERN_FLOATING_POINT_NUMBER,
    THRESHOLD_INTEGER,
    TOLERANCE_ABSOLUTE_DEFAULT,
    TOLERANCE_RELATIVE_DEFAULT,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Dict,
        Domain1,
        Literal,
        NDArrayFloat,
        NDArrayStr,
        Range1,
        Tuple,
    )

from colour.hints import ArrayLike, NDArrayFloat, cast
from colour.models import Lab_to_LCHab  # pyright: ignore
from colour.models import XYZ_to_Lab, XYZ_to_xy, xyY_to_XYZ
from colour.notation import MUNSELL_COLOURS_ALL
from colour.notation.munsell.value import munsell_value_ASTMD1535
from colour.utilities import (
    CACHE_REGISTRY,
    Lookup,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int_scalar,
    attest,
    domain_range_scale,
    from_range_1,
    from_range_10,
    get_domain_range_scale,
    is_caching_enabled,
    is_integer,
    is_numeric,
    to_domain_1,
    to_domain_10,
    tsplit,
    tstack,
    usage_warning,
)
from colour.volume import is_within_macadam_limits

__author__ = "Colour Developers, Paul Centore"
__copyright__ = "Copyright 2013 Colour Developers"
__copyright__ += ", "
__copyright__ += (
    "The Munsell and Kubelka-Munk Toolbox: Copyright  2010-2018 Paul Centore "
    "(Gales Ferry, CT 06335, USA); used by permission."
)
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MUNSELL_GRAY_PATTERN",
    "MUNSELL_COLOUR_PATTERN",
    "MUNSELL_GRAY_FORMAT",
    "MUNSELL_COLOUR_FORMAT",
    "MUNSELL_GRAY_EXTENDED_FORMAT",
    "MUNSELL_COLOUR_EXTENDED_FORMAT",
    "MUNSELL_HUE_LETTER_CODES",
    "ILLUMINANT_NAME_MUNSELL",
    "CCS_ILLUMINANT_MUNSELL",
    "munsell_specification_to_xyY_Centore2014",
    "munsell_colour_to_xyY_Centore2014",
    "xyY_to_munsell_specification_Centore2014",
    "xyY_to_munsell_colour_Centore2014",
    "parse_munsell_colour",
    "is_grey_munsell_colour",
    "normalise_munsell_specification",
    "munsell_colour_to_munsell_specification",
    "munsell_specification_to_munsell_colour",
    "xyY_from_renotation",
    "is_specification_in_renotation",
    "bounding_hues_from_renotation",
    "hue_to_hue_angle",
    "hue_angle_to_hue",
    "hue_to_ASTM_hue",
    "interpolation_method_from_renotation_ovoid",
    "xy_from_renotation_ovoid",
    "LCHab_to_munsell_specification",
    "maximum_chroma_from_renotation",
    "munsell_specification_to_xy",
]

MUNSELL_GRAY_PATTERN: str = f"N(?P<value>{PATTERN_FLOATING_POINT_NUMBER})"
MUNSELL_COLOUR_PATTERN: str = (
    f"(?P<hue>{PATTERN_FLOATING_POINT_NUMBER})\\s*"
    f"(?P<letter>BG|GY|YR|RP|PB|B|G|Y|R|P)\\s*"
    f"(?P<value>{PATTERN_FLOATING_POINT_NUMBER})\\s*\\/\\s*"
    f"(?P<chroma>[-+]?{PATTERN_FLOATING_POINT_NUMBER})"
)

MUNSELL_GRAY_FORMAT: str = "N{0}"
MUNSELL_COLOUR_FORMAT: str = "{0} {1}/{2}"
MUNSELL_GRAY_EXTENDED_FORMAT: str = "N{0:.{1}f}"
MUNSELL_COLOUR_EXTENDED_FORMAT: str = "{0:.{1}f}{2} {3:.{4}f}/{5:.{6}f}"

MUNSELL_HUE_LETTER_CODES: Lookup = Lookup(
    {
        "BG": 2,
        "GY": 4,
        "YR": 6,
        "RP": 8,
        "PB": 10,
        "B": 1,
        "G": 3,
        "Y": 5,
        "R": 7,
        "P": 9,
    }
)

ILLUMINANT_NAME_MUNSELL: str = "C"
CCS_ILLUMINANT_MUNSELL: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][ILLUMINANT_NAME_MUNSELL]

_CACHE_MUNSELL_SPECIFICATIONS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_MUNSELL_SPECIFICATIONS"
)
_CACHE_MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION"
)


def _munsell_specifications() -> NDArrayFloat:
    """
    Return the *Munsell Renotation System* specifications and cache them if
    not already existing.

    The *Munsell Renotation System* data is stored in
    :attr:`colour.notation.MUNSELL_COLOURS` attribute in a 2 columns form::

        (
            (("2.5GY", 0.2, 2.0), (0.713, 1.414, 0.237)),
            (("5GY", 0.2, 2.0), (0.449, 1.145, 0.237)),
            ...,
            (("7.5GY", 0.2, 2.0), (0.262, 0.837, 0.237)),
        )

    The first column is converted from *Munsell* colour to specification
    using
    :func:`colour.notation.munsell.munsell_colour_to_munsell_specification`
    definition:

    ('2.5GY', 0.2, 2.0) --> (2.5, 0.2, 2.0, 4)

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell Renotation System* specifications.
    """
    pass


def _munsell_maximum_chromas_from_renotation() -> Tuple[
    Tuple[Tuple[float, float], float], ...
]:
    """
    Return the maximum *Munsell* chromas from *Munsell Renotation System*
    data.

    Returns
    -------
    :class:`tuple`
        Maximum *Munsell* chromas.
    """
    pass


def _munsell_scale_factor() -> NDArrayFloat:
    """
    Return the domain-range scale factor for the *Munsell Renotation System*.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        Domain-range scale factor for the *Munsell Renotation System*.
    """
    pass


def _munsell_specification_to_xyY(specification: ArrayLike) -> NDArrayFloat:
    """
    Convert from *Munsell* *Colorlab* specification to *CIE xyY* colourspace.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace array.
    """
    pass


def munsell_specification_to_xyY_Centore2014(specification: ArrayLike) -> NDArrayFloat:
    """
    Convert specified *Munsell* *Colorlab* specification to *CIE xyY*
    colourspace.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace array.

    Notes
    -----
    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : 10       | 1             |
    |                   |                       |               |
    |                   | ``value``  : 10       | 1             |
    |                   |                       |               |
    |                   | ``chroma`` : 50       | 1             |
    |                   |                       |               |
    |                   | ``code``   : 10       | 1             |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | 1                     | 1             |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014m`

    Examples
    --------
    >>> munsell_specification_to_xyY_Centore2014(np.array([2.1, 8.0, 17.9, 4]))
    ... # doctest: +ELLIPSIS
    array([0.4400632..., 0.5522428..., 0.5761962...])
    >>> munsell_specification_to_xyY_Centore2014(
    ...     np.array([np.nan, 8.9, np.nan, np.nan])
    ... )
    ... # doctest: +ELLIPSIS
    array([0.31006  , 0.31616  , 0.7461345...])
    """
    pass


def munsell_colour_to_xyY_Centore2014(munsell_colour: ArrayLike) -> Range1:
    """
    Convert the specified *Munsell* colour to *CIE xyY* colourspace.

    Parameters
    ----------
    munsell_colour
        *Munsell* colour notation formatted as "H V/C" where H is hue,
        V is value, and C is chroma.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace array.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``xyY``   | 1                     | 1             |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> munsell_colour_to_xyY_Centore2014("4.2YR 8.1/5.3")  # doctest: +ELLIPSIS
    array([0.3873694..., 0.3575165..., 0.59362   ])
    >>> munsell_colour_to_xyY_Centore2014("N8.9")  # doctest: +ELLIPSIS
    array([0.31006  , 0.31616  , 0.7461345...])
    """
    pass


def _xyY_to_munsell_specification(xyY: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *Munsell* *Colorlab*
    specification.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the specified *CIE xyY* colourspace array is not within
        MacAdam limits.
    RuntimeError
        If the maximum iterations count has been reached without
        converging to a result.
    """
    pass


def xyY_to_munsell_specification_Centore2014(xyY: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *Munsell* *Colorlab*
    specification.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the specified *CIE xyY* colourspace array is not within
        MacAdam limits.
    RuntimeError
        If the maximum iterations count has been reached without
        converging to a result.

    Notes
    -----
    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | 1                     | 1             |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : 10       | 1             |
    |                   |                       |               |
    |                   | ``value``  : 10       | 1             |
    |                   |                       |               |
    |                   | ``chroma`` : 50       | 1             |
    |                   |                       |               |
    |                   | ``code``   : 10       | 1             |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014p`

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_specification_Centore2014(xyY)  # doctest: +ELLIPSIS
    array([4.2000019..., 8.0999999..., 5.2999996..., 6.        ])
    """
    pass


def xyY_to_munsell_colour_Centore2014(
    xyY: Domain1,
    hue_decimals: int = 1,
    value_decimals: int = 1,
    chroma_decimals: int = 1,
) -> str | NDArrayStr:
    """
    Convert from *CIE xyY* colourspace to *Munsell* colour notation.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array representing chromaticity coordinates
        and luminance.
    hue_decimals
        Number of decimal places for formatting the hue component.
    value_decimals
        Number of decimal places for formatting the value component.
    chroma_decimals
        Number of decimal places for formatting the chroma component.

    Returns
    -------
    :class:`str` or :class:`numpy.NDArrayFloat`
        *Munsell* colour notation formatted as "H V/C" where H is hue,
        V is value, and C is chroma.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_colour_Centore2014(xyY)
    '4.2YR 8.1/5.3'
    """
    pass


def parse_munsell_colour(munsell_colour: str) -> NDArrayFloat:
    """
    Parse specified *Munsell* colour and return an intermediate *Munsell*
    *Colorlab* specification.

    Parameters
    ----------
    munsell_colour
        *Munsell* colour.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        Intermediate *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the specified specification is not a valid *Munsell Renotation
        System* colour specification.

    Examples
    --------
    >>> parse_munsell_colour("N5.2")
    array([nan, 5.2, nan, nan])
    >>> parse_munsell_colour("0YR 2.0/4.0")
    array([0., 2., 4., 6.])
    """
    pass


def is_grey_munsell_colour(specification: ArrayLike) -> bool:
    """
    Determine whether the specified *Munsell* *Colorlab* specification
    represents a grey colour.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`bool`
        Whether the specification represents a grey colour.

    Examples
    --------
    >>> is_grey_munsell_colour(np.array([0.0, 2.0, 4.0, 6]))
    False
    >>> is_grey_munsell_colour(np.array([np.nan, 0.5, np.nan, np.nan]))
    True
    """
    pass


def normalise_munsell_specification(specification: ArrayLike) -> NDArrayFloat:
    """
    Normalise the specified *Munsell* *Colorlab* specification.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification to be normalised.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        Normalised *Munsell* *Colorlab* specification.

    Examples
    --------
    >>> normalise_munsell_specification(np.array([0.0, 2.0, 4.0, 6]))
    array([10.,  2.,  4.,  7.])
    >>> normalise_munsell_specification(np.array([np.nan, 0.5, np.nan, np.nan]))
    array([nan, 0.5, nan, nan])
    """
    pass


def munsell_colour_to_munsell_specification(
    munsell_colour: str,
) -> NDArrayFloat:
    """
    Convert from *Munsell* colour notation to *Munsell* *Colorlab*
    specification.

    Parameters
    ----------
    munsell_colour
        *Munsell* colour notation.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification as a 4-element array containing hue,
        value, chroma, and code values.

    Examples
    --------
    >>> munsell_colour_to_munsell_specification("N5.2")
    array([nan, 5.2, nan, nan])
    >>> munsell_colour_to_munsell_specification("0YR 2.0/4.0")
    array([10.,  2.,  4.,  7.])
    """
    pass


def munsell_specification_to_munsell_colour(
    specification: ArrayLike,
    hue_decimals: int = 1,
    value_decimals: int = 1,
    chroma_decimals: int = 1,
) -> str:
    """
    Convert from *Munsell* *Colorlab* specification to *Munsell* colour
    notation.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification as a 4-element array containing hue,
        value, chroma, and code values.
    hue_decimals
        Number of decimal places for hue formatting.
    value_decimals
        Number of decimal places for value formatting.
    chroma_decimals
        Number of decimal places for chroma formatting.

    Returns
    -------
    :class:`str`
        *Munsell* colour notation.

    Examples
    --------
    >>> munsell_specification_to_munsell_colour(np.array([np.nan, 5.2, np.nan, np.nan]))
    'N5.2'
    >>> munsell_specification_to_munsell_colour(np.array([10, 2.0, 4.0, 7]))
    '10.0R 2.0/4.0'
    """
    pass


def xyY_from_renotation(
    specification: ArrayLike,
    absolute_tolerance: float = TOLERANCE_ABSOLUTE_DEFAULT,
    relative_tolerance: float = TOLERANCE_RELATIVE_DEFAULT,
) -> NDArrayFloat:
    """
    Compute the *CIE xyY* colourspace vector for the specified *Munsell*
    *Colorlab* specification from *Munsell Renotation System* data.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.
    absolute_tolerance
        Absolute tolerance for finding the corresponding *Munsell
        Renotation System* data.
    relative_tolerance
        Relative tolerance for finding the corresponding *Munsell
        Renotation System* data.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xyY* colourspace vector.

    Raises
    ------
    ValueError
        If the specified specification does not exist in the *Munsell
        Renotation System* data.

    Examples
    --------
    >>> xyY_from_renotation(np.array([2.5, 0.2, 2.0, 4]))  # doctest: +ELLIPSIS
    array([0.71..., 1.41..., 0.23...])
    """
    pass


def is_specification_in_renotation(specification: ArrayLike) -> bool:
    """
    Determine whether the specified *Munsell* *Colorlab* specification exists
    in the *Munsell Renotation System* data.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`bool`
        Whether specification is in *Munsell Renotation System* data.

    Examples
    --------
    >>> is_specification_in_renotation(np.array([2.5, 0.2, 2.0, 4]))
    True
    >>> is_specification_in_renotation(np.array([64, 0.2, 2.0, 4]))
    False
    """
    pass


def bounding_hues_from_renotation(hue_and_code: ArrayLike) -> NDArrayFloat:
    """
    Return the two bounding hues from *Munsell Renotation System* data for
    a specified *Munsell* *Colorlab* specification hue and code.

    Parameters
    ----------
    hue_and_code
        *Munsell* *Colorlab* specification hue and *Munsell* *Colorlab*
        specification code.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        Bounding hues.

    References
    ----------
    :cite:`Centore2014o`

    Examples
    --------
    >>> bounding_hues_from_renotation([3.2, 4])
    array([[2.5, 4. ],
           [5. , 4. ]])

    # Coverage Doctests

    >>> bounding_hues_from_renotation([0.0, 1])
    array([[10.,  2.],
           [10.,  2.]])
    """
    pass


def hue_to_hue_angle(hue_and_code: ArrayLike) -> float:
    """
    Convert from *Munsell* *Colorlab* specification hue and code to hue angle
    in degrees.

    Parameters
    ----------
    hue_and_code
        *Munsell* *Colorlab* specification hue and *Munsell* *Colorlab*
        specification code.

    Returns
    -------
    :class:`float`
        Hue angle in degrees.

    References
    ----------
    :cite:`Centore2014s`

    Examples
    --------
    >>> hue_to_hue_angle([3.2, 4])  # doctest: +ELLIPSIS
    np.float64(65.5)
    """
    pass


def hue_angle_to_hue(hue_angle: float) -> NDArrayFloat:
    """
    Convert from hue angle in degrees to *Munsell* *Colorlab* specification hue
    and code.

    Parameters
    ----------
    hue_angle
        Hue angle in degrees.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification hue and *Munsell* *Colorlab*
        specification code.

    References
    ----------
    :cite:`Centore2014t`

    Examples
    --------
    >>> hue_angle_to_hue(65.54)  # doctest: +ELLIPSIS
    array([3.216, 4.   ])
    """
    pass


def hue_to_ASTM_hue(hue_and_code: ArrayLike) -> float:
    """
    Convert the *Munsell* *Colorlab* specification hue and code to *ASTM* hue
    number.

    Parameters
    ----------
    hue_and_code
        *Munsell* *Colorlab* specification hue and *Munsell* *Colorlab*
        specification code.

    Returns
    -------
    :class:`float`
        *ASTM* hue number.

    References
    ----------
    :cite:`Centore2014k`

    Examples
    --------
    >>> hue_to_ASTM_hue([3.2, 4])  # doctest: +ELLIPSIS
    np.float64(33.2...)
    """
    pass


def interpolation_method_from_renotation_ovoid(
    specification: ArrayLike,
) -> Literal["Linear", "Radial"] | None:
    """
    Determine the interpolation method for drawing ovoids in the *Munsell
    Renotation System*.

    Determine whether to use linear or radial interpolation when drawing
    ovoids through data points in the *Munsell Renotation System* data from
    the specified *Munsell* *Colorlab* specification.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`str` or :py:data:`None`
        Interpolation method.

    References
    ----------
    :cite:`Centore2014l`

    Examples
    --------
    >>> interpolation_method_from_renotation_ovoid([2.5, 5.0, 12.0, 4])
    'Radial'
    """
    pass


def xy_from_renotation_ovoid(specification: ArrayLike) -> NDArrayFloat:
    """
    Convert specified *Munsell* *Colorlab* specification to *CIE xy*
    chromaticity coordinates on *Munsell Renotation System* ovoid.

    The *CIE xy* point will be on the ovoid about the achromatic point,
    corresponding to the *Munsell* *Colorlab* specification value and
    chroma.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If an invalid interpolation method is retrieved from internal
        computations.

    References
    ----------
    :cite:`Centore2014n`

    Examples
    --------
    >>> xy_from_renotation_ovoid([2.5, 5.0, 12.0, 4])
    ... # doctest: +ELLIPSIS
    array([0.4333..., 0.5602...])
    >>> xy_from_renotation_ovoid([np.nan, 8, np.nan, np.nan])
    ... # doctest: +ELLIPSIS
    array([0.31006..., 0.31616...])
    """
    pass


def LCHab_to_munsell_specification(LCHab: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE L\\*C\\*Hab* colourspace to approximate *Munsell*
    *Colorlab* specification.

    Parameters
    ----------
    LCHab
        *CIE L\\*C\\*Hab* colourspace array.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification.

    References
    ----------
    :cite:`Centore2014u`

    Examples
    --------
    >>> LCHab = np.array([100, 17.50664796, 244.93046842])
    >>> LCHab_to_munsell_specification(LCHab)  # doctest: +ELLIPSIS
    array([ 8.0362412..., 10.        ,  3.5013295...,  1.        ])
    """
    pass


def maximum_chroma_from_renotation(hue_and_value_and_code: ArrayLike) -> float:
    """
    Return the maximum *Munsell* chroma from *Munsell Renotation System*
    data using the specified *Munsell* *Colorlab* specification hue, value, and
    code.

    Parameters
    ----------
    hue_and_value_and_code
        *Munsell* *Colorlab* specification hue, value, and code.

    Returns
    -------
    :class:`float`
        Maximum chroma.

    References
    ----------
    :cite:`Centore2014r`

    Examples
    --------
    >>> maximum_chroma_from_renotation([2.5, 5, 5])  # doctest: +ELLIPSIS
    np.float64(14.0)
    """
    pass


def munsell_specification_to_xy(specification: ArrayLike) -> NDArrayFloat:
    """
    Convert the specified *Munsell* *Colorlab* specification to *CIE xy*
    chromaticity coordinates by interpolating over *Munsell Renotation
    System* data.

    Parameters
    ----------
    specification
        *Munsell* *Colorlab* specification.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Centore2014q`

    Examples
    --------
    >>> munsell_specification_to_xy([2.1, 8.0, 17.9, 4])
    ... # doctest: +ELLIPSIS
    array([0.4400632..., 0.5522428...])
    >>> munsell_specification_to_xy([np.nan, 8, np.nan, np.nan])
    ... # doctest: +ELLIPSIS
    array([0.31006..., 0.31616...])
    """
    pass
