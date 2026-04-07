"""
Academy Color Encoding System - Log Encodings
=============================================

Define the *Academy Color Encoding System* (ACES) log encodings.

-   :func:`colour.models.log_encoding_ACESproxy`
-   :func:`colour.models.log_decoding_ACESproxy`
-   :func:`colour.models.log_encoding_ACEScc`
-   :func:`colour.models.log_decoding_ACEScc`
-   :func:`colour.models.log_encoding_ACEScct`
-   :func:`colour.models.log_decoding_ACEScct`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES) (pp. 1-40). Retrieved December 19, 2014, from
    http://j.mp/TB-2014-004
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014r` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-012 - Academy Color Encoding System Version 1.0 Component
    Names (pp. 1-8). Retrieved December 19, 2014, from http://j.mp/TB-2014-012
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014s` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2013). Specification
    S-2013-001 - ACESproxy, an int Log Encoding of ACES Image Data.
    Retrieved December 19, 2014, from http://j.mp/S-2013-001
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014t` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Specification
    S-2014-003 - ACEScc, A Logarithmic Encoding of ACES Data for use within
    Color Grading Systems (pp. 1-12). Retrieved December 19, 2014, from
    http://j.mp/S-2014-003
-   :cite:`TheAcademyofMotionPictureArtsandSciences2016c` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project. (2016). Specification S-2016-001 -
    ACEScct, A Quasi-Logarithmic Encoding of ACES Data for use within Color
    Grading Systems. Retrieved October 10, 2016, from http://j.mp/S-2016-001
-   :cite:`TheAcademyofMotionPictureArtsandSciencese` : The Academy of Motion
    Picture Arts and Sciences, Science and Technology Council, & Academy Color
    Encoding System (ACES) Project Subcommittee. (n.d.). Academy Color Encoding
    System. Retrieved February 24, 2014, from
    http://www.oscars.org/science-technology/council/projects/aces.html
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import Literal, NDArrayInt

from colour.hints import (  # noqa: TC001
    Annotated,
    Domain1,
    NDArrayFloat,
    Range1,
)
from colour.utilities import (
    Structure,
    as_float,
    as_int,
    from_range_1,
    optional,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_ACES_PROXY_10",
    "CONSTANTS_ACES_PROXY_12",
    "CONSTANTS_ACES_PROXY",
    "CONSTANTS_ACES_CCT",
    "log_encoding_ACESproxy",
    "log_decoding_ACESproxy",
    "log_encoding_ACEScc",
    "log_decoding_ACEScc",
    "log_encoding_ACEScct",
    "log_decoding_ACEScct",
]

CONSTANTS_ACES_PROXY_10: Structure = Structure(
    CV_min=64,
    CV_max=940,
    steps_per_stop=50,
    mid_CV_offset=425,
    mid_log_offset=2.5,
)
"""*ACESproxy* 10 bit constants."""

CONSTANTS_ACES_PROXY_12: Structure = Structure(
    CV_min=256,
    CV_max=3760,
    steps_per_stop=200,
    mid_CV_offset=1700,
    mid_log_offset=2.5,
)
"""*ACESproxy* 12 bit constants."""

CONSTANTS_ACES_PROXY: dict = {
    10: CONSTANTS_ACES_PROXY_10,
    12: CONSTANTS_ACES_PROXY_12,
}
"""Aggregated *ACESproxy* constants."""

CONSTANTS_ACES_CCT: Structure = Structure(
    X_BRK=0.0078125,
    Y_BRK=0.155251141552511,
    A=10.5402377416545,
    B=0.0729055341958355,
)
"""*ACEScct* constants."""


def log_encoding_ACESproxy(
    lin_AP1: Domain1,
    bit_depth: Literal[10, 12] = 10,
    out_int: bool = False,
    constants: dict | None = None,
) -> Annotated[NDArrayFloat | NDArrayInt, 1]:
    """
    Apply the *ACESproxy* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    lin_AP1
        Linear *AP1* colourspace value.
    bit_depth
        *ACESproxy* bit-depth.
    out_int
        Whether to return value as int code value or float equivalent of a
        code value at a specified bit-depth.
    constants
        *ACESproxy* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *ACESproxy* non-linear encoded value.

    Notes
    -----
    +----------------+-----------------------+---------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``lin_AP1``    | 1                     | 1             |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range**      | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``ACESproxy``  | 1                     | 1             |
    +----------------+-----------------------+---------------+

    -   This definition has an output int switch, thus the domain-range
        scale information is only specified for the floating point mode.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014s`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACESproxy(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4164222...)
    >>> log_encoding_ACESproxy(0.18, out_int=True)
    np.int64(426)
    """
    pass


def log_decoding_ACESproxy(
    ACESproxy: Domain1,
    bit_depth: Literal[10, 12] = 10,
    in_int: bool = False,
    constants: dict | None = None,
) -> Range1:
    """
    Apply the *ACESproxy* log decoding inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    ACESproxy
        *ACESproxy* non-linear encoded value.
    bit_depth
        *ACESproxy* bit-depth.
    in_int
        Whether to treat the input value as integer code value or floating
        point equivalent of a code value at specified bit-depth.
    constants
        *ACESproxy* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear *AP1* colourspace value.

    Notes
    -----
    +----------------+-----------------------+---------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``ACESproxy``  | 1                     | 1             |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range**      | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``lin_AP1``    | 1                     | 1             |
    +----------------+-----------------------+---------------+

    -   This definition has an input int switch, thus the domain-range
        scale information is only specified for the floating point mode.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014s`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_decoding_ACESproxy(0.416422287390029)  # doctest: +ELLIPSIS
    np.float64(0.1...)
    >>> log_decoding_ACESproxy(426, in_int=True)  # doctest: +ELLIPSIS
    np.float64(0.1...)
    """
    pass


def log_encoding_ACEScc(lin_AP1: Domain1) -> Range1:
    """
    Apply the *ACEScc* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    lin_AP1
        Linear *AP1* colourspace value.

    Returns
    -------
    :class:`numpy.ndarray`
        *ACEScc* non-linear encoded value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScc``  | 1                     | 1             |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014t`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACEScc(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4135884...)
    """
    pass


def log_decoding_ACEScc(ACEScc: Domain1) -> Range1:
    """
    Apply the *ACEScc* log decoding inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    ACEScc
        *ACEScc* non-linear encoded value.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear *AP1* colourspace value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScc``  | 1                     | 1             |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014t`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_decoding_ACEScc(0.413588402492442)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass


def log_encoding_ACEScct(
    lin_AP1: Domain1, constants: Structure | None = None
) -> Range1:
    """
    Apply the *ACEScct* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    lin_AP1
        Linear *AP1* colourspace value.
    constants
        *ACEScct* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *ACEScct* non-linear encoded value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScct`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2016c`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_encoding_ACEScct(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4135884...)
    """
    pass


def log_decoding_ACEScct(
    ACEScct: Domain1, constants: Structure | None = None
) -> Range1:
    """
    Apply the *ACEScct* log decoding inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    ACEScct
        *ACEScct* non-linear encoded value.
    constants
        *ACEScct* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear *AP1* colourspace value.

    Notes
    -----
    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``ACEScct`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``lin_AP1`` | 1                     | 1             |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2016c`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> log_decoding_ACEScct(0.413588402492442)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass
