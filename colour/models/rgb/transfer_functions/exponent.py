"""
Basic and Monitor-Curve Exponent Transfer Functions
===================================================

Define the exponent transfer functions.

-   :func:`colour.models.exponent_function_basic`
-   :func:`colour.models.exponent_function_monitor_curve`

References
----------
-   :cite: `TheAcademyofMotionPictureArtsandSciences2020` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2020). Specification
    S-2014-006 - Common LUT Format (CLF) - A Common File Format for Look-Up
    Tables. Retrieved June 24, 2020, from http://j.mp/S-2014-006
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import sdiv, sdiv_mode

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import as_float, as_float_array, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "exponent_function_basic",
    "exponent_function_monitor_curve",
]


def exponent_function_basic(
    x: ArrayLike,
    exponent: ArrayLike = 1,
    style: (
        Literal[
            "basicFwd",
            "basicRev",
            "basicMirrorFwd",
            "basicMirrorRev",
            "basicPassThruFwd",
            "basicPassThruRev",
        ]
        | str
    ) = "basicFwd",
) -> NDArrayFloat:
    """
    Apply a *basic* exponent transfer function to the specified array.

    Parameters
    ----------
    x
        Exponentially encoded data :math:`x`.
    exponent
        Exponent value used for the conversion.
    style
        Specifies the behaviour for the exponentiation function to operate:

        -   *basicFwd*: *Basic Forward* exponential behaviour where the
            definition applies a basic power law using the exponent.
            Values less than zero are clamped.
        -   *basicRev*: *Basic Reverse* exponential behaviour where the
            definition applies a basic power law using the exponent.
            Values less than zero are clamped.
        -   *basicMirrorFwd*: *Basic Mirror Forward* exponential
            behaviour where the definition applies a basic power law
            using the exponent for values greater than or equal to zero
            and mirrors the function for values less than zero (i.e.,
            rotationally symmetric around the origin).
        -   *basicMirrorRev*: *Basic Mirror Reverse* exponential
            behaviour where the definition applies a basic power law
            using the exponent for values greater than or equal to zero
            and mirrors the function for values less than zero (i.e.,
            rotationally symmetric around the origin).
        -   *basicPassThruFwd*: *Basic Pass Forward* exponential
            behaviour where the definition applies a basic power law
            using the exponent for values greater than or equal to zero
            and passes values less than zero unchanged.
        -   *basicPassThruRev*: *Basic Pass Reverse* exponential
            behaviour where the definition applies a basic power law
            using the exponent for values greater than or equal to zero
            and passes values less than zero unchanged.

    Returns
    -------
    :class:`numpy.ndarray`
        Exponentially converted data.

    Examples
    --------
    >>> exponent_function_basic(0.18, 2.2)  # doctest: +ELLIPSIS
    np.float64(0.0229932...)
    >>> exponent_function_basic(-0.18, 2.2)
    np.float64(0.0)
    >>> exponent_function_basic(0.18, 2.2, "basicRev")  # doctest: +ELLIPSIS
    np.float64(0.4586564...)
    >>> exponent_function_basic(-0.18, 2.2, "basicRev")
    np.float64(0.0)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicMirrorFwd"
    ... )
    np.float64(0.0229932...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicMirrorFwd"
    ... )
    np.float64(-0.0229932...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicMirrorRev"
    ... )
    np.float64(0.4586564...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicMirrorRev"
    ... )
    np.float64(-0.4586564...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicPassThruFwd"
    ... )
    np.float64(0.0229932...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicPassThruFwd"
    ... )
    np.float64(-0.18)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, "basicPassThruRev"
    ... )
    np.float64(0.4586564...)
    >>> exponent_function_basic(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, "basicPassThruRev"
    ... )
    np.float64(-0.18)
    """
    pass


def exponent_function_monitor_curve(
    x: ArrayLike,
    exponent: ArrayLike = 1,
    offset: ArrayLike = 0,
    style: (
        Literal[
            "monCurveFwd",
            "monCurveRev",
            "monCurveMirrorFwd",
            "monCurveMirrorRev",
        ]
        | str
    ) = "monCurveFwd",
) -> NDArrayFloat:
    """
    Apply the *Monitor Curve* exponent transfer function to the specified array.

    Parameters
    ----------
    x
        Exponentially encoded data :math:`x`.
    exponent
        Exponent value used for the conversion.
    offset
        Offset value used for the conversion.
    style
        Specifies the behaviour for the exponentiation function to operate:

        -   *monCurveFwd*: *Monitor Curve Forward* exponential behaviour
            where the definition applies a power law function with a
            linear segment near the origin.
        -   *monCurveRev*: *Monitor Curve Reverse* exponential behaviour
            where the definition applies a power law function with a
            linear segment near the origin.
        -   *monCurveMirrorFwd*: *Monitor Curve Mirror Forward*
            exponential behaviour where the definition applies a power law
            function with a linear segment near the origin and mirrors the
            function for values less than zero (i.e., rotationally
            symmetric around the origin).
        -   *monCurveMirrorRev*: *Monitor Curve Mirror Reverse*
            exponential behaviour where the definition applies a power law
            function with a linear segment near the origin and mirrors the
            function for values less than zero (i.e., rotationally
            symmetric around the origin).

    Returns
    -------
    :class:`numpy.ndarray`
        Exponentially converted data.

    Examples
    --------
    >>> exponent_function_monitor_curve(0.18, 2.2, 0.001)  # doctest: +ELLIPSIS
    np.float64(0.0232240...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001
    ... )
    np.float64(-0.0002054...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, "monCurveRev"
    ... )
    np.float64(0.4581151...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveRev"
    ... )
    np.float64(-157.7302795...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 2, "monCurveMirrorFwd"
    ... )
    np.float64(0.1679399...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveMirrorFwd"
    ... )
    np.float64(-0.0232240...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     0.18, 2.2, 0.001, "monCurveMirrorRev"
    ... )
    np.float64(0.4581151...)
    >>> exponent_function_monitor_curve(  # doctest: +ELLIPSIS
    ...     -0.18, 2.2, 0.001, "monCurveMirrorRev"
    ... )
    np.float64(-0.4581151...)
    """
    pass
