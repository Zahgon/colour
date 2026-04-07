"""
Sony Encodings
==============

Define the *Sony* log encodings.

-   :func:`colour.models.log_encoding_SLog`
-   :func:`colour.models.log_decoding_SLog`
-   :func:`colour.models.log_encoding_SLog2`
-   :func:`colour.models.log_decoding_SLog2`
-   :func:`colour.models.log_encoding_SLog3`
-   :func:`colour.models.log_decoding_SLog3`

References
----------
-   :cite:`SonyCorporation2012a` : Sony Corporation. (2012). S-Log2 Technical
    Paper (pp. 1-9). https://drive.google.com/file/d/\
1Q1RYri6BaxtYYxX0D4zVD6lAmbwmgikc/view?usp=sharing
-   :cite:`SonyCorporationd` : Sony Corporation. (n.d.). Technical Summary
    for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3 (pp. 1-7).
    http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (
    as_float,
    as_float_array,
    domain_range_scale,
    from_range_1,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_SLog",
    "log_decoding_SLog",
    "log_encoding_SLog2",
    "log_decoding_SLog2",
    "log_encoding_SLog3",
    "log_decoding_SLog3",
]


def log_encoding_SLog(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log* log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Sony S-Log* non-linear data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Sony S-Log* non-linear encoded data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    np.float64(0.3849708...)

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog(x, 10, False) * 100).astype(np.int_)
    array([ 3, 38, 65])
    >>> np.around(log_encoding_SLog(x) * (2**10 - 1)).astype(np.int_)
    array([ 90, 394, 636])
    """
    pass


def log_decoding_SLog(
    y: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    y
        *Sony S-Log* non-linear encoded data :math:`y`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Sony S-Log* non-linear data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the output light level :math:`x` represents reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog(0.384970815928670)  # doctest: +ELLIPSIS
    np.float64(0.1...)
    """
    pass


def log_encoding_SLog2(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log2* log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Sony S-Log2* non-linear data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Sony S-Log2* non-linear encoded data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    np.float64(0.3395325...)

    The values of *IRE and CV of S-Log2 @ISO800* table in
    :cite:`SonyCorporation2012a` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog2(x, 10, False) * 100).astype(np.int_)
    array([ 3, 32, 59])
    >>> np.around(log_encoding_SLog2(x) * (2**10 - 1)).astype(np.int_)
    array([ 90, 347, 582])
    """
    pass


def log_decoding_SLog2(
    y: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log2* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    y
        *Sony S-Log2* non-linear encoded data :math:`y`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Sony S-Log2* non-linear data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the output light level :math:`x` represents reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporation2012a`

    Examples
    --------
    >>> log_decoding_SLog2(0.339532524633774)  # doctest: +ELLIPSIS
    np.float64(0.1...)
    """
    pass


def log_encoding_SLog3(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log3* log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Sony S-Log3* non-linear data data :math:`y` is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Sony S-Log3* non-linear encoded data :math:`y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4105571...)

    The values of *S-Log3 10bit code values (18%, 90%)* table in
    :cite:`SonyCorporationd` are obtained as follows:

    >>> x = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_SLog3(x, 10, False) * 100).astype(np.int_)
    array([ 4, 41, 61])
    >>> np.around(log_encoding_SLog3(x) * (2**10 - 1)).astype(np.int_)
    array([ 95, 420, 598])
    """
    pass


def log_decoding_SLog3(
    y: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Sony S-Log3* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    y
        *Sony S-Log3* non-linear encoded data :math:`y`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Sony S-Log3* non-linear data :math:`y` is encoded as
        normalised code values.
    out_reflection
        Whether the output light level :math:`x` represents reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Reflection or :math:`IRE / 100` input light level :math:`x` to a
        camera.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyCorporationd`

    Examples
    --------
    >>> log_decoding_SLog3(0.410557184750733)  # doctest: +ELLIPSIS
    np.float64(0.1...)
    """
    pass
