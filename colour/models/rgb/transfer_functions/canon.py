"""
Canon Log Encodings
===================

Define the *Canon Log* encodings.

-   :attr:`colour.models.CANON_LOG_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog`
-   :attr:`colour.models.CANON_LOG_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog`
-   :attr:`colour.models.CANON_LOG_2_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog2`
-   :attr:`colour.models.CANON_LOG_2_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog2`
-   :attr:`colour.models.CANON_LOG_3_ENCODING_METHODS`
-   :func:`colour.models.log_encoding_CanonLog3`
-   :attr:`colour.models.CANON_LOG_3_DECODING_METHODS`
-   :func:`colour.models.log_decoding_CanonLog3`

Notes
-----
-   :cite:`Canon2016` is available as a *Drivers & Downloads* *Software* for
    Windows 7 *Operating System*, a copy of the archive is hosted at
    this url: https://drive.google.com/open?id=0B_IQZQdc4Vy8ZGYyY29pMEVwZU0
-   :cite:`Canon2020` is available as a *Drivers & Downloads* *Software* for
    Windows 10 *Operating System*, a copy of the archive is hosted at
    this url: https://drive.google.com/open?id=1Vcz8RVIXgXL54lhZsOwGUjjVZRObZSc5

References
----------
-   :cite:`Canon2016` : Canon. (2016). Input Transform Version 201612 for EOS
    C300 Mark II. Retrieved August 23, 2016, from https://www.usa.canon.com/\
internet/portal/us/home/support/details/cameras/cinema-eos/eos-c300-mark-ii
-   :cite:`Canon2020` : Canon. (2020). Input Transform Version 202007 for EOS
    C300 Mark II. Retrieved July 16, 2023, from https://www.usa.canon.com/\
internet/portal/us/home/support/details/cameras/cinema-eos/eos-c300-mark-ii
-   :cite:`Thorpe2012a` : Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC.
    Retrieved September 25, 2014, from
    http://downloads.canon.com/CDLC/Canon-Log_Transfer_Characteristic_6-20-2012.pdf
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import Literal

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (
    CanonicalMapping,
    as_float,
    domain_range_scale,
    from_range_1,
    to_domain_1,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_encoding_CanonLog_v1",
    "log_decoding_CanonLog_v1",
    "log_encoding_CanonLog_v1_2",
    "log_decoding_CanonLog_v1_2",
    "CANON_LOG_ENCODING_METHODS",
    "log_encoding_CanonLog",
    "CANON_LOG_DECODING_METHODS",
    "log_decoding_CanonLog",
    "log_encoding_CanonLog2_v1",
    "log_decoding_CanonLog2_v1",
    "log_encoding_CanonLog2_v1_2",
    "log_decoding_CanonLog2_v1_2",
    "CANON_LOG_2_ENCODING_METHODS",
    "log_encoding_CanonLog2",
    "CANON_LOG_2_DECODING_METHODS",
    "log_decoding_CanonLog2",
    "log_encoding_CanonLog3_v1",
    "log_decoding_CanonLog3_v1",
    "log_encoding_CanonLog3_v1_2",
    "log_decoding_CanonLog3_v1_2",
    "CANON_LOG_3_ENCODING_METHODS",
    "log_encoding_CanonLog3",
    "CANON_LOG_3_DECODING_METHODS",
    "log_decoding_CanonLog3",
]


def log_encoding_CanonLog_v1(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log* v1 log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* v1 non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* v1 non-linear encoded data.

    References
    ----------
    :cite:`Canon2016`, :cite:`Thorpe2012a`

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
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog_v1(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389651...)

    The values of *Table 2 Canon-Log Code Values* table in
    :cite:`Thorpe2012a` are obtained as follows:

    >>> x = np.array([0, 2, 18, 90, 720]) / 100
    >>> np.around(log_encoding_CanonLog_v1(x) * (2**10 - 1)).astype(np.int_)
    array([ 128,  169,  351,  614, 1016])
    >>> np.around(log_encoding_CanonLog_v1(x, 10, False) * 100, 1)
    array([  7.3,  12. ,  32.8,  62.7, 108.7])
    """
    pass


def log_decoding_CanonLog_v1(
    clog: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log* v1 log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog
        *Canon Log* v1 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* v1 non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_decoding_CanonLog_v1(34.338965172606912 / 100)  # doctest: +ELLIPSIS
    np.float64(0.17999999...)
    """
    pass


def log_encoding_CanonLog_v1_2(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log* v1.2 log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* v1.2 non-linear data is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* v1.2 non-linear encoded data.

    References
    ----------
    :cite:`Canon2020`

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
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389649...)
    """
    pass


def log_decoding_CanonLog_v1_2(
    clog: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log* v1.2 log decoding inverse opto-electronic transfer

    function (OETF).

    Parameters
    ----------
    clog
        *Canon Log* v1.2 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* v1.2 non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog_v1_2(34.338964929528061 / 100)
    ... # doctest: +ELLIPSIS
    np.float64(0.17999999...)
    """
    pass


CANON_LOG_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog_v1,
        "v1.2": log_encoding_CanonLog_v1_2,
    }
)
CANON_LOG_ENCODING_METHODS.__doc__ = """
Supported *Canon Log* log encoding curve / opto-electronic transfer function
(OETF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log* non-linear encoded data.

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`, :cite:`Thorpe2012a`

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
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> log_encoding_CanonLog(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389649...)
    >>> log_encoding_CanonLog(0.18, method="v1") * 100  # doctest: +ELLIPSIS
    np.float64(34.3389651...)

    The values of *Table 2 Canon-Log Code Values* table in
    :cite:`Thorpe2012a` are obtained as follows:

    >>> x = np.array([0, 2, 18, 90, 720]) / 100
    >>> np.around(log_encoding_CanonLog(x, method="v1") * (2**10 - 1)).astype(np.int_)
    array([ 128,  169,  351,  614, 1016])
    >>> np.around(log_encoding_CanonLog(x, 10, False, method="v1") * 100, 1)
    array([  7.3,  12. ,  32.8,  62.7, 108.7])
    """
    pass


CANON_LOG_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog_v1,
        "v1.2": log_decoding_CanonLog_v1_2,
    }
)
CANON_LOG_DECODING_METHODS.__doc__ = """
Supported *Canon Log* log decoding curve / electro-optical transfer function
(EOTF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog(
    clog: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log* log decoding inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    clog
        *Canon Log* non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`, :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_decoding_CanonLog(34.338964929528061 / 100)  # doctest: +ELLIPSIS
    np.float64(0.17999999...)
    >>> log_decoding_CanonLog(34.338965172606912 / 100, method="v1")
    ... # doctest: +ELLIPSIS
    np.float64(0.17999999...)
    """
    pass


def log_encoding_CanonLog2_v1(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 2* v1 log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* v1 non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* v1 non-linear encoded data.

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
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_encoding_CanonLog2_v1(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(39.8254694...)
    """
    pass


def log_decoding_CanonLog2_v1(
    clog2: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 2* v1 log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog2
        *Canon Log 2* v1 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* v1 non-linear data is encoded with
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_decoding_CanonLog2_v1(39.825469498316735 / 100)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass


def log_encoding_CanonLog2_v1_2(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 2* v1.2 log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* v1.2 non-linear data is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* v1.2 non-linear encoded data.

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
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog2_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(39.8254692...)
    """
    pass


def log_decoding_CanonLog2_v1_2(
    clog2: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 2* v1.2 log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog2
        *Canon Log 2* v1.2 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* v1.2 non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog2_v1_2(39.825469256149191 / 100)
    ... # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass


CANON_LOG_2_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog2_v1,
        "v1.2": log_encoding_CanonLog2_v1_2,
    }
)
CANON_LOG_2_ENCODING_METHODS.__doc__ = """
Supported *Canon Log 2* log encoding curve / opto-electronic transfer function
(OETF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog2(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log 2* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 2* non-linear encoded data.

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
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog2(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(39.8254692...)
    """
    pass


CANON_LOG_2_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog2_v1,
        "v1.2": log_decoding_CanonLog2_v1_2,
    }
)
CANON_LOG_2_DECODING_METHODS.__doc__ = """
Supported *Canon Log 2* log decoding curve / electro-optical transfer function
(EOTF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog2(
    clog2: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log 2* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog2
        *Canon Log 2* non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 2* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog2``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog2(39.825469256149191 / 100)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass


def log_encoding_CanonLog3_v1(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 3* v1 log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* v1 non-linear data is encoded as
        normalised code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* v1 non-linear encoded data.

    Notes
    -----
    -   Introspection of the grafting points by Shaw, N. (2018) shows
        that the *Canon Log 3* v1 IDT was likely derived from its
        encoding curve as the latter is grafted at *+/-0.014*::

            >>> clog3 = 0.04076162
            >>> (clog3 - 0.073059361) / 2.3069815
            -0.014000000000000002
            >>> clog3 = 0.105357102
            >>> (clog3 - 0.073059361) / 2.3069815
            0.013999999999999997

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_encoding_CanonLog3_v1(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389369...)
    """
    pass


def log_decoding_CanonLog3_v1(
    clog3: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 3* v1 log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog3
        *Canon Log 3* v1 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* v1 non-linear data is encoded with
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`

    Examples
    --------
    >>> log_decoding_CanonLog3_v1(34.338936938868677 / 100)  # doctest: +ELLIPSIS
    np.float64(0.1800000...)
    """
    pass


def log_encoding_CanonLog3_v1_2(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 3* v1.2 log encoding opto-electronic transfer function
    (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* v1.2 non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* v1.2 non-linear encoded data.

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
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog3_v1_2(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389370...)
    """
    pass


def log_decoding_CanonLog3_v1_2(
    clog3: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
) -> Range1:
    """
    Apply the *Canon Log 3* v1.2 log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog3
        *Canon Log 3* v1.2 non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* v1.2 non-linear data is encoded with
        normalised code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog3_v1_2(34.338937037393549 / 100)
    ... # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass


CANON_LOG_3_ENCODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_encoding_CanonLog3_v1,
        "v1.2": log_encoding_CanonLog3_v1_2,
    }
)
CANON_LOG_3_ENCODING_METHODS.__doc__ = """
Supported *Canon Log 3* log encoding curve / opto-electronic transfer function
(OETF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_encoding_CanonLog3(
    x: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log 3* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    x
        Linear data :math:`x`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded as normalised
        code values.
    in_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *Canon Log 3* non-linear encoded data.

    Notes
    -----
    -   Introspection of the grafting points by Shaw, N. (2018) shows that
        the *Canon Log 3* v1 IDT was likely derived from its encoding curve
        as the latter is grafted at *+/-0.014*::

            >>> clog3 = 0.04076162
            >>> (clog3 - 0.073059361) / 2.3069815
            -0.014000000000000002
            >>> clog3 = 0.105357102
            >>> (clog3 - 0.073059361) / 2.3069815
            0.013999999999999997

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_encoding_CanonLog3(0.18) * 100  # doctest: +ELLIPSIS
    np.float64(34.3389370...)
    """
    pass


CANON_LOG_3_DECODING_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "v1": log_decoding_CanonLog3_v1,
        "v1.2": log_decoding_CanonLog3_v1_2,
    }
)
CANON_LOG_3_DECODING_METHODS.__doc__ = """
Supported *Canon Log 3* log decoding curve / electro-optical transfer function
(EOTF) methods.

References
----------
:cite:`Canon2016`, :cite:`Canon2020`
"""


def log_decoding_CanonLog3(
    clog3: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    method: Literal["v1", "v1.2"] | str = "v1.2",
) -> Range1:
    """
    Apply the *Canon Log 3* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    clog3
        *Canon Log 3* non-linear encoded data.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Canon Log 3* non-linear data is encoded with normalised
        code values.
    out_reflection
        Whether the light level :math:`x` to a camera is reflection.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``clog3``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Canon2016`, :cite:`Canon2020`

    Examples
    --------
    >>> log_decoding_CanonLog3(34.338937037393549 / 100)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass
