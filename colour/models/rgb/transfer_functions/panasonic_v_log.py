"""
Panasonic V-Log Log Encoding
============================

Define the *Panasonic V-Log* log encoding.

-   :func:`colour.models.log_encoding_VLog`
-   :func:`colour.models.log_decoding_VLog`

References
----------
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut (pp.
    1-7).
    http://pro-av.panasonic.net/en/varicam/common/pdf/VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import Structure, as_float, from_range_1, optional, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANTS_VLOG",
    "log_encoding_VLog",
    "log_decoding_VLog",
]

CONSTANTS_VLOG: Structure = Structure(
    cut1=0.01, cut2=0.181, b=0.00873, c=0.241514, d=0.598206
)
"""*Panasonic V-Log* constants."""


def log_encoding_VLog(
    L_in: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Panasonic V-Log* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    L_in
        Linear reflection data :math:`L_{in}`.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Panasonic V-Log* non-linear data :math:`V_{out}` is
        encoded as normalised code values.
    in_reflection
        Whether the light level :math:`L_{in}` to a camera is reflection.
    constants
        *Panasonic V-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *Panasonic V-Log* mon-linear encoded data :math:`V_{out}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_encoding_VLog(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4233114...)

    The values of *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a`
    are obtained as follows:

    >>> L_in = np.array([0, 18, 90]) / 100
    >>> np.around(log_encoding_VLog(L_in, 10, False) * 100).astype(np.int_)
    array([ 7, 42, 61])
    >>> np.around(log_encoding_VLog(L_in) * (2**10 - 1)).astype(np.int_)
    array([128, 433, 602])
    >>> np.around(log_encoding_VLog(L_in) * (2**12 - 1)).astype(np.int_)
    array([ 512, 1733, 2409])

    Note that some values in the last column values of
    *Fig.2.2 V-Log Code Value* table in :cite:`Panasonic2014a` are different
    by a code: [512, 1732, 2408].
    """
    pass


def log_decoding_VLog(
    V_out: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Panasonic V-Log* log decoding inverse opto-electronic transfer

    function (OETF).

    Parameters
    ----------
    V_out
        *Panasonic V-Log* mon-linear encoded data :math:`V_{out}`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Panasonic V-Log* non-linear data :math:`V_{out}` is
        encoded as normalised code values.
    out_reflection
        Whether the light level :math:`L_{in}` to a camera is reflection.
    constants
        *Panasonic V-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear reflection data :math:`L_{in}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V_out``  | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_in``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> log_decoding_VLog(0.423311448760136)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    """
    pass
