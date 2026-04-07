"""
Leica L-Log Log Encoding
========================

Define the *Leica L-Log* log encoding.

-   :func:`colour.models.log_encoding_LLog`
-   :func:`colour.models.log_decoding_LLog`

References
----------
-   :cite:`LeicaCameraAG2022` : Leica Camera AG. (2022). Leica L-Log Reference
    Manual. https://leica-camera.com/sites/default/files/\
pm-65976-210914__L-Log_Reference_Manual_EN.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
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
    "CONSTANTS_LLOG",
    "log_encoding_LLog",
    "log_decoding_LLog",
]

CONSTANTS_LLOG: Structure = Structure(
    cut1=0.006,
    cut2=0.1380,
    a=8,
    b=0.09,
    c=0.27,
    d=1.3,
    e=0.0115,
    f=0.6,
)
"""*Leica L-Log* constants."""


def log_encoding_LLog(
    LSR: Domain1,
    bit_depth: int = 10,
    out_normalised_code_value: bool = True,
    in_reflection: bool = True,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Leica L-Log* log encoding opto-electronic transfer function (OETF).

    Parameters
    ----------
    LSR
        Linear scene reflection :math:`LSR` values.
    bit_depth
        Bit-depth used for conversion.
    out_normalised_code_value
        Whether the *Leica L-Log* non-linear data :math:`L-Log` is encoded
        as normalised code values.
    in_reflection
        Whether the light level :math:`in` to a camera is reflection.
    constants
        *Leica L-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *L-Log* 10-bit equivalent code value :math:`L-Log`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LSR``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LLog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`LeicaCameraAG2022`

    Examples
    --------
    >>> log_encoding_LLog(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4353139...)
    """
    pass


def log_decoding_LLog(
    LLog: Domain1,
    bit_depth: int = 10,
    in_normalised_code_value: bool = True,
    out_reflection: bool = True,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *Leica L-Log* log decoding inverse opto-electronic transfer
    function (OETF).

    Parameters
    ----------
    LLog
        *L-Log* 10-bit equivalent code value :math:`L-Log`.
    bit_depth
        Bit-depth used for conversion.
    in_normalised_code_value
        Whether the *Leica L-Log* non-linear data :math:`L-Log` is encoded
        as normalised code values.
    out_reflection
        Whether the light level :math:`in` to a camera is reflection.
    constants
        *Leica L-Log* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear scene reflection :math:`LSR` values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LLog``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LSR``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`LeicaCameraAG2022`

    Examples
    --------
    >>> log_decoding_LLog(0.43531390404392656)  # doctest: +ELLIPSIS
    np.float64(0.1800000...)
    """
    pass
