"""
X-Rite Data Input
=================

Define input functionality for X-Rite spectral data files.

-   :func:`colour.read_sds_from_xrite_file`
"""

from __future__ import annotations

import codecs
import re
import typing

from colour.colorimetry import SpectralDistribution

if typing.TYPE_CHECKING:
    from colour.hints import Dict, PathLike

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XRITE_FILE_ENCODING",
    "read_sds_from_xrite_file",
]

XRITE_FILE_ENCODING: str = "utf-8"


def read_sds_from_xrite_file(
    path: str | PathLike,
) -> Dict[str, SpectralDistribution]:
    """
    Read spectral data from the specified *X-Rite* file and convert it to a
    *dict* of :class:`colour.SpectralDistribution` class instances.

    Parameters
    ----------
    path
        Absolute *X-Rite* file path.

    Returns
    -------
    :class:`dict`
        *dict* of :class:`colour.SpectralDistribution` class instances.

    Raises
    ------
    IOError
        If the file cannot be read.

    Notes
    -----
    -   This parser is minimalistic and absolutely not bullet-proof.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> xrite_file = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "tests",
    ...     "resources",
    ...     "X-Rite_Digital_Colour_Checker.txt",
    ... )
    >>> sds_data = read_sds_from_xrite_file(xrite_file)
    >>> pprint(list(sds_data.keys()))  # doctest: +SKIP
    ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    """
    pass
