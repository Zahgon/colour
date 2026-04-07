"""
Recommendation ITU-R BT.1361
============================

Define the *Recommendation ITU-R BT.1361* opto-electrical transfer function
(OETF) and its inverse.

-   :func:`colour.models.oetf_BT1361`
-   :func:`colour.models.oetf_inverse_BT1361`

References
----------
-   :cite:`InternationalTelecommunicationUnion1998` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.1361 - Worldwide
    unified colorimetry and related characteristics of future television and
    imaging systems (pp. 1-32). https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.1361-0-199802-W!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import (  # noqa: TC001
    Domain1,
    Range1,
)
from colour.models.rgb.transfer_functions import oetf_BT709, oetf_inverse_BT709
from colour.utilities import as_float, domain_range_scale, from_range_1, to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "oetf_BT1361",
    "oetf_inverse_BT1361",
]


def oetf_BT1361(L: Domain1) -> Range1:
    """
    Apply the *Recommendation ITU-R BT.1361* extended colour gamut system
    opto-electronic transfer function (OETF).

    Parameters
    ----------
    L
        Scene *Luminance* :math:`L`.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear primary signal :math:`E'`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p'``   | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion1998`

    Examples
    --------
    >>> oetf_BT1361(0.18)  # doctest: +ELLIPSIS
    np.float64(0.4090077288641...)
    >>> oetf_BT1361(-0.25)  # doctest: +ELLIPSIS
    np.float64(-0.25)
    >>> oetf_BT1361(1.33)  # doctest: +ELLIPSIS
    np.float64(1.1504846663972...)
    """
    pass


def oetf_inverse_BT1361(E_p: Domain1) -> Range1:
    """
    Apply the *Recommendation ITU-R BT.1361* extended colour gamut system
    inverse opto-electronic transfer function (OETF).

    Parameters
    ----------
    E_p
        Non-linear primary signal :math:`E'`.

    Returns
    -------
    :class:`numpy.ndarray`
        Scene *Luminance* :math:`L`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion1998`

    Examples
    --------
    >>> oetf_inverse_BT1361(0.4090077288641)  # doctest: +ELLIPSIS
    np.float64(0.1799999...)
    >>> oetf_inverse_BT1361(-0.25)  # doctest: +ELLIPSIS
    np.float64(-0.25)
    >>> oetf_inverse_BT1361(1.1504846663972)  # doctest: +ELLIPSIS
    np.float64(1.3299999...)
    """
    pass
