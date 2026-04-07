"""
DICOM - Grayscale Standard Display Function
===========================================

Define the *DICOM - Grayscale Standard Display Function* electro-optical
transfer function (EOTF) and its inverse.

-   :func:`colour.models.eotf_inverse_DICOMGSDF`
-   :func:`colour.models.eotf_DICOMGSDF`

The Grayscale Standard Display Function is defined for the Luminance Range from
:math:`0.05` to :math:`4000 cd/m^2`. The minimum Luminance corresponds to the
lowest practically useful Luminance of cathode-ray-tube (CRT) monitors and the
maximum exceeds the unattenuated Luminance of very bright light-boxes used for
interpreting X-Ray mammography. The Grayscale Standard Display Function
explicitly includes the effects of the diffused ambient Illuminance.

References
----------
-   :cite:`NationalElectricalManufacturersAssociation2004b` : National
    Electrical Manufacturers Association. (2004). Digital Imaging and
    Communications in Medicine (DICOM) Part 14: Grayscale Standard Display
    Function. http://medical.nema.org/dicom/2004/04_14PU.PDF
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayReal

from colour.hints import (  # noqa: TC001
    Annotated,
    Domain1,
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
    "CONSTANTS_DICOMGSDF",
    "eotf_inverse_DICOMGSDF",
    "eotf_DICOMGSDF",
]

CONSTANTS_DICOMGSDF: Structure = Structure(
    a=-1.3011877,
    b=-2.5840191e-2,
    c=8.0242636e-2,
    d=-1.0320229e-1,
    e=1.3646699e-1,
    f=2.8745620e-2,
    g=-2.5468404e-2,
    h=-3.1978977e-3,
    k=1.2992634e-4,
    m=1.3635334e-3,
    A=71.498068,
    B=94.593053,
    C=41.912053,
    D=9.8247004,
    E=0.28175407,
    F=-1.1878455,
    G=-0.18014349,
    H=0.14710899,
    I=-0.017046845,
)
"""*DICOM Grayscale Standard Display Function* constants."""


def eotf_inverse_DICOMGSDF(
    L: Domain1,
    out_int: bool = False,
    constants: Structure | None = None,
) -> Annotated[NDArrayReal, 1]:
    """
    Apply the *DICOM - Grayscale Standard Display Function* inverse
    electro-optical transfer function (EOTF).

    Parameters
    ----------
    L
        *Luminance* :math:`L`.
    out_int
        Whether to return value as integer code value or float equivalent
        of a code value at a specified bit-depth.
    constants
        *DICOM - Grayscale Standard Display Function* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        Just-Noticeable Difference (JND) Index, :math:`j`.

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
    | ``J``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`NationalElectricalManufacturersAssociation2004b`

    Examples
    --------
    >>> eotf_inverse_DICOMGSDF(130.0662)  # doctest: +ELLIPSIS
    np.float64(0.5004862...)
    >>> eotf_inverse_DICOMGSDF(130.0662, out_int=True)
    np.int64(512)
    """
    pass


def eotf_DICOMGSDF(
    J: Domain1,
    in_int: bool = False,
    constants: Structure | None = None,
) -> Range1:
    """
    Apply the *DICOM - Grayscale Standard Display Function* electro-optical
    transfer function (EOTF).

    Parameters
    ----------
    J
        Just-Noticeable Difference (JND) Index, :math:`j`.
    in_int
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a specified bit-depth.
    constants
        *DICOM - Grayscale Standard Display Function* constants.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luminance* :math:`L`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``J``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`NationalElectricalManufacturersAssociation2004b`

    Examples
    --------
    >>> eotf_DICOMGSDF(0.500486263438448)  # doctest: +ELLIPSIS
    np.float64(130.0628647...)
    >>> eotf_DICOMGSDF(512, in_int=True)  # doctest: +ELLIPSIS
    np.float64(130.0652840...)
    """
    pass
