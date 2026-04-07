"""
Munsell - ONNX
==============

Define *ONNX* model-based objects for *Munsell Renotation System* conversions.

The models are pre-trained neural networks from the *colour-science/
learning-munsell* *HuggingFace* repository that perform *Munsell* <-> *CIE xyY*
conversions with approximately 0.5 :math:`\\Delta E` accuracy.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Domain1,
        NDArrayFloat,
        NDArrayStr,
        Range1,
    )

from colour.notation.munsell.centore2014 import (
    _munsell_scale_factor,
    munsell_colour_to_munsell_specification,
    munsell_specification_to_munsell_colour,
)
from colour.utilities import (
    CACHE_REGISTRY,
    as_float_array,
    from_range_1,
    from_range_10,
    optional,
    required,
    to_domain_1,
    to_domain_10,
    url_download,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "HF_REPOSITORY_MUNSELL",
    "ONNX_MODELS_TO_XYY",
    "ONNX_MODELS_FROM_XYY",
    "normalization_parameters",
    "onnx_inference_session",
    "munsell_specification_to_xyY_Onnx",
    "munsell_colour_to_xyY_Onnx",
    "xyY_to_munsell_specification_Onnx",
    "xyY_to_munsell_colour_Onnx",
]

HF_REPOSITORY_MUNSELL: str = "colour-science/learning-munsell"
"""*HuggingFace* repository for *Munsell* *ONNX* models."""

ONNX_MODELS_TO_XYY: dict = {
    "model": "models/to_xyY/multi_mlp.onnx",
    "error_predictor": "models/to_xyY/multi_mlp_multi_error_predictor.onnx",
    "parameters": "models/to_xyY/multi_mlp_normalization_parameters.npz",
    "sha256": {
        "models/to_xyY/multi_mlp.onnx": (
            "b7b5a0535fd483721d51f76858f643b261def092fc6844f6ede10d21cc999c68"
        ),
        "models/to_xyY/multi_mlp.onnx.data": (
            "9d81a2e06aba8f49a089b4a37efb8170b3f2dbcf1ce126b147b7d2ee5a27d88c"
        ),
        "models/to_xyY/multi_mlp_multi_error_predictor.onnx": (
            "73b3b031051afc086fc08bb48fa1db4a740e86991a2bc1b520c81f7b4c164592"
        ),
        "models/to_xyY/multi_mlp_multi_error_predictor.onnx.data": (
            "b2115cd06dbda1314a3374fa30e4a6ae25720d7935aad580bde72c7dbf9bf2a9"
        ),
        "models/to_xyY/multi_mlp_normalization_parameters.npz": (
            "3c5d462fdefbb7c45cc9d469f73246423466cc2583f2c5094b9d5e76461c3541"
        ),
    },
}
"""
*ONNX* model configuration for *Munsell* specification to *CIE xyY*.

Keys
----
model
    Base model filename relative to the *HuggingFace* repository.
error_predictor
    Optional error predictor filename for two-stage inference.
parameters
    Normalization parameters filename.
"""

ONNX_MODELS_FROM_XYY: dict = {
    "model": "models/from_xyY/multi_mlp_class_code.onnx",
    "error_predictor": (
        "models/from_xyY/multi_mlp_class_code_aware_multi_error_predictor.onnx"
    ),
    "parameters": "models/from_xyY/multi_mlp_class_code_normalization_parameters.npz",
    "type": "class_code",
    "sha256": {
        "models/from_xyY/multi_mlp_class_code.onnx": (
            "da700d985f15fbc978fd1e3a657a79cc5130d9848a053b095eca94398468a248"
        ),
        "models/from_xyY/multi_mlp_class_code.onnx.data": (
            "ec71887eb1a17c161cc60588125381cf40b18ed9b8001637d0db47c6ac700bc7"
        ),
        "models/from_xyY/multi_mlp_class_code_aware_multi_error_predictor.onnx": (
            "3141aea2175f365eeb5a6a74773aa1939e70b8db8cd4639718a17e22be2e83a7"
        ),
        "models/from_xyY/multi_mlp_class_code_aware_multi_error_predictor.onnx.data": (
            "5a7a2ccb199b5a173d00c90abd44857accd996658aef75ed58a062983e58ceda"
        ),
        "models/from_xyY/multi_mlp_class_code_normalization_parameters.npz": (
            "96f041685cf0fc716dc601c2bd8f29a5c2b06242d5a6cfb3a27334e4e0a53e92"
        ),
    },
}
"""
*ONNX* model configuration for *CIE xyY* to *Munsell* specification.

Keys
----
model
    Base model filename relative to the *HuggingFace* repository.
error_predictor
    Optional error predictor filename for two-stage inference.
parameters
    Normalization parameters filename.
type
    Model type identifier for output decoding.
"""

_CACHE_ONNX_SESSIONS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_ONNX_SESSIONS"
)

_CACHE_NORMALIZATION_PARAMETERS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_NORMALIZATION_PARAMETERS"
)


def normalization_parameters(
    filename: str,
    sha256: dict | None = None,
) -> dict:
    """
    Download, load and cache normalization parameters.

    Parameters
    ----------
    filename
        Parameters filename relative to the *HuggingFace* repository root.
    sha256
        Mapping of filenames to *SHA-256* hashes for download verification.

    Returns
    -------
    :class:`dict`
        Dictionary with ``input_parameters`` and ``output_parameters`` keys.
    """
    pass


@required("onnxruntime")
def onnx_inference_session(
    filename: str,
    sha256: dict | None = None,
) -> typing.Any:
    """
    Load and cache an *ONNX* inference session.

    Parameters
    ----------
    filename
        Model filename relative to the *HuggingFace* repository root.
    sha256
        Mapping of filenames to *SHA-256* hashes for download verification.

    Returns
    -------
    :class:`onnxruntime.InferenceSession`
        *ONNX* inference session.
    """
    pass


@required("onnxruntime")
def munsell_specification_to_xyY_Onnx(
    specification: ArrayLike,
) -> NDArrayFloat:
    """
    Convert specified *Munsell* *Colorlab* specification to *CIE xyY*
    colourspace using the *ONNX* model.

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

    Examples
    --------
    >>> munsell_specification_to_xyY_Onnx(  # doctest: +SKIP
    ...     np.array([2.1, 8.0, 17.9, 4])
    ... )
    array([...])
    """
    pass


@required("onnxruntime")
def munsell_colour_to_xyY_Onnx(munsell_colour: ArrayLike) -> Range1:
    """
    Convert the specified *Munsell* colour to *CIE xyY* colourspace using
    the *ONNX* model.

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

    Examples
    --------
    >>> munsell_colour_to_xyY_Onnx("4.2YR 8.1/5.3")  # doctest: +SKIP
    array([...])
    """
    pass


@required("onnxruntime")
def xyY_to_munsell_specification_Onnx(xyY: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *Munsell* *Colorlab*
    specification using the *ONNX* model.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.

    Returns
    -------
    :class:`numpy.NDArrayFloat`
        *Munsell* *Colorlab* specification.

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

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_specification_Onnx(xyY)  # doctest: +SKIP
    array([...])
    """
    pass


@required("onnxruntime")
def xyY_to_munsell_colour_Onnx(
    xyY: Domain1,
    hue_decimals: int = 1,
    value_decimals: int = 1,
    chroma_decimals: int = 1,
) -> str | NDArrayStr:
    """
    Convert from *CIE xyY* colourspace to *Munsell* colour notation using
    the *ONNX* model.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.
    hue_decimals
        Number of decimal places for formatting the hue component.
    value_decimals
        Number of decimal places for formatting the value component.
    chroma_decimals
        Number of decimal places for formatting the chroma component.

    Returns
    -------
    :class:`str` or :class:`numpy.NDArrayStr`
        *Munsell* colour notation.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_colour_Onnx(xyY)  # doctest: +SKIP
    '...'
    """
    pass
