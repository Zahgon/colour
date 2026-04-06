"""
Pytest Configuration
====================

Configure *pytest* with array backend fixtures for *Array API* testing.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from colour.hints import Generator, ModuleType

from colour.utilities import array_api_enable

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xp",
]

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:
    torch = None


def _make_backend_parameters() -> list:
    """Build the parametrised backend list."""

    params = [pytest.param(np, id="numpy")]

    if jnp is not None:
        params.append(pytest.param(jnp, id="jax"))

    if torch is not None:
        params.append(pytest.param(torch, id="torch"))

    return params


@pytest.fixture(params=_make_backend_parameters())
def xp(request: pytest.FixtureRequest) -> Generator[ModuleType, None, None]:
    """
    Parametrised array namespace fixture.

    Yields :mod:`numpy` and, when available, :mod:`jax.numpy` and
    :mod:`torch`. Non-NumPy backends automatically enable Array API dispatch
    for the duration of the test.
    """

    backend = request.param

    if backend is np:
        yield backend
    else:
        with array_api_enable(True):
            yield backend
