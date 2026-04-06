"""Define the unit tests for the :mod:`colour.temperature.kang2002` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_Kang2002, xy_to_CCT_Kang2002
from colour.utilities import (
    ignore_numpy_errors,
    is_scipy_installed,
    xp_asarray,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXy_to_CCT_Kang2002",
    "TestCCT_to_xy_Kang2002",
]


class TestXy_to_CCT_Kang2002:
    """
    Define :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
    definition unit tests methods.
    """

    def test_xy_to_CCT_Kang2002(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition.
        """

        xp_assert_close(
            xy_to_CCT_Kang2002(
                xp_asarray([0.380528282812500, 0.376733530961114], xp=xp),
                {"method": "Nelder-Mead"},
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_Kang2002(
                xp_asarray([0.306374019533528, 0.316552869726577], xp=xp),
                {"method": "Nelder-Mead"},
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_Kang2002(
                xp_asarray([0.252472994438400, 0.252254791243654], xp=xp),
                {"method": "Nelder-Mead"},
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_CCT_Kang2002(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition n-dimensional arrays support.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        uv = xp_asarray([0.380528282812500, 0.376733530961114], xp=xp)
        CCT = np.asarray(xy_to_CCT_Kang2002(uv))

        uv = xp.tile(xp_asarray(uv, xp=xp), (6, 1))
        CCT = xp.tile(xp_asarray(CCT, xp=xp), (6,))
        xp_assert_close(xy_to_CCT_Kang2002(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_asarray(uv, xp=xp), (2, 3, 2), xp=xp)
        CCT = xp_reshape(xp_asarray(CCT, xp=xp), (2, 3), xp=xp)
        xp_assert_close(xy_to_CCT_Kang2002(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_Kang2002(self) -> None:
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition nan support.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_Kang2002(cases)


class TestCCT_to_xy_Kang2002:
    """
    Define :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
    unit tests methods.
    """

    def test_CCT_to_xy_Kang2002(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002`
        definition.
        """

        xp_assert_close(
            CCT_to_xy_Kang2002(xp_asarray([4000], xp=xp)),
            np.array([[0.380528282812500, 0.376733530961114]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_Kang2002(xp_asarray([7000], xp=xp)),
            np.array([[0.306374019533528, 0.316552869726577]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_Kang2002(xp_asarray([25000], xp=xp)),
            np.array([[0.252472994438400, 0.252254791243654]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_Kang2002(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = np.asarray(CCT_to_xy_Kang2002(CCT))

        CCT = xp.tile(xp_asarray(CCT, xp=xp), (6,))
        xy = xp.tile(xp_asarray(xy, xp=xp), (6, 1))
        xp_assert_close(CCT_to_xy_Kang2002(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = xp_reshape(xp_asarray(CCT, xp=xp), (2, 3), xp=xp)
        xy = xp_reshape(xp_asarray(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_xy_Kang2002(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_Kang2002(self) -> None:
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_Kang2002(cases)
