"""Define the unit tests for the :mod:`colour.difference.din99` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import delta_E_DIN99
from colour.utilities import (
    domain_range_scale,
    ignore_numpy_errors,
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
    "TestDelta_E_DIN99",
]


class TestDelta_E_DIN99:
    """
    Define :func:`colour.difference.din99.delta_E_DIN99` definition unit
    tests methods.
    """

    def test_delta_E_DIN99(self, xp: ModuleType) -> None:
        """Test :func:`colour.difference.din99.delta_E_DIN99` definition."""

        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp),
                xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp),
            ),
            1.177216620111552,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([63.01090000, -31.09610000, -5.86630000], xp=xp),
                xp_asarray([62.81870000, -29.79460000, -4.08640000], xp=xp),
            ),
            0.987529977993114,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([35.08310000, -44.11640000, 3.79330000], xp=xp),
                xp_asarray([35.02320000, -40.07160000, 1.59010000], xp=xp),
            ),
            1.535894757971742,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # testing textiles boolean
        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp),
                xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp),
                textiles=True,
            ),
            1.215652775586509,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([63.01090000, -31.09610000, -5.86630000], xp=xp),
                xp_asarray([62.81870000, -29.79460000, -4.08640000], xp=xp),
                textiles=True,
            ),
            1.025997138865984,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(
                xp_asarray([35.08310000, -44.11640000, 3.79330000], xp=xp),
                xp_asarray([35.02320000, -40.07160000, 1.59010000], xp=xp),
                textiles=True,
            ),
            1.539922810033725,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # testing additional data boolean
        xp_assert_close(
            np.array(
                delta_E_DIN99(
                    xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp),
                    xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp),
                    additional_data=True,
                ).values
            ),
            np.array([1.1772166201115533, -0.17509302, -0.58040452, -1.00911446]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_delta_E_DIN99_method(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.din99.delta_E_DIN99` definition
        *method* parameter support.
        """

        Lab_1 = xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp)
        Lab_2 = xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp)

        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2, method="DIN99"),
            1.177216620111552,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2, method="DIN99b"),
            1.711312965743716,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2, method="DIN99c"),
            1.554667171681764,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2, method="DIN99d"),
            1.441930871002728,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.din99.delta_E_DIN99` definition
        n-dimensional arrays support.
        """

        Lab_1 = xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp)
        Lab_2 = xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp)
        delta_E = np.asarray(delta_E_DIN99(Lab_1, Lab_2))
        additional_data = delta_E_DIN99(Lab_1, Lab_2, additional_data=True)

        Lab_1 = xp.tile(xp_asarray(Lab_1, xp=xp), (6, 1))
        Lab_2 = xp.tile(xp_asarray(Lab_2, xp=xp), (6, 1))
        delta_E = xp.tile(xp_asarray(delta_E, xp=xp), (6,))
        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            np.array(delta_E_DIN99(Lab_1, Lab_2, additional_data=True).values),
            np.array(
                [
                    xp.tile(xp_asarray(val, xp=xp), (6,))
                    for val in additional_data.values
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab_1 = xp_reshape(xp_asarray(Lab_1, xp=xp), (2, 3, 3), xp=xp)
        Lab_2 = xp_reshape(xp_asarray(Lab_2, xp=xp), (2, 3, 3), xp=xp)
        delta_E = xp_reshape(xp_asarray(delta_E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            delta_E_DIN99(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            np.array(delta_E_DIN99(Lab_1, Lab_2, additional_data=True).values),
            np.array([np.full((2, 3), val) for val in additional_data.values]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_delta_E_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.difference.din99.delta_E_DIN99` definition
        domain and range scale support.
        """

        Lab_1 = xp_asarray([60.25740000, -34.00990000, 36.26770000], xp=xp)
        Lab_2 = xp_asarray([60.46260000, -34.17510000, 39.43870000], xp=xp)
        delta_E = np.asarray(delta_E_DIN99(Lab_1, Lab_2))
        additional_data = delta_E_DIN99(Lab_1, Lab_2, additional_data=True)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    delta_E_DIN99(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    np.array(
                        delta_E_DIN99(
                            Lab_1 * factor, Lab_2 * factor, additional_data=True
                        ).values
                    ),
                    np.array(additional_data.values),
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_delta_E_DIN99(self) -> None:
        """
        Test :func:`colour.difference.din99.delta_E_DIN99` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_DIN99(cases, cases)
        delta_E_DIN99(cases, cases, additional_data=True)
