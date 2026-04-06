"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.st_2084` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_inverse_ST2084, eotf_ST2084
from colour.utilities import (
    domain_range_scale,
    ignore_numpy_errors,
    xp_asarray,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestEotf_inverse_ST2084",
    "TestEotf_ST2084",
]


class TestEotf_inverse_ST2084:
    """
    Define :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition unit tests methods.
    """

    def test_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition.
        """

        xp_assert_close(
            eotf_inverse_ST2084(xp_asarray(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_asarray(100, xp=xp)),
            0.508078421517399,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_asarray(400, xp=xp)),
            0.652578597563067,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_asarray(5000, xp=xp), 5000),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition n-dimensional arrays support.
        """

        C = 100
        N = np.asarray(eotf_inverse_ST2084(xp_asarray(C, xp=xp)))

        C = xp.tile(xp_asarray(C, xp=xp), (6,))
        N = xp.tile(xp_asarray(N, xp=xp), (6,))
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

        C = xp_reshape(xp_asarray(C, xp=xp), (2, 3), xp=xp)
        N = xp_reshape(xp_asarray(N, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

        C = xp_reshape(xp_asarray(C, xp=xp), (2, 3, 1), xp=xp)
        N = xp_reshape(xp_asarray(N, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition domain and range scale support.
        """

        C = 100
        N = np.asarray(eotf_inverse_ST2084(xp_asarray(C, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_ST2084(xp_asarray(C * factor, xp=xp)),
                    N * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_ST2084(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition nan support.
        """

        eotf_inverse_ST2084(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ST2084:
    """
    Define :func:`colour.models.rgb.transfer_functions.st_2084.eotf_ST2084`
    definition unit tests methods.
    """

    def test_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition.
        """

        xp_assert_close(
            eotf_ST2084(xp_asarray(0.0, xp=xp)), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xp_assert_close(
            eotf_ST2084(xp_asarray(0.508078421517399, xp=xp)),
            100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_ST2084(xp_asarray(0.652578597563067, xp=xp)),
            400,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_ST2084(xp_asarray(1.0, xp=xp), 5000),
            5000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition n-dimensional arrays support.
        """

        N = 0.508078421517399
        C = np.asarray(eotf_ST2084(xp_asarray(N, xp=xp)))

        N = xp.tile(xp_asarray(N, xp=xp), (6,))
        C = xp.tile(xp_asarray(C, xp=xp), (6,))
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

        N = xp_reshape(xp_asarray(N, xp=xp), (2, 3), xp=xp)
        C = xp_reshape(xp_asarray(C, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

        N = xp_reshape(xp_asarray(N, xp=xp), (2, 3, 1), xp=xp)
        C = xp_reshape(xp_asarray(C, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition domain and range scale support.
        """

        N = 0.508078421517399
        C = np.asarray(eotf_ST2084(xp_asarray(N, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_ST2084(xp_asarray(N * factor, xp=xp)),
                    C * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_ST2084(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition nan support.
        """

        eotf_ST2084(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
