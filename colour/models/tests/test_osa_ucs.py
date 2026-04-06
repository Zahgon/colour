"""Define the unit tests for the :mod:`colour.models.osa_ucs` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import OSA_UCS_to_XYZ, XYZ_to_OSA_UCS
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
    "TestXYZ_to_OSA_UCS",
    "TestOSA_UCS_to_XYZ",
]


class TestXYZ_to_OSA_UCS:
    """
    Define :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_OSA_UCS(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition."""

        xp_assert_close(
            XYZ_to_OSA_UCS(
                xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
            ),
            np.array([-3.00499790, 2.99713697, -9.66784231]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_OSA_UCS(
                xp_asarray([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100
            ),
            np.array([-1.64657491, 4.59201565, 5.31738757]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_OSA_UCS(
                xp_asarray([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100
            ),
            np.array([-5.08589672, -7.91062749, 0.98107575]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_OSA_UCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition
        n-dimensional support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        Ljg = np.asarray(XYZ_to_OSA_UCS(XYZ))

        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        Ljg = xp.tile(xp_asarray(Ljg, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_OSA_UCS(XYZ), Ljg, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Ljg = xp_reshape(xp_asarray(Ljg, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_OSA_UCS(XYZ), Ljg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_OSA_UCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition domain
        and range scale support.
        """

        XYZ = xp_asarray([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        Ljg = np.asarray(XYZ_to_OSA_UCS(XYZ))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_OSA_UCS(XYZ * factor),
                    Ljg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_OSA_UCS(self) -> None:
        """
        Test :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_OSA_UCS(cases)


class TestOSA_UCS_to_XYZ:
    """
    Define :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_OSA_UCS_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition."""

        xp_assert_close(
            OSA_UCS_to_XYZ(
                xp_asarray([-3.00499790, 2.99713697, -9.66784231], xp=xp),
                {"disp": False},
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS * 500,
        )

        xp_assert_close(
            OSA_UCS_to_XYZ(
                xp_asarray([-1.64657491, 4.59201565, 5.31738757], xp=xp),
                {"disp": False},
            ),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS * 500,
        )

        xp_assert_close(
            OSA_UCS_to_XYZ(
                xp_asarray([-5.08589672, -7.91062749, 0.98107575], xp=xp),
                {"disp": False},
            ),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS * 500,
        )

    def test_n_dimensional_OSA_UCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition
        n-dimensional support.
        """

        Ljg = xp_asarray([-3.00499790, 2.99713697, -9.66784231], xp=xp)
        XYZ = np.asarray(OSA_UCS_to_XYZ(Ljg))

        Ljg = xp.tile(xp_asarray(Ljg, xp=xp), (6, 1))
        XYZ = xp.tile(xp_asarray(XYZ, xp=xp), (6, 1))
        xp_assert_close(OSA_UCS_to_XYZ(Ljg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Ljg = xp_reshape(xp_asarray(Ljg, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_asarray(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(OSA_UCS_to_XYZ(Ljg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_OSA_UCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition domain
        and range scale support.
        """

        Ljg = xp_asarray([-3.00499790, 2.99713697, -9.66784231], xp=xp)
        XYZ = np.asarray(OSA_UCS_to_XYZ(Ljg))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    OSA_UCS_to_XYZ(Ljg * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_OSA_UCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        OSA_UCS_to_XYZ(cases)
