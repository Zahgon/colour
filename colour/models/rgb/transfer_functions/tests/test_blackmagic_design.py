"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
blackmagic_design` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
)
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
    "TestOetf_BlackmagicFilmGeneration5",
    "TestOetf_inverse_BlackmagicFilmGeneration5",
]


class TestOetf_BlackmagicFilmGeneration5:
    """
    Define :func:`colour.models.rgb.transfer_functions.blackmagic_design.\
oetf_BlackmagicFilmGeneration5` definition unit tests methods.
    """

    def test_oetf_BlackmagicFilmGeneration5(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition.
        """

        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(xp_asarray(0.0, xp=xp)),
            0.092465753424658,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(xp_asarray(0.18, xp=xp)),
            0.383561643835617,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(xp_asarray(1.0, xp=xp)),
            0.530489624957305,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(xp_asarray(100.0, xp=xp)),
            0.930339851899973,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(xp_asarray(222.86, xp=xp)),
            0.999999631713769,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_BlackmagicFilmGeneration5(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition n-dimensional
        arrays support.
        """

        L = 0.18
        V = np.asarray(oetf_BlackmagicFilmGeneration5(xp_asarray(L, xp=xp)))

        L = xp.tile(xp_asarray(L, xp=xp), (6,))
        V = xp.tile(xp_asarray(V, xp=xp), (6,))
        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(L),
            V,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3), xp=xp)
        V = xp_reshape(xp_asarray(V, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(L),
            V,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3, 1), xp=xp)
        V = xp_reshape(xp_asarray(V, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            oetf_BlackmagicFilmGeneration5(L),
            V,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_oetf_BlackmagicFilmGeneration5(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition domain and range
        scale support.
        """

        L = 0.18
        V = np.asarray(oetf_BlackmagicFilmGeneration5(xp_asarray(L, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BlackmagicFilmGeneration5(xp_asarray(L * factor, xp=xp)),
                    V * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BlackmagicFilmGeneration5(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition nan support.
        """

        oetf_BlackmagicFilmGeneration5(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestOetf_inverse_BlackmagicFilmGeneration5:
    """
    Define :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition unit tests
    methods.
    """

    def test_oetf_inverse_BlackmagicFilmGeneration5(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition.
        """

        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(
                xp_asarray(0.092465753424658, xp=xp)
            ),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(
                xp_asarray(0.383561643835617, xp=xp)
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(
                xp_asarray(0.530489624957305, xp=xp)
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(
                xp_asarray(0.930339851899973, xp=xp)
            ),
            100.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(
                xp_asarray(0.999999631713769, xp=xp)
            ),
            222.86,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BlackmagicFilmGeneration5(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition
        n-dimensional arrays support.
        """

        V = 0.383561643835617
        L = np.asarray(oetf_inverse_BlackmagicFilmGeneration5(xp_asarray(V, xp=xp)))

        V = xp.tile(xp_asarray(V, xp=xp), (6,))
        L = xp.tile(xp_asarray(L, xp=xp), (6,))
        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V = xp_reshape(xp_asarray(V, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V = xp_reshape(xp_asarray(V, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_asarray(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            oetf_inverse_BlackmagicFilmGeneration5(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_oetf_inverse_BlackmagicFilmGeneration5(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition domain and
        range scale support.
        """

        V = 0.383561643835617
        L = np.asarray(oetf_inverse_BlackmagicFilmGeneration5(xp_asarray(V, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BlackmagicFilmGeneration5(
                        xp_asarray(V * factor, xp=xp)
                    ),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BlackmagicFilmGeneration5(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition nan
        support.
        """

        oetf_inverse_BlackmagicFilmGeneration5(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )
