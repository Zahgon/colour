"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.cineon` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Cineon,
    log_encoding_Cineon,
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
    "TestLogEncoding_Cineon",
    "TestLogDecoding_Cineon",
]


class TestLogEncoding_Cineon:
    """
    Define :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition unit tests methods.
    """

    def test_log_encoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition.
        """

        xp_assert_close(
            log_encoding_Cineon(xp_asarray(0.0, xp=xp)),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Cineon(xp_asarray(0.18, xp=xp)),
            0.457319613085418,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Cineon(xp_asarray(1.0, xp=xp)),
            0.669599217986315,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition n-dimensional arrays support.
        """

        x = 0.18
        y = np.asarray(log_encoding_Cineon(xp_asarray(x, xp=xp)))

        x = xp.tile(xp_asarray(x, xp=xp), (6,))
        y = xp.tile(xp_asarray(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_asarray(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_asarray(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_asarray(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_asarray(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition domain and range scale support.
        """

        x = 0.18
        y = np.asarray(log_encoding_Cineon(xp_asarray(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Cineon(xp_asarray(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Cineon(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition nan support.
        """

        log_encoding_Cineon(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Cineon:
    """
    Define :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition unit tests methods.
    """

    def test_log_decoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition.
        """

        xp_assert_close(
            log_decoding_Cineon(xp_asarray(0.092864125122190, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Cineon(xp_asarray(0.457319613085418, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Cineon(xp_asarray(0.669599217986315, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = np.asarray(log_decoding_Cineon(xp_asarray(y, xp=xp)))

        y = xp.tile(xp_asarray(y, xp=xp), (6,))
        x = xp.tile(xp_asarray(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_asarray(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_asarray(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_asarray(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_asarray(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Cineon(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = np.asarray(log_decoding_Cineon(xp_asarray(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Cineon(xp_asarray(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Cineon(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition nan support.
        """

        log_decoding_Cineon(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
