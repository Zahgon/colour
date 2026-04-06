"""Define the unit tests for the :mod:`colour.geometry.intersection` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.geometry import (
    extend_line_segment,
    intersect_line_segments,
    intersect_ray_circle_2d,
)
from colour.utilities import xp_asarray, xp_assert_close, xp_assert_equal

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExtendLineSegment",
    "TestIntersectLineSegments",
    "TestIntersectRayCircle2D",
]


class TestExtendLineSegment:
    """
    Define :func:`colour.geometry.intersection.extend_line_segment` definition unit
    tests methods.
    """

    def test_extend_line_segment(self, xp: ModuleType) -> None:
        """Test :func:`colour.geometry.intersection.extend_line_segment` definition."""

        xp_assert_close(
            extend_line_segment(
                xp_asarray([0.95694934, 0.13720932], xp=xp),
                xp_asarray([0.28382835, 0.60608318], xp=xp),
            ),
            np.array([-0.5367248, 1.17765341]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            extend_line_segment(
                xp_asarray([0.95694934, 0.13720932], xp=xp),
                xp_asarray([0.28382835, 0.60608318], xp=xp),
                5,
            ),
            np.array([-3.81893739, 3.46393435]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            extend_line_segment(
                xp_asarray([0.95694934, 0.13720932], xp=xp),
                xp_asarray([0.28382835, 0.60608318], xp=xp),
                -1,
            ),
            np.array([1.1043815, 0.03451295]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIntersectLineSegments:
    """
    Define :func:`colour.geometry.intersection.intersect_line_segments`
    definition unit tests methods.
    """

    def test_intersect_line_segments(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.geometry.intersection.intersect_line_segments`
        definition.
        """

        l_1 = xp_asarray(
            [
                [[0.15416284, 0.7400497], [0.26331502, 0.53373939]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ],
            xp=xp,
        )
        l_2 = xp_asarray(
            [
                [[0.95694934, 0.13720932], [0.28382835, 0.60608318]],
                [[0.94422514, 0.85273554], [0.00225923, 0.52122603]],
                [[0.55203763, 0.48537741], [0.76813415, 0.16071675]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ],
            xp=xp,
        )

        s = intersect_line_segments(l_1, l_2)

        xp_assert_close(
            s.xy,
            np.array(
                [
                    [
                        [np.nan, np.nan],
                        [0.22791841, 0.60064309],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ],
                    [
                        [0.42814517, 0.50555685],
                        [0.30560559, 0.62798382],
                        [0.7578749, 0.17613012],
                        [np.nan, np.nan],
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(
            s.intersect,
            np.array([[False, True, False, False], [True, True, True, False]]),
        )

        xp_assert_equal(
            s.parallel,
            np.array([[False, False, False, False], [False, False, False, True]]),
        )

        xp_assert_equal(
            s.coincident,
            np.array([[False, False, False, False], [False, False, False, True]]),
        )


class TestIntersectRayCircle2D:
    """
    Define :func:`colour.geometry.intersection.intersect_ray_circle_2d`
    definition unit tests methods.
    """

    def test_intersect_ray_circle_2d(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.geometry.intersection.\
intersect_ray_circle_2d` definition.
        """

        # Ray pointing up from inside a circle.
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([0.0, 5.0], xp=xp), xp_asarray([0.0, 1.0], xp=xp), 10.0
                )
            )
        )
        assert d > 0.0
        xp_assert_close(d, 5.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # Ray pointing down from inside, should hit far side.
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([0.0, 5.0], xp=xp), xp_asarray([0.0, -1.0], xp=xp), 10.0
                )
            )
        )
        assert d > 0.0
        xp_assert_close(d, 15.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # No intersection (outside, pointing away).
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([0.0, 15.0], xp=xp), xp_asarray([0.0, 1.0], xp=xp), 10.0
                )
            )
        )
        assert d < 0.0

        # Horizontal ray from offset origin (3,0) -> hits circle r=5 at x=5.
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([3.0, 0.0], xp=xp), xp_asarray([1.0, 0.0], xp=xp), 5.0
                )
            )
        )
        assert d > 0.0
        xp_assert_close(d, 2.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # Tangent (touch only) returns negative.
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([0.0, 10.0], xp=xp), xp_asarray([1.0, 0.0], xp=xp), 10.0
                )
            )
        )
        assert d <= 0.0

        # Ray from origin pointing outward.
        d = float(
            np.asarray(
                intersect_ray_circle_2d(
                    xp_asarray([0.0, 0.0], xp=xp), xp_asarray([1.0, 0.0], xp=xp), 5.0
                )
            )
        )
        assert d > 0.0
        xp_assert_close(d, 5.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)
