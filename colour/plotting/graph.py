"""
Automatic Colour Conversion Graph Plotting
==========================================

Define the automatic colour conversion graph plotting objects.

-   :func:`colour.plotting.plot_automatic_colour_conversion_graph`
"""

from __future__ import annotations

import os
import typing

import colour
from colour.graph import CONVERSION_GRAPH_NODE_LABELS, describe_conversion_path

if typing.TYPE_CHECKING:
    from colour.hints import Literal

from colour.hints import cast
from colour.utilities import required, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_automatic_colour_conversion_graph",
]


@required("Pydot")
@required("NetworkX")
def plot_automatic_colour_conversion_graph(
    filename: str,
    prog: Literal["circo", "dot", "fdp", "neato", "nop", "twopi"] | str = "fdp",
) -> Dot:  # pyright: ignore  # noqa: F821  # pragma: no cover
    """
    Plot *Colour* automatic colour conversion graph using
    `Graphviz <https://www.graphviz.org>`__ and
    `pyraphviz <https://pygraphviz.github.io>`__.

    Parameters
    ----------
    filename
        Filename to use to save the image.
    prog
        *Graphviz* layout method.

    Returns
    -------
    :class:`pydot.Dot`
        *Pydot* graph.

    Notes
    -----
    -   This definition does not directly plot the *Colour* automatic
        colour conversion graph but instead writes it to an image.

    Examples
    --------
    >>> import tempfile
    >>> import colour
    >>> from colour import read_image
    >>> from colour.plotting import plot_image
    >>> filename = "{0}.png".format(tempfile.mkstemp()[-1])
    >>> _ = plot_automatic_colour_conversion_graph(filename, "dot")
    ... # doctest: +SKIP
    >>> plot_image(read_image(filename))  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Colour_Automatic_Conversion_Graph.png
        :align: center
        :alt: plot_automatic_colour_conversion_graph
    """
    pass
