# -*- coding: utf-8 -*-
"""Tools for plotting graphs.

.. autosummary::
   :toctree: markov_helpers/graphs
"""
from __future__ import annotations

import typing as _ty
import functools as _ft
from numbers import Number as _Number

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import _tricks as _gt
from ._tricks import ArrayLike, Edge
from .. import _options as _op
from .. import markov as _mk
from .. import _utilities as _util


__all__ = [
    "StyleOptions",
    "GraphOptions",
    "GraphPlots",
    "NodeCollection",
    "DiEdgeCollection",
    "get_node_colours",
    "get_edge_colours",
    "linear_layout",
    "good_direction",
]
# =============================================================================
# Options
# =============================================================================


# pylint: disable=too-many-ancestors
class _ImageOptions(_op.AnyOptions):
    """Options for heatmaps

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting. You can also subscript attributes of attributes with
    dotted keys: `options['suboptions.name']`.

    Parameters
    ----------
    cmap : str|Colormap
        Colour map used to map numbers to colours. By default, `'YlOrBr'`.
    norm : Normalize
        Maps heatmap values to interval `[0, 1]` for `cmap`.
        By default: `Normalise(0, 1)`.
    vmin : float
        Lower bound of `norm`. By default: `0`.
    vmax : float
        Lower bound of `norm`. By default: `1`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: _op.Attrs = ('cmap',)
    _cmap: mpl.colors.Colormap
    norm: mpl.colors.Normalize
    """Maps heatmap values to interval `[0, 1]` for `cmap`."""

    def __init__(self, *args, **kwds) -> None:
        self._cmap = mpl.cm.get_cmap('YlOrBr')
        self.norm = mpl.colors.Normalize(0., 1.)
        super().__init__(*args, **kwds)

    @property
    def cmap(self) -> mpl.colors.Colormap:
        """Get the colour map, used to map numbers to colours.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, value: _ty.Union[str, mpl.colors.Colormap]) -> None:
        """Set the colour map, used to map numbers to colours.

        Does nothing if `value` is `None`. Converts to `Colormap` if `str`.
        """
        if value is None:
            pass
        elif isinstance(value, str):
            self._cmap = mpl.cm.get_cmap(value)
        elif isinstance(value, mpl.colors.Colormap):
            self._cmap = value
        else:
            raise TypeError("cmap must be `str` or `mpl.colors.Colormap`, not "
                            + type(value).__name__)

    @property
    def vmin(self) -> float:
        """The lower bound for the colour map.
        """
        return self.norm.vmin

    @vmin.setter
    def vmin(self, value: float) -> None:
        """Set the lower bound for the colour map.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmin = value

    @property
    def vmax(self) -> float:
        """The upper bound for the colour map.
        """
        return self.norm.vmax

    @vmax.setter
    def vmax(self, value: float) -> None:
        """Set the upper bound for the colour map.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmax = value

    def val_to_colour(self, values) -> np.ndarray:
        """Normalise and convert values to colours

        Parameters
        ----------
        values : array_like (N,)
            Values to convert to colours

        Returns
        -------
        cols : np.ndarray (N, 4)
            RGBA array representing colours.
        """
        return self._cmap(self.norm(values))
# pylint: enable=too-many-ancestors


# pylint: disable=too-many-ancestors
class StyleOptions(_ImageOptions):
    """Options for node/edge colours when drawing graphs.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting. You can also subscript attributes of attributes with
    dotted keys: `options['suboptions.name']`.

    Parameters
    ----------
    cmap : str|Colormap
        Maps the interval `[0, 1]` to colours. By default `'YlOrBr'`.
    norm : Normalize
        Maps heatmap values to the interval `[0, 1]` for `cmap`.
        By default: `Normalise(0, 1)`.
    vmin : float
        Lower bound of `norm`. By default `0`.
    vmax : float
        Lower bound of `norm`. By default `1`.
    key_attr : str
        Name of node/edge attribute used to determine colour.
    val_attr : str
        Name of node/edge attribute used to determine area/width.
    mult : float
        Scale factor between `node/edge[siz_attr]` and area/width.
    mut_scale : float
        Ratio of `FancyArrowPatch.mutation_scale` to `linewidth` (for edges).
    thresh : float
        Threshold on size value to be made visible.
    entity : str
        Type of graph element, 'node' or 'edge'

        All parameters are optional keywords. Any dictionary passed as
        positional parameters will be popped for the relevant items. Keyword
        parameters must be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: _op.Attrs = _ImageOptions.prop_attributes + ('entity',)
    # topology specifying options
    _method: str = 'get_node_attr'
    key_attr: str = 'key'
    """Name of node/edge attribute used to determine colour."""
    val_attr: str = 'value'
    """Name of node/edge attribute used to determine area/width."""
    mut_scale: float = 2.
    """Ratio of `FancyArrowPatch.mutation_scale` to `linewidth` (for edges)."""
    mult: float = 1.
    """Scale factor between `node/edge[siz_attr]` and area/width."""
    thresh: float = 1e-3

    def __init__(self, *args, **kwds) -> None:
        self._method = self._method
        self.key_attr = self.key_attr
        self.val_attr = self.val_attr
        self.mult = self.mult
        self.mut_scale = self.mut_scale
        self.thresh = self.thresh
        super().__init__(*args, **kwds)

    def to_colour(self, graph: _gt.GraphAttrs) -> np.ndarray:
        """Get sizes from graph attributes

        Parameters
        ----------
        graph : GraphAttrs
            The graph whose edge/node attributes set node area/edge colour.

        Returns
        -------
        sizes : np.ndarray
            Array of node areas/edge colours.
        """
        vals = getattr(graph, self._method)(self.key_attr)
        return self.val_to_colour(vals)

    def to_size(self, graph: _gt.GraphAttrs) -> np.ndarray:
        """Get sizes from graph attributes

        Parameters
        ----------
        graph : GraphAttrs
            The graph whose edge/node attributes set node area/edge width.

        Returns
        -------
        sizes : np.ndarray
            Array of node areas/edge widths.
        """
        return getattr(graph, self._method)(self.val_attr) * self.mult

    @property
    def entity(self) -> str:
        """Type of graph element, 'node' or 'edge'."""
        return self._method[4:-5]

    @entity.setter
    def entity(self, value: str) -> None:
        """Set the type of graph element, 'node' or 'edge'.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        elif value in {'node', 'edge'}:
            self._method = f"get_{value}_attr"
        else:
            raise ValueError(f"Entity must be 'node' or 'edge', not {value}")
# pylint: enable=too-many-ancestors


# pylint: disable=too-many-ancestors
class GraphOptions(_op.Options):
    """Options for drawing graphs.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting. You can also subscript attributes of attributes with
    dotted keys: `options['suboptions.name']`.

    Parameters
    ----------
    topology : TopologyOptions
        Topology specifying options, for creating graphs/reference for `judge`.
    layout : Callable[DiGraph -> Dict[Node, ArrayLike]]
        Function to compute node positions. Keywords passed to `set_layout`
        are saved.
    nodes : ImageOptions
        Options for mapping `node[attr]` to node colour/area.
    edges : ImageOptions
        Options for mapping `edge[attr]` to edge colour/thickness.
    rad : List[float]
        Curvature of edges: aspect ratio of the (isoceles) Bezier triangle for
        [good, bad] directions. Positive -> anticlockwise.
    judge : Callable[[graph, toplogy] -> ndarray[bool]]
        Function that decides which edges are good and which are bad.

        All parameters are optional keywords. Any dictionary passed as
        positional parameters will be popped for the relevant items. Keyword
        parameters must be valid keys, otherwise a `KeyError` is raised.
    """
    map_attributes: _op.Attrs = ('topology', 'nodes', 'edges')
    prop_attributes: _op.Attrs = ('layout',)
    topology: _mk.TopologyOptions
    """Topology specifying options for creating graphs/for `judge`."""
    nodes: StyleOptions
    """Options for mapping `node[attr]` to node colour/area."""
    edges: StyleOptions
    """Options for mapping `edge[attr]` to edge colour/thickness."""
    rad: _ty.List[float]
    """Curvature of edges: aspect ratio of the (isoceles) Bezier triangle for
    [good, bad] directions. Positive -> anticlockwise."""
    judge: _ty.Optional[Judger]
    """Function that decides which edges are good and which are bad."""
    layout: Layout
    """Function to compute node positions."""

    def __init__(self, *args, **kwds) -> None:
        self.topology = _mk.TopologyOptions(serial=True)
        self.nodes = StyleOptions(cmap='coolwarm', mult=600)
        self.nodes.entity = 'node'
        self.edges = StyleOptions(cmap='seismic', mult=5)
        self.edges.entity = 'edge'
        self.rad = [-0.7, 0.35]
        self.judge = good_direction
        self.layout = linear_layout
        super().__init__(*args, **kwds)

    def choose_rads(self, graph: _gt.MultiDiGraph) -> np.ndarray:
        """Choose curvature of each edge.

        Assigns `self.rad[0]` or `self.rad[1]` to each edge, depending on
        whether `self.judge(graph, self.topology)` returns `True` or `False`
        in that edge's position.

        Returns
        -------
        rads : ndarray[float] (E,)
            Curvature assigned to each edge: aspect ratio of the containing
            oval. Positive -> counter-clockwise.
        """
        if self.judge is None:
            good_drn = np.ones(len(graph.edges), bool)
        else:
            good_drn = self.judge(graph, self.topology)
        return np.where(good_drn, *self.rad)

    def set_layout(self, value: Layout, **kwds) -> None:
        """Set the layout function. `kwds` are saved.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.layout = _ft.partial(value, **kwds)
# pylint: enable=too-many-ancestors


# =============================================================================
# Plot graph
# =============================================================================


def get_node_colours(graph: _gt.GraphAttrs, data: str) -> _ty.Dict[str, np.ndarray]:
    """Collect values of node attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with nodes whose attributes we want.
    data : str
        Name of attribute to map to colour.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_nodes` related to
        colour values: `{'node_color', 'vmin', 'vmax'}`.
    """
    vals = graph.get_node_attr(data)
    vmin, vmax = vals.min(), vals.max()
    return {'node_color': vals, 'vmin': vmin, 'vmax': vmax}


def get_edge_colours(graph: _gt.GraphAttrs, data: str) -> _ty.Dict[str, np.ndarray]:
    """Collect values of edge attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with edges whose attributes we want.
    data : str
        Aattribute mapped to colour. Ignored if `graph` is a `MultiDiGraph`.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_edges` related to
        colour values: `{'edge_color', 'edge_vmin', 'edge_vmax'}`.
    """
    if isinstance(graph, _gt.MultiDiGraph) and data == 'key':
        vals = graph.edge_key()
    else:
        vals = graph.get_edge_attr(data)
    vmin, vmax = vals.min(), vals.max()
    return {'edge_color': vals, 'edge_vmin': vmin, 'edge_vmax': vmax}


def linear_layout(graph: nx.Graph, sep: ArrayLike = (1., 0.),
                  origin: ArrayLike = (0., 0.)) -> NodePos:
    """Layout graph nodes in a line.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph whose nodes need laying out.
    sep : ArrayLike, optional
        Separation of nodes along line, by default `(1.0, 0.0)`.
    origin : ArrayLike, optional
        Position of node 0, by default `(0.0, 0.0)`.

    Returns
    -------
    pos : _ty.Dict[Node, np.ndarray]
        Dictionary of node ids -> position vectors.
    """
    sep, origin = np.asarray(sep), np.asarray(origin)
    return {node: origin + pos * sep for pos, node in enumerate(graph.nodes)}


def good_direction(graph: _gt.MultiDiGraph, ideal: _mk.TopologyOptions) -> np.ndarray:
    """Which edges are in a good direction?

    Parameters
    ----------
    graph : MultiDiGraph, (N,E)
        The graph whose edges we're testing.
    ideal : TopologyOptions
        Description of the reference topology, which defines good directions.

    Returns
    -------
    good : np.ndarray[bool] (E,)
        True if the direction of the edge is similar to `ideal`.
    """
    edges = np.array(graph.edge_order)
    _, key_inds = _gt.list_edge_keys(graph, True)
    best_drn = np.array(ideal.directions)[key_inds]
    real_drn = edges[:, 1] - edges[:, 0]
    if ideal.ring:
        num = len(graph.nodes)
        real_drn = (real_drn + num/2) % num - num/2
    return real_drn * best_drn >= 0


# =============================================================================
# Edge collection
# =============================================================================


class NodeCollection:
    """A collection of node plots.

    Parameters
    ----------
    graph : GraphAttrs
        The graph being drawn.
    pos : Dict[Node, ArrayLike]|None, optional
        Place to plot each node, by default `None -> opts.layout(graph)`.
    axs : mpl.axes.Axes|None, optional
        The axes to draw the graph on, by default `None -> plt.gca()`.
    opts : GraphOptions|None, optional
        Options for drawing the graph, by default `None -> GraphOptions()`.
    """
    _nodes: _NodePlots
    _node_ids: _ty.List[_gt.Node]
    style: StyleOptions
    """Options for drawing the nodes."""
    node_pos: NodePos
    """Place to plot each node."""
    node_size: np.ndarray
    """Actual node sizes, after scaling by size.mult"""

    def __init__(self, graph: _gt.GraphAttrs,
                 pos: _ty.Optional[NodePos] = None,
                 axs: _ty.Optional[mpl.axes.Axes] = None,
                 opts: _ty.Optional[GraphOptions] = None, **kwds) -> None:
        self._node_ids = list(graph.nodes)
        opts = _util.default_eval(opts, GraphOptions)
        axs = _util.default_eval(axs, plt.gca)
        opts.pop_my_args(kwds)
        self.style = opts.nodes
        self.node_pos = _util.default(pos, opts.layout)
        if callable(self.node_pos):
            self.node_pos = self.node_pos(graph)

        self.node_size = self.style.to_size(graph)
        node_col = self.style.to_colour(graph)
        self.style.vmin = node_col.min()
        self.style.vmax = node_col.max()

        kwds.update(ax=axs, node_color=node_col, node_size=self.node_size,
                    edgecolors='k')
        self._nodes = nx.draw_networkx_nodes(graph, self.node_pos, **kwds)

    @property
    def collection(self) -> _NodePlots:
        """The underlying matplotlib objects"""
        return self._nodes

    def set_color(self, col_vals: ArrayLike) -> None:
        """Set node colour values

        Parameters
        ----------
        col_vals : ArrayLike[float] (N,)
            Values that produce node colours, before conversion to colours.
        """
        cols = self.style.val_to_colour(col_vals)
        self._nodes.set_color(cols)

    def set_sizes(self, node_siz: ArrayLike) -> None:
        """Set node sizes

        Parameters
        ----------
        node_siz : ArrayLike[float] (N,)
            Sizes of the nodes in graph units, before scaling to plot units.
        """
        self.node_size = np.asarray(node_siz) * self.style.mult
        self._nodes.set_sizes(self.node_size)

    def set_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes

        Parameters
        ----------
        pos : Dict[Node, ArrayLike]|None
            Place to plot each node.
        """
        self.node_pos = pos
        pos_array = np.array([pos[node] for node in self._node_ids])
        self._nodes.set_offsets(pos_array)

    def get_sizes(self) -> np.ndarray:
        """Get node sizes in plot units.

        Returns
        -------
        node_siz : ArrayLike[float] (N,)
            Node sizes after scaling to plot units.
        """
        return self.node_size
        # return self._nodes.get_sizes()

    def get_pos(self) -> NodePos:
        """Get node positions.

        Returns
        -------
        pos : Dict[Node, ArrayLike]|None
            Place to plot each node.
        """
        return self.node_pos
        # return dict(zip(self._node_ids, self._nodes.get_offsets()))


class DiEdgeCollection:
    """A collection of directed edge plots.

    Parameters
    ----------
    graph : GraphAttrs
        The graph being drawn.
    nodes : NodeCollection
        The result of drawing the nodes.
    axs : mpl.axes.Axes|None, optional
        The axes to draw the graph on, by default `None -> plt.gca()`.
    opts : GraphOptions|None, optional
        Options for drawing the graph, by default `None -> GraphOptions()`.
    """
    _edges: _ty.Dict[Edge, _EdgePlot]
    _node_ids: _ty.List[_gt.Node]
    style: StyleOptions
    """Options for drawing the edges."""

    def __init__(self, graph: _gt.GraphAttrs, nodes: NodeCollection,
                 axs: _ty.Optional[mpl.axes.Axes] = None,
                 opts: _ty.Optional[GraphOptions] = None, **kwds) -> None:
        self._node_ids = list(graph.nodes)
        opts = _util.default_eval(opts, GraphOptions)
        axs = _util.default_eval(axs, plt.gca)
        opts.pop_my_args(kwds)
        self.style = opts.edges

        edge_wid = self.style.to_size(graph)
        edge_col = self.style.to_colour(graph)
        self.style.vmin = edge_col.min()
        self.style.vmax = edge_col.max()

        kwds.update(ax=axs, edge_color=edge_col, width=edge_wid,
                    node_size=nodes.get_sizes(),
                    connectionstyle=f'arc3,rad={opts.rad[0]}')
        edges = nx.draw_networkx_edges(graph, nodes.get_pos(), **kwds)
        self._edges = dict(zip(graph.edges, edges))
        self.set_rads(opts.choose_rads(graph))
        self.set_widths(edge_wid)

    @property
    def collection(self) -> _ty.List[_EdgePlot]:
        """The underlying matplotlib objects"""
        return list(self.values())

    def __len__(self) -> int:
        return len(self._edges)

    def __getitem__(self, key: Edge) -> _EdgePlot:
        return self._edges[key]

    def __iter__(self) -> _ty.Iterable[Edge]:
        return iter(self._edges)

    def keys(self) -> _ty.Iterable[Edge]:
        """A view of edge dictionary keys"""
        return self._edges.keys()

    def values(self) -> _ty.Iterable[_EdgePlot]:
        """An iterable view the underlying matplotlib objects"""
        return self._edges.values()

    def items(self) -> _ty.Iterable[_ty.Tuple[Edge, _EdgePlot]]:
        """A view of edge dictionary items"""
        return self._edges.items()

    def set_color(self, col_vals: ArrayLike) -> None:
        """Set line colour values

        Parameters
        ----------
        col_vals : ArrayLike[float] (E,)
            Values that produce edge colours, before conversion to colours.
        """
        cols = self.style.val_to_colour(col_vals)
        cols = mpl.colors.to_rgba_array(cols)
        cols = np.broadcast_to(cols, (len(self), 4), True)
        for edge, col in zip(self.values(), cols):
            edge.set_color(col)

    def set_widths(self, edge_vals: ArrayLike) -> None:
        """Set line widths of edges

        Parameters
        ----------
        edge_vals : ArrayLike[float] (E,)
            Edge widths in graph units, before scaling to plot units.
        """
        edge_vals = np.broadcast_to(edge_vals, (len(self),), True)
        for edge, wid in zip(self.values(), edge_vals * self.style.mult):
            edge.set_linewidth(wid)
            edge.set_mutation_scale(max(self.style.mut_scale * wid, 1e-3))
            edge.set_visible(wid >= self.style.thresh)

    def set_node_sizes(self, node_siz: ArrayLike) -> None:
        """Set sizes of nodes

        Parameters
        ----------
        node_siz : ArrayLike[float] (N,)
            Node sizes after scaling to plot units.
        """
        siz_dict = dict(zip(self._node_ids, _util.repeatify(node_siz)))
        for edge, edge_plot in self.items():
            edge_plot.shrinkA = _to_marker_edge(siz_dict[edge[0]], 'o')
            edge_plot.shrinkB = _to_marker_edge(siz_dict[edge[1]], 'o')

    def set_node_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes.

        Parameters
        ----------
        pos : Dict[Node, ArrayLike]|None
            Place to plot each node.
        """
        for edge_id, edge_plot in self.items():
            edge_plot.set_position(pos[edge_id[0]], pos[edge_id[1]])

    def set_rads(self, rads: ArrayLike) -> None:
        """Set the curvature of the edges.

        Parameters
        ----------
        rads : ndarray[float] (E,)
            Curvature assigned to each edge: aspect ratio of the containing
            oval. Positive -> counter-clockwise.
        """
        rads = np.broadcast_to(np.asanyarray(rads).ravel(), (len(self),), True)
        for edge, rad in zip(self.values(), rads):
            edge.set_connectionstyle('arc3', rad=rad)


def _to_marker_edge(marker_size: _Number, marker: str) -> _Number:
    """Space to leave for node at end of fancy arrrow patch"""
    if marker in "s^>v<d":  # `large` markers need extra space
        return np.sqrt(2 * marker_size) / 2
    return np.sqrt(marker_size) / 2


class GraphPlots:
    """Class for plotting markov process as a graph.

    Parameters
    ----------
    graph : DiGraph
        Graph object describing model. Nodes have attributes `key` and
        `value`. Edges have attributes `key`, `value` and `pind` (if the
        model was a `SynapseParamModel`).
    pos : Dict[Node, ArrayLike]|None, optional
        Place to plot each node, by default `None -> opts.layout(graph)`.
    axs : mpl.axes.Axes|None, optional
        The axes to draw the graph on, by default `None -> plt.gca()`.
    opts : GraphOptions|None, optional
        Options for plotting the graph, by default `None -> GraphOptions()`.
    Other keywords passed to `opt` or `nx.draw_networkx_nodes` and
    `nx.draw_networkx_edges`.
    """
    nodes: NodeCollection
    """The collection of node plots."""
    edges: DiEdgeCollection
    """The collection of directed edge plots."""
    opts: GraphOptions
    """Options for plotting the graph."""

    def __init__(self, graph: _gt.GraphAttrs,
                 pos: _ty.Optional[NodePos] = None,
                 axs: _ty.Optional[mpl.axes.Axes] = None,
                 opts: _ty.Optional[GraphOptions] = None, **kws) -> None:
        self.opts = _util.default_eval(opts, GraphOptions)
        self.opts.pop_my_args(kws)
        axs = _util.default_eval(axs, plt.gca)

        self.nodes = NodeCollection(graph, pos, axs, self.opts, **kws)
        self.edges = DiEdgeCollection(graph, self.nodes, axs, self.opts, **kws)

    @property
    def collection(self) -> _ty.List[mpl.artist.Artist]:
        """The underlying matplotlib objects"""
        return [self.nodes.collection] + self.edges.collection

    def update(self, edge_vals: _ty.Optional[np.ndarray],
               node_vals: _ty.Optional[np.ndarray]) -> None:
        """Update plots.

        Parameters
        ----------
        edge_vals : None|np.ndarray (E,)
            Edge line widths.
        node_vals : None|np.ndarray (N,)
            Equilibrium distribution,for nodes sizes (area)
        """
        if edge_vals is not None:
            self.set_widths(edge_vals)
        if node_vals is not None:
            self.set_node_sizes(node_vals)

    def update_from(self, graph: _gt.GraphAttrs) -> None:
        """Update plots using a graph object.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph object describing model. Nodes have attributes `key` and
            `value`. Edges have attributes `key`, `value`.
        """
        edge_val = graph.get_edge_attr(self.opts.edges.val_attr)
        node_val = graph.get_node_attr(self.opts.nodes.val_attr)
        self.update(edge_val, node_val)

    def set_node_colors(self, cols: ArrayLike) -> None:
        """Set node colour values

        Parameters
        ----------
        col_vals : ArrayLike[float] (N,)
            Values that produce node colours, before conversion to colours.
        """
        self.nodes.set_color(cols)

    def set_edge_colors(self, cols: ArrayLike) -> None:
        """Set line colour values

        Parameters
        ----------
        col_vals : ArrayLike[float] (E,)
            Values that produce edge colours, before conversion to colours.
        """
        self.edges.set_color(cols)

    def set_node_sizes(self, node_vals: ArrayLike) -> None:
        """Set node sizes

        Parameters
        ----------
        node_siz : ArrayLike[float] (N,)
            Sizes of the nodes in graph units, before scaling to plot units.
        """
        self.nodes.set_sizes(node_vals)
        self.edges.set_node_sizes(self.nodes.get_sizes())

    def set_widths(self, edge_vals: ArrayLike) -> None:
        """Set line widths of edges

        Parameters
        ----------
        edge_vals : ArrayLike[float] (E,)
            Edge widths in graph units, before scaling to plot units.
        """
        self.edges.set_widths(np.asarray(edge_vals).ravel())

    def set_node_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes

        Parameters
        ----------
        pos : Dict[Node, ArrayLike]|None
            Place to plot each node.
        """
        self.nodes.set_pos(pos)
        self.edges.set_node_pos(pos)

    def set_rads(self, rads: ArrayLike) -> None:
        """Set the curvature of the edges.

        Parameters
        ----------
        rads : ndarray[float] (E,)
            Curvature assigned to each edge: aspect ratio of the containing
            oval. Positive -> counter-clockwise.
        """
        self.edges.set_rads(rads)


# =============================================================================
# Aliases
# =============================================================================
_NodePlots = mpl.collections.PathCollection
_EdgePlot = mpl.patches.FancyArrowPatch
NodePos = _ty.Dict[_gt.Node, ArrayLike]
Layout = _ty.Callable[[nx.Graph], NodePos]
Colour = _ty.Union[str, _ty.Sequence[float]]
Judger = _ty.Callable[[nx.Graph, _mk.TopologyOptions], np.ndarray]
