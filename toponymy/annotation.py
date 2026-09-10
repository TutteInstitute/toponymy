from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
    Self,
    Tuple,
    TypeVar,
)
from collections.abc import Iterator, Mapping, MutableMapping
from collections import defaultdict
from enum import Enum

T = TypeVar("T")


class NodeId(NamedTuple):
    layer: int
    cluster: int

    def __repr__(self):
        return f"NodeId({self.layer}, {self.cluster})"

    def __str__(self):
        return str(tuple(self))


@runtime_checkable
class _HasClusterAnnotationTree(Protocol):
    """Anything exposing a cluster_tree_."""

    @property
    def cluster_tree_(self) -> Mapping: ...


class InvalidAnnotationTree(ValueError):
    """Raised when the input mapping is not a single well-formed tree."""


class AnnotationTree:
    """Read-only navigation over a hierarchical cluster tree.

    Takes a  {parent: [children]} dict (or Mapping more generally)
    of the edges representing a cluster tree and allows it to be navigated.

    Node ids are (layer, cluster) pairs.

    Here are the assumptions about the cluster tree that must hold.

    * There is exactly one root node.
    * Every non-root node has exactly one parent.
    * Edges may span more than one layer.
    * The root is a catch-all node, and will not be considered a cluster node or a part of a layer.
    * Layer ids are 0-indexed and there are no empty layers (this assumption is used by the executor)

    Parameters
    ----------
    cluster_tree : Mapping of parent-child adjacencies
        Parent-to-child adjacency, of the condensed cluster treekeyed by (layer, cluster) pairs.

    Attributes
    ----------
    root : NodeId or None
        The unique node with no parent; None for an empty tree.

    Raises
    ------
    InvalidAnnotationTree
        If `validate` is True and the dict is not a single tree: a node
        with more than one parent, more than one root, or a cycle.

    Examples
    --------
    >>> cluster_tree = {(1, 0): [(0, 0)],
    ...         (1, 1): [(0, 1)],
    ...         (1, 2): [(0, 7), (0, 6), (0, 5)],
    ...         (1, 3): [(0, 4)],
    ...         (2, 0): [(1, 0), (1, 1), (1, 2), (1, 3), (0, 2), (0, 3)]}
    >>> tree = AnnotationTree(cluster_tree)
    >>> len(tree)
    13
    >>> tree.root
    NodeId(2, 0)
    >>> tree.n_layers
    2
    >>> tree.root.layer, tree.root.cluster
    (2, 0)
    """

    def __init__(
        self,
        cluster_tree: Dict[NodeId, Iterable[NodeId]],
        *,
        validate: bool = True,
    ) -> None:

        children_dict: Dict[NodeId, Tuple[NodeId, ...]] = {}
        parent_dict: Dict[NodeId, NodeId] = {}
        nodes: set = set()
        conflicts: List[Tuple[NodeId, NodeId, NodeId]] = []

        if len(cluster_tree) == 0:
            raise InvalidAnnotationTree("Empty cluster tree")

        self._cluster_tree = cluster_tree

        for raw_parent, raw_children in cluster_tree.items():
            parent_node = NodeId(*raw_parent)
            child_nodes = sorted(tuple(NodeId(*child) for child in raw_children))
            children_dict[parent_node] = child_nodes
            nodes.add(parent_node)
            for child_node in child_nodes:
                if child_node in parent_dict:
                    conflicts.append((child_node, parent_dict[child_node], parent_node))
                parent_dict[child_node] = parent_node
                nodes.add(child_node)

        roots = sorted(nodes - parent_dict.keys())

        if conflicts:
            node, first_parent, second_parent = conflicts[0]
            raise InvalidAnnotationTree(
                f"{node} has multiple parents ({first_parent} and {second_parent}); "
                f"{len(conflicts)} conflict(s) total"
            )
        if nodes and not roots:
            raise InvalidAnnotationTree("No root found")
        if len(roots) > 1:
            raise InvalidAnnotationTree(
                f"expected a single root, found multiple roots: {roots}"
            )

        self._children = children_dict
        self._parent = parent_dict
        self._set_nodes = frozenset(nodes - set(roots))
        self._sorted_nodes = tuple(sorted(self._set_nodes))
        self._root: Optional[NodeId] = roots[0]

        layers: Dict[int, List[NodeId]] = defaultdict(list)
        for node in self._set_nodes:
            layers[node.layer].append(node)
        self._layers = {
            layer_id: tuple(sorted(layer_nodes))
            for layer_id, layer_nodes in layers.items()
        }
        self._layer_ids = tuple(sorted(self._layers))

    @classmethod
    def from_clusterer(cls, clusterer: _HasClusterAnnotationTree, **kwargs) -> Self:
        return cls(clusterer.cluster_tree_, **kwargs)

    def __len__(self):
        return len(self._set_nodes)

    def __contains__(self, node: object) -> bool:
        try:
            return node in self._set_nodes
        except TypeError:
            return False

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, AnnotationTree):
            return NotImplemented
        return (
            self._root == other._root
            and self._set_nodes == other._set_nodes
            and self._parent == other._parent
        )

    @property
    def n_layers(self) -> int:
        return len(self._layer_ids)

    @property
    def nodes(self) -> Iterable[NodeId]:
        return self._sorted_nodes

    @property
    def layer_ids(self) -> Iterable[int]:
        return self._layer_ids

    def layer(self, layer_id: int) -> Iterable[NodeId]:
        return self._layers[layer_id]

    def children(self, node: NodeId) -> Iterable[NodeId] | None:
        return self._children.get(node, None)

    def parent(self, node: NodeId) -> NodeId | None:
        return self._parent.get(node, None)

    def descendants(
        self, node: NodeId, depth: Optional[int] = None
    ) -> Iterable[NodeId]:
        """
        Return descendants of a node within a certain depth.

        Parameters
        ----------
        node : NodeId
            Node whose descendants to return.
        depth : int or None
            Maximum number of edges to traverse downward.
            - None: traverse all reachable descendants
            - 0: return no descendants
            - k>0: include descendants up to k edges away

        Returns
        -------
        Iterable[NodeId]
        """
        if depth is not None and depth < 0:
            raise ValueError("depth must be >= 0 or None")

        if depth == 0:
            return []

        result: List[NodeId] = []
        frontier: List[NodeId] = [NodeId(*node)]
        remaining_depth = depth

        while frontier and (remaining_depth is None or remaining_depth > 0):
            next_frontier: List[NodeId] = []
            for parent in frontier:
                child_nodes = self._children.get(parent, ())
                result.extend(child_nodes)
                next_frontier.extend(child_nodes)
            frontier = next_frontier
            if remaining_depth is not None:
                remaining_depth -= 1

        return result


class AnnotationState(Enum):
    EMPTY = "empty"
    COMPUTED = "computed"
    FAILED = "failed"


class Annotation(MutableMapping[NodeId, T]):
    def __init__(self, name: str, tree: AnnotationTree):
        self.name = name
        self.tree = tree

        self._values: dict[NodeId, T] = {}
        self._states: dict[NodeId, AnnotationState] = {
            node: AnnotationState.EMPTY for node in tree.nodes
        }

    def __getitem__(self, node: NodeId) -> T:
        node = self._check(node)
        return self._values[node]

    def __setitem__(self, node: NodeId, value: T) -> None:
        node = self._check(node)
        self._values[node] = value
        self._states[node] = AnnotationState.COMPUTED

    def __delitem__(self, node: NodeId) -> None:
        node = self._check(node)
        self._values.pop(node, None)
        self._states[node] = AnnotationState.EMPTY

    def __iter__(self) -> Iterator[NodeId]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def states(self) -> Mapping[NodeId, AnnotationState]:
        return self._states

    def fail(self, node: NodeId) -> None:
        node = self._check(node)
        self._values.pop(node, None)
        self._states[node] = AnnotationState.FAILED

    def _check(self, node: Any) -> NodeId:
        if not isinstance(node, NodeId):
            try:
                node = NodeId(*node)
            except (TypeError, ValueError):
                raise KeyError(f"Invalid node ID: {node!r}")
        if node not in self.tree:
            raise KeyError(f"{node} is not a node of the cluster tree")
        return node
