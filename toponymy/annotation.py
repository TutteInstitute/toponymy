from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
    Self,
    Sequence,
    Tuple,
    TypeVar,
)
from collections.abc import Iterator, Mapping, MutableMapping
from collections import defaultdict
from enum import Enum

T = TypeVar("T")

NODE_NODE = "node-node"


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
    * Layer ids are 0-indexed and there are no empty layers
    * Cluster ids are 0-indexed and the indices are contiguous within each layer

    Note: the last two assumptions are for backwards compatibility with the list of lists representations.
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

        self._root = roots[0]
        self._root_children = children_dict.get(self._root, ())
        self._parent = {
            child: parent
            for child, parent in parent_dict.items()
            if parent not in roots
        }
        self._children = {
            parent: children
            for parent, children in children_dict.items()
            if parent not in roots
        }
        self._set_nodes = frozenset(nodes - set(roots))
        self._sorted_nodes = tuple(sorted(self._set_nodes))

        layers: Dict[int, List[NodeId]] = defaultdict(list)
        for node in self._set_nodes:
            layers[node.layer].append(node)
        self._layers = {
            layer_id: tuple(sorted(layer_nodes))
            for layer_id, layer_nodes in layers.items()
        }

        for layer_id, layer_nodes in self._layers.items():
            cluster_ids = [node.cluster for node in layer_nodes]

            if cluster_ids and cluster_ids[0] != 0:
                raise InvalidAnnotationTree(
                    f"Cluster ids in layer {layer_id} must start at 0, got {cluster_ids[0]}"
                )

            expected = list(range(len(cluster_ids)))
            if cluster_ids != expected:
                raise InvalidAnnotationTree(
                    f"Cluster ids in layer {layer_id} must be contiguous, got {cluster_ids}"
                )

        self._layer_ids = tuple(sorted(self._layers))
        if self._layer_ids[0] != 0:
            raise InvalidAnnotationTree(
                f"Layer ids must start at 0, got {self._layer_ids[0]}"
            )

        if len(self._layer_ids) != max(self._layer_ids) + 1:
            raise InvalidAnnotationTree(
                f"Layer ids must be contiguous, got {self._layer_ids}"
            )

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

    def check(self, node: NodeId) -> NodeId:
        if not isinstance(node, NodeId):
            try:
                node = NodeId(*node)
            except (TypeError, ValueError):
                raise KeyError(f"Invalid node ID: {node!r}")
        if node not in self._set_nodes:
            raise KeyError(f"Node {node} is not in the annotation tree")
        return node

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
        node = self.tree.check(node)
        return self._values[node]

    def __setitem__(self, node: NodeId, value: T) -> None:
        node = self.tree.check(node)
        self._values[node] = value
        self._states[node] = AnnotationState.COMPUTED

    def __delitem__(self, node: NodeId) -> None:
        node = self.tree.check(node)
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
        node = self.tree.check(node)
        self._values.pop(node, None)
        self._states[node] = AnnotationState.FAILED

    # def _check(self, node: Any) -> NodeId:
    #     if not isinstance(node, NodeId):
    #         try:
    #             node = NodeId(*node)
    #         except (TypeError, ValueError):
    #             raise KeyError(f"Invalid node ID: {node!r}")
    #     if node not in self.tree:
    #         raise KeyError(f"{node} is not a node of the cluster tree")
    #     return node

    @classmethod
    def from_layered_list(
        cls, name: str, tree: AnnotationTree, layered_list: list[NodeId]
    ) -> Self:
        annotation = cls(name, tree)
        for node in tree.nodes:
            try:
                annotation[node] = layered_list[node.layer][node.cluster]
            except IndexError:
                raise IndexError(
                    f"Layered list does not contain a value for node {node}"
                )
        return annotation

    def to_layered_list(self) -> list[list[NodeId]]:
        layered_list: list[list[NodeId]] = []
        if not all(
            state is AnnotationState.COMPUTED for state in self._states.values()
        ):
            raise ValueError("Not all nodes have computed values")
        for layer_id in self.tree.layer_ids:
            layer_nodes = [self._values[node] for node in self.tree.layer(layer_id)]
            layered_list.append(layer_nodes)
        return layered_list


class AnnotationStore(Mapping[str, Annotation[Any]]):
    """Annotations by name, for one tree.

    The public collection interface is read-only. Annotations are added
    through `add()`, which validates their identity and tree.
    """

    def __init__(
        self,
        tree: AnnotationTree,
        annotations: Iterable[Annotation[Any]] = (),
    ):
        self.tree = tree
        self._annotations: dict[str, Annotation[Any]] = {}

        for annotation in annotations:
            self.add(annotation)

    def __getitem__(self, name: str) -> Annotation[Any]:
        return self._annotations[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._annotations)

    def __len__(self) -> int:
        return len(self._annotations)

    def __getattr__(self, name: str) -> Annotation[Any]:
        """
        Return the annotation with the given name as an attribute.

        Raises AttributeError if the annotation does not exist.
        """
        # Called only after normal attribute lookup fails.
        annotations = self.__dict__.get("_annotations", {})

        try:
            return annotations[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no annotation " f"or attribute {name!r}"
            ) from None

    def add(
        self,
        annotation: Annotation[Any],
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(annotation, Annotation):
            raise TypeError("annotation must be an Annotation")

        name = annotation.name

        if annotation.tree is not self.tree:
            raise ValueError(f"annotation {name!r} is bound to a different tree")

        if name in self._annotations and not replace:
            raise ValueError(
                f"annotation {name!r} is already in the store; "
                "pass replace=True to replace it"
            )

        if hasattr(type(self), name) or name in self.__dict__:
            raise ValueError(
                f"annotation name {name!r} collides with "
                "an AnnotationStore attribute; use store[{name!r}]"
            )

        self._annotations[name] = annotation

    def node(self, node: NodeId) -> dict[str, Any]:
        """Return computed annotation values for one node."""
        node = self.tree.check(node)

        return {
            name: annotation[node]
            for name, annotation in self._annotations.items()
            if annotation.states[node] is AnnotationState.COMPUTED
        }

    def node_states(self, node: NodeId) -> dict[str, AnnotationState]:
        """Return the state of every annotation for one node."""
        node = self.tree.check(node)

        return {
            name: annotation.states[node]
            for name, annotation in self._annotations.items()
        }

    def nodes(
        self,
        nodes: Iterable[NodeId],
    ) -> dict[NodeId, dict[str, Any]]:
        return {self.tree.check(node): self.node(node) for node in nodes}

    def layer(self, layer_id: int) -> dict[NodeId, dict[str, Any]]:
        return self.nodes(self.tree.layer(layer_id))

    def __repr__(self) -> str:
        if not self._annotations:
            return "AnnotationStore(empty)"

        total = len(self.tree)
        parts = ", ".join(
            f"{name}[{len(annotation)}/{total}]"
            for name, annotation in self._annotations.items()
        )
        return f"AnnotationStore({parts})"


@runtime_checkable
class Annotator(Protocol):
    """Structural contract for an annotation algorithm."""

    inputs: Sequence[str]
    outputs: Sequence[str]
    algorithm_type: str

    def annotate(
        self,
        node: NodeId,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...


class Executor:

    def run(
        self,
        annotator: Annotator,
        store: AnnotationStore,
        *,
        nodes: Iterable[NodeId] | None = None,
    ) -> dict[NodeId, str]:
        """Run one node-to-node annotator over selected nodes.

        Return ``{node: reason}`` for nodes whose annotation failed."""

        if not isinstance(annotator, Annotator):
            raise TypeError(
                "an Annotator needs inputs, outputs, algorithm_type, "
                "and annotate(node, **kwargs)"
            )

        if annotator.algorithm_type != NODE_NODE:
            raise NotImplementedError(
                f"algorithm_type {annotator.algorithm_type!r} " "is not supported yet"
            )

        inputs = tuple(annotator.inputs)
        outputs = tuple(annotator.outputs)

        if len(set(inputs)) != len(inputs):
            raise ValueError("annotator.inputs contains duplicate names")

        if len(set(outputs)) != len(outputs):
            raise ValueError("annotator.outputs contains duplicate names")

        missing = [name for name in inputs if name not in store]
        if missing:
            raise KeyError(f"missing input annotation(s): {missing}")

        selected_nodes = (
            store.tree.nodes
            if nodes is None
            else tuple(store.tree.check(node) for node in nodes)
        )

        for name in outputs:
            if name not in store:
                store.add(Annotation(name, store.tree))

        failures: dict[NodeId, str] = {}

        for node in selected_nodes:
            try:
                values = {name: store[name][node] for name in inputs}
                result = annotator.annotate(node, **values)
            except Exception as error:
                for name in outputs:
                    store[name].fail(node)

                failures[node] = f"{type(error).__name__}: {error}"
                continue

            if set(result) != set(outputs):
                raise ValueError(
                    f"expected outputs {outputs!r}, " f"got {tuple(result)!r}"
                )

            for name, value in result.items():
                store[name][node] = value

        return failures
