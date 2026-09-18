from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    TypeVar,
)
from collections.abc import Iterator, Mapping, MutableMapping, ValuesView
from collections import defaultdict
from enum import Enum
from warnings import warn

T = TypeVar("T")

NODE_NODE = "node-node"
_RESERVED = {"annotate", "inputs", "optional_inputs", "outputs", "algorithm_type"}


class NotAvailable(Exception):
    """Raised by InputSpec.gather when the value doesn't exist for this unit."""


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
    * A parent is in a higher layer than its children.

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

        inverted = [(c, p) for c, p in parent_dict.items() if c.layer >= p.layer]
        if inverted:
            child, parent = inverted[0]
            raise InvalidAnnotationTree(
                f"{child} must be in a lower layer than its parent {parent}"
            )
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

    def __repr__(self) -> str:
        return f"Annotation(name={self.name!r}, tree={self.tree!r})"

    @property
    def states(self) -> Mapping[NodeId, AnnotationState]:
        return self._states

    def fail(self, node: NodeId) -> None:
        node = self.tree.check(node)
        self._values.pop(node, None)
        self._states[node] = AnnotationState.FAILED

    def computed(self, nodes: Iterable[NodeId] | None = None) -> dict[NodeId, T]:
        """Return {node: value} for computed nodes.

        With nodes=None, it covers the whole tree. With nodes, it covers only the given nodes
        skipping any that aren't computed. Raises KeyError for nodes not in the tree.
        """
        if nodes is None:
            return dict(self._values)
        checked = (self.tree.check(node) for node in nodes)
        return {node: self._values[node] for node in checked if node in self._values}

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
    """Structural contract for an annotation algorithm.

    Required:
    - inputs: input annotation names that must be present and computed per node
    - outputs: output annotation names produced by annotate(...)
    - algorithm_type: type for executor. currently supports "node-node"
    - annotate(node, **kwargs): returns a mapping with exactly keys in outputs

    Optional:
    - optional_inputs: input annotation names that are passed only when available
      and computed for the current node.
    """

    inputs: Sequence[str]
    outputs: Sequence[str]
    algorithm_type: Any

    def annotate(
        self,
        unit: Any,
        **kwargs: Any,
    ) -> Mapping[str, Any]: ...


class Order(Enum):
    BOTTOM_UP = "bottom-up"
    TOP_DOWN = "top-down"


class InputSpec:
    source = None
    from_output = None
    order = None

    def prepare(self, annotator, annotation, nodes):
        """Execute once per run, before any annotate(). Good for once-per-run precomputations
        by the annotator needs. For example, computing a needed aggregate statistic or
        figuring out what the units need to be, as required for disambiguation."""
        return None

    def units(self, prepared):
        """Return the units, that is, groups of nodes each .annotate() will be called on,
        if this spec defines the units, else None. The executor takes the unit to be a node if None.
        """
        return None

    def gather(self, annotation, unit, prepared):
        """Return the kwarg values per unit."""
        raise NotImplementedError


@dataclass(frozen=True)
class NodeInput(InputSpec):
    source: Optional[str] = None
    from_output: Optional[str] = None

    def gather(self, annotation, unit, prepared):
        state = annotation.states[unit]
        if getattr(state, "value", state) != AnnotationState.COMPUTED.value:
            raise NotAvailable(f"{annotation.name!r} is not computed for {unit}")
        return annotation[unit]


@dataclass(frozen=True)
class DescendantsInput(InputSpec):
    order = Order.BOTTOM_UP
    depth: Optional[int] = None
    source: Optional[str] = None
    from_output: Optional[str] = None

    def gather(self, annotation, unit, prepared):
        below = annotation.tree.descendants(unit, depth=self.depth)
        return annotation.computed(nodes=below)


@dataclass(frozen=True)
class MultiNodeInput(InputSpec):
    groups: Optional[str] = None
    source: Optional[str] = None
    from_output: Optional[str] = None

    def prepare(self, annotator, annotation, nodes):
        if self.groups is None:
            return None
        return call_method(annotator, self.groups, computed_values(annotation, nodes))

    def units(self, prepared):
        return prepared

    def gather(self, annotation, unit, prepared):
        return computed_values(annotation, unit)


@dataclass(frozen=True)
class NodeOutput:
    def write(self, annotation, unit, value):
        annotation[unit] = value

    def fail(self, annotation, unit):
        annotation.fail(unit)


@dataclass(frozen=True)
class MultiNodeOutput:
    def write(self, annotation, unit, values):
        for node in values:
            if node not in unit:
                raise ValueError(
                    f"{annotation.name!r} got {node}, which is outside {unit}"
                )
        for node, value in values.items():
            annotation[node] = value

    def fail(self, annotation, unit):
        for node in unit:
            annotation.fail(node)


INPUT_SPECS = {
    "node": NodeInput,
    "descendants": DescendantsInput,
    "multinode": MultiNodeInput,
}
OUTPUT_SPECS = {
    "node": NodeOutput,
    "multinode": MultiNodeOutput,
}


def parse_algorithm_type(annotator):
    """Given an annotator, format the algorithm_type in the canonical form:
    return ({input_name: InputSpec}, output_spec)."""
    names = tuple(annotator.inputs) + tuple(getattr(annotator, "optional_inputs", ()))
    algorithm_type = annotator.algorithm_type

    if isinstance(algorithm_type, str):
        lhs, rhs = algorithm_type.split("-")
        if not names:
            return {}, OUTPUT_SPECS[rhs]()
        algorithm_type = {name: (lhs, rhs) for name in names}

    if set(algorithm_type) != set(names):
        raise ValueError(
            f"algorithm_type keys {sorted(algorithm_type)} must match "
            f"inputs + optional_inputs {sorted(names)}"
        )

    specs, output_specs = {}, set()
    for name, (lhs, rhs) in algorithm_type.items():
        spec = INPUT_SPECS[lhs]() if isinstance(lhs, str) else lhs
        if spec.source is not None and spec.from_output is not None:
            raise ValueError(f"input {name!r} can't set both source and from_output")
        specs[name] = spec
        output_specs.add(OUTPUT_SPECS[rhs]() if isinstance(rhs, str) else rhs)

    if len(output_specs) != 1:
        raise ValueError(
            f"all inputs must have the same kind of output, got {output_specs}"
        )
    return specs, output_specs.pop()


class Executor:
    """Executes annotators against its annotation store."""

    def __init__(self, store: AnnotationStore):
        self.store = store

    @property
    def tree(self) -> AnnotationTree:
        return self.store.tree

    def run(
        self,
        annotator: Annotator,
        *,
        nodes: Iterable[NodeId] | None = None,
        input_names: Mapping[str, str] | None = None,
        output_names: Mapping[str, str] | None = None,
    ) -> dict[NodeId, str]:
        """Run an annotator, reading and writing the store names given by
        input_names={kwarg: store_name} and output_names={output: store_name}.
        Unlisted names map to themselves.

        Return ``{unit: reason}`` for nodes whose annotation failed."""

        if not isinstance(annotator, Annotator):
            raise TypeError(
                "an Annotator needs inputs, outputs, algorithm_type, " "and annotate"
            )

        required_inputs = set(annotator.inputs)
        optional_inputs = set(getattr(annotator, "optional_inputs", ()))
        if len(set(required_inputs)) != len(required_inputs):
            raise ValueError("annotator.inputs do not match")

        if len(set(optional_inputs)) != len(optional_inputs):
            raise ValueError("annotator.optional_inputs contains duplicate names")

        specs, output = parse_algorithm_type(annotator)
        required_inputs = set(annotator.inputs)
        outputs = tuple(annotator.outputs)
        input_names = dict(input_names or {})
        output_names = dict(output_names or {})

        # Wiring of the annotator's inputs and outputs to the store names
        if set(output_names) - set(outputs):
            raise ValueError(
                f"output_names has unknown outputs: {sorted(set(output_names) - set(outputs))}"
            )
        writes_to = {name: output_names.get(name, name) for name in outputs}
        if len(set(writes_to.values())) != len(writes_to):
            raise ValueError(f"two outputs write to the same annotation: {writes_to}")

        if set(input_names) - set(specs):
            raise ValueError(
                f"input_names has unknown inputs: {sorted(set(input_names) - set(specs))}"
            )
        reads_from = {}
        for name, spec in specs.items():
            if spec.from_output is None:
                reads_from[name] = input_names.get(name, spec.source or name)
            elif name in input_names:
                raise ValueError(
                    f"input {name!r} reads from an output; rewire the output instead"
                )
            elif spec.from_output not in writes_to:
                raise ValueError(
                    f"input {name!r} reads from unknown output {spec.from_output!r}"
                )
            else:
                reads_from[name] = writes_to[spec.from_output]

        available = set(self.store) | set(writes_to.values())
        missing_required = [
            name
            for name in required_inputs
            if name in specs and reads_from[name] not in available
        ]
        if missing_required:
            missing_store_names = sorted(
                {reads_from[name] for name in missing_required}
            )
            raise KeyError(f"missing input annotation(s): {missing_store_names}")

        missing_optional = [
            name
            for name in optional_inputs
            if name in specs and reads_from[name] not in available
        ]
        if missing_optional:
            explicit_missing = [
                name for name in missing_optional if name in input_names
            ]
            if explicit_missing:
                missing_store_names = sorted(
                    {reads_from[name] for name in explicit_missing}
                )
                warn(
                    f"missing explicitly-wired optional input annotation(s) will be ignored: "
                    f"{missing_store_names}"
                )
            specs = {
                name: spec
                for name, spec in specs.items()
                if name not in missing_optional
            }

        for store_name in writes_to.values():
            if store_name not in self.store:
                self.store.add(Annotation(store_name, self.tree))

        selected_nodes = (
            self.store.tree.nodes
            if nodes is None
            else tuple(self.store.tree.check(node) for node in nodes)
        )

        # Run the prepare step on each input spec if it defines one
        # Note that this runs on the subset of nodes to be run on the annotator, which
        # might not be the behaviour expected for some of the preparation steps.
        # will adapt this later as needed.
        prepared = {
            name: spec.prepare(annotator, self.store[reads_from[name]], selected_nodes)
            for name, spec in specs.items()
        }
        # Determine the units to apply .annotate to
        unit_sources = [
            groups
            for name, spec in specs.items()
            if (groups := spec.units(prepared[name])) is not None
        ]
        if len(unit_sources) > 1:
            raise ValueError("only one input may define the units")
        if unit_sources:
            unit_groups = unit_sources[0]
            allowed, seen, result = set(selected_nodes), set(), []
            for group in unit_groups:
                group = tuple(sorted(self.tree.check(n) for n in group))
                if not allowed.issuperset(group):
                    raise ValueError(f"group {group} has nodes outside the selection")
                if seen.intersection(group):
                    raise ValueError(f"group {group} overlaps another group")
                seen.update(group)
                result.append(group)
            units = sorted(result)
        elif isinstance(output, NodeOutput):
            units = selected_nodes
        else:
            raise ValueError("a multinode annotator needs an input that defines groups")
        units = self._order_units(units, specs, reads_from, writes_to)

        # A write to an annotation this run also reads keeps its old value on failure.
        # Replace with more explicit instructions for handling in-place updates later
        in_place = set(writes_to.values()) & {reads_from[n] for n in specs}

        # Finally run .annotate over the units
        failures = {}
        for unit in units:
            prior_output_state = {
                store_name: self.store[store_name].states[unit]
                for store_name in writes_to.values()
            }

            try:
                kwargs = {}
                for name, spec in specs.items():
                    try:
                        kwargs[name] = spec.gather(
                            self.store[reads_from[name]], unit, prepared[name]
                        )
                    except NotAvailable as e:
                        if name in required_inputs:
                            raise e
                result = annotator.annotate(unit, **kwargs)
            except Exception as error:
                for store_name in writes_to.values():
                    was_computed = (
                        prior_output_state[store_name] is AnnotationState.COMPUTED
                    )
                    keep_old_value = store_name in in_place and was_computed
                    if not keep_old_value:
                        output.fail(self.store[store_name], unit)
                failures[unit] = f"{type(error).__name__}: {error}"
                continue

            if set(result) != set(outputs):
                raise ValueError(f"expected outputs {outputs!r}, got {tuple(result)!r}")
            for name, value in result.items():
                output.write(self.store[writes_to[name]], unit, value)

        return failures

    def _order_units(self, units, specs, reads_from, writes_to):
        """
        Reorders the units based on the specified layer order in the specs, or
        return them as-is if no order is specified.

        Only matters when the output of one unit affects the input of another unit.
        """
        written = set(writes_to.values())
        orders = {
            spec.order
            for name, spec in specs.items()
            if spec.order is not None and reads_from[name] in written
        }
        if len(orders) > 1:
            raise ValueError(f"Input specs require conflicting orders: {orders}")
        if not orders:
            return units
        order = orders.pop()

        def layer(unit):
            layers = {n.layer for n in ((unit,) if isinstance(unit, NodeId) else unit)}
            if len(layers) > 1:
                raise ValueError(
                    f"{order.value} order can't place {unit}, which spans layers {sorted(layers)}"
                )
            return layers.pop()

        return sorted(units, key=layer, reverse=order is Order.TOP_DOWN)
