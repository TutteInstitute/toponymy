from collections import Counter
import pathlib
from toponymy.treemap import _layer_value, _rooted_tree


def construct_topic_hierarchy(clusterer, topic_names, root_name="Root"):
    """
    Constructs a hierarchical representation of topics based on clustering results.

    Args:
        clusterer: A ToponymyClusterer with a cluster_tree_ attribute representing the hierarchy.
        topic_names: A nested list or structure of topic names corresponding to cluster labels.
        root_name: Name to use for the root node in the hierarchy (default 'Root').

    Returns:
        A nested dictionary representing the hierarchy, suitable for visualization or further processing.
    """

    result = getattr(clusterer, "topic_model_", clusterer)
    if hasattr(result, "topics"):
        cluster_tree = result.cluster_tree
        counters = [Counter() for _ in result.cluster_layers]
        for (layer, label), topic in result.topics.items():
            counters[layer][label] = len(topic.members)
    else:
        cluster_tree = clusterer.cluster_tree_
        counters = [
            Counter(layer.cluster_labels) for layer in clusterer.cluster_layers_
        ]
    cluster_tree, root = _rooted_tree(cluster_tree, topic_names)

    n_objects = (
        len(result.embedding_vectors)
        if hasattr(result, "topics")
        else sum(counters[0].values()) if counters else 0
    )
    hierarchy = recurse_hierarchy(
        root, cluster_tree, counters, topic_names, root_name, n_objects=n_objects
    )

    return hierarchy


def recurse_hierarchy(
    node, cluster_tree, counters, topic_names, root_name, n_objects=None
):
    """
    Recursively traverses the cluster hierarchy tree to build nested topic structure.

    Args:
        node: Current node in the hierarchy (tuple or root index).
        cluster_tree: Dictionary mapping parent nodes to their children nodes.
        counters: List of Counter objects tracking label counts at each layer.
        topic_names: Nested list of topic names corresponding to cluster labels.
        root_name: Name to assign when at the root (used in the topmost call).

    Returns:
        A dictionary representing the current node with optional children and size.
    """

    layer, label = node
    children = cluster_tree.get(node, [])
    if any(child[0] >= layer for child in children):
        raise ValueError("Topic tree edges must lead to lower layers")
    if layer >= len(counters):
        size = (
            n_objects
            if n_objects is not None
            else sum(counters[0].values()) if counters else 0
        )
        topic = root_name
    else:
        size = counters[layer][label]
        topic = _layer_value(topic_names, layer, label)
        if topic is None:
            topic = f"Layer {layer}, Cluster {label}"
    item = {"name": topic, "size": size}
    if children:
        item["children"] = [
            recurse_hierarchy(
                child,
                cluster_tree,
                counters,
                topic_names,
                root_name,
                n_objects=n_objects,
            )
            for child in children
        ]
    return item


def __getattr__(name):
    if name not in {"IndentedTree", "RadialTidyTree", "CirclePacking"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import anywidget
    import traitlets

    class IndentedTree(anywidget.AnyWidget):
        """
        A widget class for visualizing hierarchical data as an indented tree structure.
        Adapted from https://observablehq.com/@d3/indented-tree

        Static Path:
            _esm (str): Path to the widget's JavaScript module for frontend rendering.
        Attributes:
            data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
                representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
            width (traitlets.int): The width of the widget svg.
            palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
                Must be one of ['latte','frappe','macchiato','mocha']

        Usage:
            - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
            - Instantiate the widget in a Jupyter notebook.
        """

        _esm = pathlib.Path(__file__).parent / "widgets/indentedTree/dist/widget.js"
        data = traitlets.Dict(default_value={"name": "Root", "size": 0}).tag(sync=True)
        width = traitlets.Int(default_value=1024).tag(sync=True)
        palette = traitlets.Unicode(default_value="latte").tag(sync=True)

    class RadialTidyTree(anywidget.AnyWidget):
        """
        A widget class for visualizing hierarchical data as a radial, tidy tree diagram.
        Adapted from https://observablehq.com/@d3/radial-tree/2

        Static Paths:
            _esm (str): Path to the widget's JavaScript module for frontend rendering.
            _css (str): Path to the widget's CSS stylesheet for styling.
        Attributes:
            data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
                representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
            width (traitlets.int): The width of the widget svg.
            maxTextChars (traitlets.int): Number of characters from label to display (useful when label names
                are long).
            palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
                Must be one of ['latte','frappe','macchiato','mocha']
        Usage:
            - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
            - Instantiate the widget in a Jupyter notebook.
        """

        _esm = pathlib.Path(__file__).parent / "widgets/radialTidyTree/dist/widget.js"
        _css = pathlib.Path(__file__).parent / "widgets/radialTidyTree/dist/widget.css"
        data = traitlets.Dict(default_value={"name": "Root", "size": 0}).tag(sync=True)
        width = traitlets.Int(default_value=1024).tag(sync=True)
        maxTextChars = traitlets.Int(default_value=30).tag(sync=True)
        palette = traitlets.Unicode(default_value="latte").tag(sync=True)

    class CirclePacking(anywidget.AnyWidget):
        """
        A widget class for visualizing hierarchical data as a circle packing diagram.
        Adapted from https://observablehq.com/@d3/zoomable-circle-packing.


        Static Paths:
            _esm (str): Path to the widget's JavaScript module for frontend rendering.
            _css (str): Path to the widget's CSS stylesheet for styling.
        Attributes:
            data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
                representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
            width (traitlets.int): The width of the widget svg.
            maxTextChars (traitlets.int): Number of characters from label to display (useful when label names
                are long).
            palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
                Must be one of ['latte','frappe','macchiato','mocha']
        Usage:
            - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
            - Instantiate the widget in a Jupyter notebook.
        """

        _esm = pathlib.Path(__file__).parent / "widgets/circlePacking/dist/widget.js"
        _css = pathlib.Path(__file__).parent / "widgets/circlePacking/dist/widget.css"
        data = traitlets.Dict(default_value={"name": "Root", "size": 0}).tag(sync=True)
        width = traitlets.Int(default_value=1024).tag(sync=True)
        maxTextChars = traitlets.Int(default_value=50).tag(sync=True)
        palette = traitlets.Unicode(default_value="latte").tag(sync=True)

    globals().update(
        IndentedTree=IndentedTree,
        RadialTidyTree=RadialTidyTree,
        CirclePacking=CirclePacking,
    )
    return globals()[name]
