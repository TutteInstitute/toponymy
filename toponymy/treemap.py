"""Convert layered topic trees to compact plotting tables."""

from collections.abc import Mapping

import pandas as pd


def _layer_value(layers, layer, label, default=None):
    if not 0 <= layer < len(layers) or label < 0:
        return default
    values = layers[layer]
    if isinstance(values, Mapping):
        return values.get(label, default)
    return values[label] if label < len(values) else default


def _rooted_tree(tree, topics):
    """Copy a tree and supply one display root for an unrooted forest."""
    result = {parent: list(children) for parent, children in tree.items()}
    for parent, children in result.items():
        if any(child[0] >= parent[0] for child in children):
            raise ValueError("Topic tree edges must lead to lower layers")
    synthetic = [node for node in result if node[0] >= len(topics)]
    if synthetic:
        return result, max(synthetic)
    nodes = set(result)
    children = {child for group in result.values() for child in group}
    nodes.update(children)
    for layer, names in enumerate(topics):
        labels = names if isinstance(names, Mapping) else range(len(names))
        nodes.update((layer, label) for label in labels)
    root = (len(topics), 0)
    result[root] = sorted(nodes - children, reverse=True)
    return result, root


def treemap_dataframe(topic_tree):
    """Build an ID-preserving table from the reachable topic hierarchy.

    Values are actual member counts; sparse cluster labels never determine
    allocation sizes. Unreachable entries retained after duplicate pruning do
    not become additional roots. The synthetic root counts the whole corpus.
    ``layout_value`` makes drawing weights additive when topic memberships
    overlap across layers; ``value`` always retains actual member counts.
    """
    rows = []
    tree, root = _rooted_tree(topic_tree.tree, topic_tree.topics)
    n_layers = len(topic_tree.topics)
    visited = set()

    def node_id(node):
        return f"L{node[0]}_C{node[1]}"

    def visit(node, parent):
        if node in visited:
            raise ValueError("A treemap requires each topic to have one parent")
        visited.add(node)
        layer, label = node
        children = tree.get(node, [])
        if layer >= n_layers:
            size = topic_tree.n_objects
            name = f"Everything | {size} objects"
        else:
            size = _layer_value(topic_tree.topic_sizes, layer, label, 0)
            name = _layer_value(topic_tree.topics, layer, label)
            if name is None:
                name = f"Layer {layer}, Cluster {label}"
        row = {
            "id": node_id(node),
            "parent": node_id(parent) if parent else "",
            "label": name,
            "value": size,
        }
        rows.append(row)
        row["layout_value"] = max(size, sum(visit(child, node) for child in children))
        return row["layout_value"]

    visit(root, None)
    return pd.DataFrame(
        rows, columns=["id", "parent", "label", "value", "layout_value"]
    )
