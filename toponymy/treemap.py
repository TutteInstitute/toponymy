import pandas as pd
import numpy as np 

def treemap_dataframe(
    topic_tree
):
    """
    Builds a dataframe out of a TopicTree for use in a plotly.express treemap.
    """
    rows = []
    n_layers = len(topic_tree.topics)

    def node_id(node):
        return f"L{node[0]}_C{node[1]}"

    def node_label(node):
        layer, c = node
        if layer == n_layers:
            # root node!
            clustered = sum(topic_tree.topic_sizes[-1])
            pc = clustered/topic_tree.n_objects*100
            return f"Everything | {clustered}/{topic_tree.n_objects} clustered ({pc:.2f}%)"
        try:
            return topic_tree.topics[layer][c]
        except Exception:
            return f"Layer {layer}, Cluster {c}"

    def node_value(node):
        layer, c = node
        if layer == n_layers:
            # root node!
            size = sum(topic_tree.topic_sizes[-1])
        else:
            size = topic_tree.topic_sizes[layer][c]
        return size

    nodes = set(topic_tree.tree.keys())
    for children in topic_tree.tree.values():
        nodes.update(children)

    for parent, children in topic_tree.tree.items():
        parent_id = node_id(parent)

        for child in children:
            rows.append({
                "id": node_id(child),
                "parent": parent_id,
                "label": node_label(child),
                "value": node_value(child)
            })

    child_ids = {node_id(c) for cs in topic_tree.tree.values() for c in cs}

    for node in nodes:
        nid = node_id(node)
        if nid not in child_ids:
            rows.append({
                "id": nid,
                "parent": "",
                "label": node_label(node),
                "value": node_value(node)
            })

    return pd.DataFrame(rows)