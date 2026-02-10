import pandas as pd
import numpy as np 

def treemap_dataframe(
    toponymy
):
    """
    Builds a dataframe out of a Toponymy for use in a plotly.express treemap.
    """
    rows = []

    layer_info = {
        layer: {
            "labels": cl.cluster_labels,
            "names": cl.topic_names
        }
        for layer, cl in enumerate(toponymy.cluster_layers_)
    }

    def node_id(node):
        return f"L{node[0]}_C{node[1]}"

    def node_label(node):
        layer, c = node
        if layer == len(toponymy.cluster_layers_):
            # root node!
            total = np.size(toponymy.cluster_layers_[-1].cluster_labels)
            clustered = np.size(
                np.where(toponymy.cluster_layers_[-1].cluster_labels != -1)
            )
            pc = clustered/total*100
            return f"Everything | {clustered}/{total} clustered ({pc:.2f}%)"
        try:
            return layer_info[layer]["names"][c]
        except Exception:
            return f"Layer {layer}, Cluster {c}"

    def node_value(node):
        layer, c = node
        if layer == len(toponymy.cluster_layers_):
            # root node!
            labels = 0
        else:
            labels = layer_info[layer]["labels"]
        return int(np.sum(labels == c))

    nodes = set(toponymy.cluster_tree_.keys())
    for children in toponymy.cluster_tree_.values():
        nodes.update(children)

    for parent, children in toponymy.cluster_tree_.items():
        parent_id = node_id(parent)

        for child in children:
            rows.append({
                "id": node_id(child),
                "parent": parent_id,
                "label": node_label(child),
                "value": node_value(child)
            })

    child_ids = {node_id(c) for cs in toponymy.cluster_tree_.values() for c in cs}

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