from collections import Counter 
import pathlib
import anywidget
import traitlets

def construct_topic_hierarchy(clusterer, topic_names, root_name='Root'):

    cluster_tree = clusterer.cluster_tree_
    root = max(cluster_tree.keys())
    counters = [Counter(layer.cluster_labels) for layer in clusterer.cluster_layers_]

    hierarchy = recurse_hierarchy(root, cluster_tree, counters, topic_names, root_name)

    return hierarchy

def recurse_hierarchy(node, cluster_tree, counters, topic_names, root_name):
    
    try:
        u,v = node
        size = counters[u][v]
        topic = topic_names[u][v]
    except IndexError:
        assert u == len(counters)
        assert v == 0
        size = counters[0].total()
        topic = root_name

    try:
        children = cluster_tree[node]
        item = {
                "name": topic,
                "children":[recurse_hierarchy(child,cluster_tree) for child in children],
            }
        if size is not None:
            item.update({'size':size})
        return item
        
    except KeyError:
        return {"name": topic,'size':1}

class IndentedTree(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "widgets/indentedTree/widget.js"
    _css = pathlib.Path(__file__).parent / "widgets/indentedTree/widget.css"
    data = traitlets.Dict(default_value={}).tag(sync=True) 
    palette = traitlets.Unicode(default_value="catppuccin_frappe").tag(sync=True) 


class RadialTidyTree(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "widgets/radialTidyTree/widget.js"
    _css = pathlib.Path(__file__).parent / "widgets/radialTidyTree/widget.css"
    data = traitlets.Dict(default_value={}).tag(sync=True) 
    palette = traitlets.Unicode(default_value="catppuccin_frappe").tag(sync=True) 
