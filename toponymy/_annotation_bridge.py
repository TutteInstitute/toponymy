"""Temporary bridge that copies annotations from the store back onto the legacy layers.

Remove once the prompt templates read from the annotation store.
"""

from toponymy.cluster_layer import ClusterLayerText


def has_stock_exemplars(layer) -> bool:
    """False for layer classes that override make_exemplar_texts."""
    return type(layer).make_exemplar_texts is ClusterLayerText.make_exemplar_texts


def copy_exemplars_to_layers(store, layers) -> None:
    """Copies the store's exemplars onto the layers whose exemplars were annotated."""
    for layer_id, layer in enumerate(layers):
        if has_stock_exemplars(layer):
            nodes = store.tree.layer(layer_id)
            layer.exemplars = [store.exemplars[node] for node in nodes]
            layer.exemplar_indices = [
                store.exemplar_original_indices[node] for node in nodes
            ]
