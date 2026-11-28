"""Temporary bridge that copies annotations between the store and the legacy layers.

Remove once the prompt templates read from the annotation store.
"""

from toponymy.annotation import Annotation, AnnotationState
from toponymy.cluster_layer import ClusterLayerText


def copy_layers_to_store(store, layers, name, attribute) -> None:
    """Copies each layer's ``attribute`` into ``store[name]`` for the nodes no annotator computed."""
    if name not in store:
        store.add(Annotation(name, store.tree))
    annotation = store[name]
    for layer_id, layer in enumerate(layers):
        nodes = store.tree.layer(layer_id)
        for node, value in zip(nodes, getattr(layer, attribute), strict=True):
            if annotation.states[node] is not AnnotationState.COMPUTED:
                annotation[node] = value


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


def has_stock_keyphrases(layer) -> bool:
    """False for layer classes that override make_keyphrases."""
    return type(layer).make_keyphrases is ClusterLayerText.make_keyphrases


def copy_keyphrases_to_layers(store, layers) -> None:
    """Copies the store's keyphrases onto the layers whose keyphrases were annotated."""
    for layer_id, layer in enumerate(layers):
        if has_stock_keyphrases(layer):
            layer.keyphrases = [
                store.keyphrases[node] for node in store.tree.layer(layer_id)
            ]


def copy_subtopics_to_layers(store, layers) -> None:
    """Copies the store's subtopics onto the layers above layer 0."""
    for layer_id, layer in enumerate(layers[1:], 1):
        layer.subtopics = [store.subtopics[node] for node in store.tree.layer(layer_id)]


def copy_topics_to_layers(store, layers) -> None:
    """Copies the store's topics, and any summaries, onto the layers whose topics were annotated."""
    copied = [
        (store_name, layer_name)
        for store_name, layer_name in [
            ("topic_names", "topic_names"),
            ("topic_summaries", "topic_summaries"),
            ("topic_explanations", "topic_explanations"),
        ]
        if store_name in store
    ]
    for layer_id, layer in enumerate(layers):
        for store_name, layer_name in copied:
            annotation = store[store_name]
            setattr(
                layer,
                layer_name,
                [
                    (
                        annotation[node]
                        if annotation.states[node] == AnnotationState.COMPUTED
                        else ""
                    )
                    for node in store.tree.layer(layer_id)
                ],
            )
