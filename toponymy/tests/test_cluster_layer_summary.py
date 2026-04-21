from toponymy.cluster_layer import (
    ClusterLayerText,
    ClusterLayerSummaryText,
    ClusterLayer,
)
from toponymy.keyphrases import KeyphraseBuilder

import numpy as np

import pytest


def test_summary_layer_creation(
    cluster_label_vector, cluster_centroid_vectors, embedder
):
    """ClusterLayerSummaryText should construct and inherit from ClusterLayerText."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    assert cluster_layer
    assert isinstance(cluster_layer, ClusterLayerSummaryText)
    assert isinstance(cluster_layer, ClusterLayerText)
    assert isinstance(cluster_layer, ClusterLayer)
    # Ensure the summary-specific init properly wires up attributes
    assert cluster_layer.n_keyphrases == 16
    assert cluster_layer.n_exemplars == 8
    assert cluster_layer.n_subtopics == 16
    # The summary layer aliases the embedding model when provided
    assert cluster_layer.embedding_model is embedder


def test_summary_layer_kwargs_override(
    cluster_label_vector, cluster_centroid_vectors, embedder
):
    """Custom n_* parameters should pass through to the instance."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
        n_keyphrases=4,
        n_exemplars=3,
        n_subtopics=5,
        keyphrase_diversify_alpha=0.7,
        exemplars_diversify_alpha=0.8,
        subtopic_diversify_alpha=0.9,
    )
    assert cluster_layer.n_keyphrases == 4
    assert cluster_layer.n_exemplars == 3
    assert cluster_layer.n_subtopics == 5
    assert cluster_layer.keyphrase_diversify_alpha == 0.7
    assert cluster_layer.exemplars_diversify_alpha == 0.8
    assert cluster_layer.subtopic_diversify_alpha == 0.9


def test_summary_make_data(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Full pipeline exercise on a summary layer with default methods."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)

    # Fabricate layer 0 summaries/explanations to match the layer 0 topic names
    layer_zero_names = all_subtopics
    layer_zero_summaries = [f"Summary of {name}" for name in layer_zero_names]
    layer_zero_explanations = [f"Explanation for {name}" for name in layer_zero_names]

    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)

    # Unlike the plain text layer, the summary layer requires summaries/explanations
    cluster_layer.make_subtopics(
        layer_zero_names,
        subtopic_label_vector,
        subtopic_centroid_vectors,
        topic_summaries=layer_zero_summaries,
        topic_explanations=layer_zero_explanations,
    )

    # make_prompts also requires all_topic_summaries and all_topic_explanations
    all_topic_names = [layer_zero_names, []]
    all_topic_summaries = [layer_zero_summaries, []]
    all_topic_explanations = [layer_zero_explanations, []]

    cluster_layer.make_prompts(
        1.0,
        all_topic_names,
        "sentences",
        "about specific popular topics",
        cluster_tree,
        all_topic_summaries=all_topic_summaries,
        all_topic_explanations=all_topic_explanations,
    )

    assert len(cluster_layer.prompts) == cluster_centroid_vectors.shape[0]

    # Pretend we named clusters (summary layer tracks three parallel lists)
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.topic_summaries = [
        f"Summary of {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.topic_explanations = [
        f"Explanation for {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_summary_make_data_alternative_methods1(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Pipeline exercise with system_user prompt format and central subtopic method."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
        prompt_format="system_user",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)

    layer_zero_summaries = [f"Summary of {name}" for name in all_subtopics]
    layer_zero_explanations = [f"Explanation for {name}" for name in all_subtopics]

    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="central"
    )
    cluster_layer.make_subtopics(
        all_subtopics,
        subtopic_label_vector,
        subtopic_centroid_vectors,
        method="central",
        topic_summaries=layer_zero_summaries,
        topic_explanations=layer_zero_explanations,
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
        all_topic_summaries=[layer_zero_summaries, []],
        all_topic_explanations=[layer_zero_explanations, []],
    )

    # Pretend we named clusters
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.topic_summaries = [
        f"Summary of {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.topic_explanations = [
        f"Explanation for {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_summary_make_data_alternative_methods2(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Pipeline exercise with bm25 keyphrases and information_weighted subtopic method."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
        prompt_format="combined",
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)

    layer_zero_summaries = [f"Summary of {name}" for name in all_subtopics]
    layer_zero_explanations = [f"Explanation for {name}" for name in all_subtopics]

    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(
        keyphrases, matrix, keyphrase_vectors, embedder, method="bm25"
    )
    cluster_layer.make_subtopics(
        all_subtopics,
        subtopic_label_vector,
        subtopic_centroid_vectors,
        method="information_weighted",
        topic_summaries=layer_zero_summaries,
        topic_explanations=layer_zero_explanations,
    )
    cluster_layer.make_prompts(
        1.0,
        [all_subtopics, []],
        "sentences",
        "about specific popular topics",
        cluster_tree,
        all_topic_summaries=[layer_zero_summaries, []],
        all_topic_explanations=[layer_zero_explanations, []],
    )

    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    cluster_layer.topic_summaries = [
        f"Summary of {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.topic_explanations = [
        f"Explanation for {x['topic']}" for x in subtopic_objects
    ]
    cluster_layer.embed_topic_names(embedder)
    cluster_layer._make_disambiguation_prompts(
        1.0,
        [all_subtopics, [x["topic"] for x in subtopic_objects]],
        "sentences",
        "about specific popular topics",
        cluster_tree,
    )


def test_summary_make_data_submodular_subtopics(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    subtopic_objects,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Exercises the submodular path (facility_location / saturated_coverage)
    which routes to submodular_summary_subtopics."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)

    layer_zero_summaries = [f"Summary of {name}" for name in all_subtopics]
    layer_zero_explanations = [f"Explanation for {name}" for name in all_subtopics]

    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)

    for method in ("facility_location", "saturated_coverage"):
        cluster_layer.make_subtopics(
            all_subtopics,
            subtopic_label_vector,
            subtopic_centroid_vectors,
            method=method,
            topic_summaries=layer_zero_summaries,
            topic_explanations=layer_zero_explanations,
        )
        assert cluster_layer.subtopics is not None
        assert len(cluster_layer.subtopics) == cluster_centroid_vectors.shape[0]


def test_summary_make_prompts_requires_summaries_and_explanations(
    all_sentences,
    object_vectors,
    embedder,
    all_subtopics,
    cluster_tree,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """make_prompts must assert that all_topic_summaries and all_topic_explanations are provided."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    keyphrase_builder = KeyphraseBuilder()
    matrix, keyphrases, vectors = keyphrase_builder.fit_transform(all_sentences)
    keyphrase_vectors = embedder.encode(keyphrases)
    cluster_layer.make_exemplar_texts(all_sentences, object_vectors)
    cluster_layer.make_keyphrases(keyphrases, matrix, keyphrase_vectors, embedder)
    # subtopics empty is fine; asserts below fire before subtopics are touched
    cluster_layer.subtopics = [[] for _ in range(cluster_centroid_vectors.shape[0])]

    # Missing both
    with pytest.raises(AssertionError):
        cluster_layer.make_prompts(
            1.0,
            [all_subtopics, []],
            "sentences",
            "about specific popular topics",
            cluster_tree,
        )
    # Missing explanations only
    with pytest.raises(AssertionError):
        cluster_layer.make_prompts(
            1.0,
            [all_subtopics, []],
            "sentences",
            "about specific popular topics",
            cluster_tree,
            all_topic_summaries=[[f"s {n}" for n in all_subtopics], []],
        )
    # Missing summaries only
    with pytest.raises(AssertionError):
        cluster_layer.make_prompts(
            1.0,
            [all_subtopics, []],
            "sentences",
            "about specific popular topics",
            cluster_tree,
            all_topic_explanations=[[f"e {n}" for n in all_subtopics], []],
        )


def test_summary_make_subtopics_requires_summaries_and_explanations(
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    embedder,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """make_subtopics must assert that topic_summaries and topic_explanations are provided."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    with pytest.raises(AssertionError):
        cluster_layer.make_subtopics(
            all_subtopics,
            subtopic_label_vector,
            subtopic_centroid_vectors,
        )
    with pytest.raises(AssertionError):
        cluster_layer.make_subtopics(
            all_subtopics,
            subtopic_label_vector,
            subtopic_centroid_vectors,
            topic_summaries=[f"s {n}" for n in all_subtopics],
        )
    with pytest.raises(AssertionError):
        cluster_layer.make_subtopics(
            all_subtopics,
            subtopic_label_vector,
            subtopic_centroid_vectors,
            topic_explanations=[f"e {n}" for n in all_subtopics],
        )


def test_summary_make_subtopics_invalid_method(
    all_subtopics,
    subtopic_label_vector,
    subtopic_centroid_vectors,
    embedder,
    cluster_label_vector,
    cluster_centroid_vectors,
):
    """Unknown subtopic method should raise ValueError."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    layer_zero_summaries = [f"s {n}" for n in all_subtopics]
    layer_zero_explanations = [f"e {n}" for n in all_subtopics]
    with pytest.raises(ValueError):
        cluster_layer.make_subtopics(
            all_subtopics,
            subtopic_label_vector,
            subtopic_centroid_vectors,
            method="nonexistent_method",
            topic_summaries=layer_zero_summaries,
            topic_explanations=layer_zero_explanations,
        )


def test_summary_make_topic_name_vector(
    cluster_label_vector,
    cluster_centroid_vectors,
    subtopic_objects,
    embedder,
):
    """make_topic_name_vector should broadcast topic names across cluster members
    and leave unlabelled points as 'Unlabelled'."""
    cluster_layer = ClusterLayerSummaryText(
        cluster_label_vector,
        cluster_centroid_vectors,
        1,
        embedder,
    )
    cluster_layer.topic_names = [x["topic"] for x in subtopic_objects]
    result = cluster_layer.make_topic_name_vector()
    assert result.shape == cluster_label_vector.shape
    # Every labelled cluster index should map to its topic name
    for cluster_idx, topic_name in enumerate(cluster_layer.topic_names):
        mask = cluster_label_vector == cluster_idx
        if mask.any():
            assert np.all(result[mask] == topic_name)
    # Unclustered points (-1 in HDBSCAN convention) should remain 'Unlabelled'
    unlabelled_mask = cluster_label_vector < 0
    if unlabelled_mask.any():
        assert np.all(result[unlabelled_mask] == "Unlabelled")