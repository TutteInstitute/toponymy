import warnings
from collections import defaultdict
from dataclasses import dataclass

import numba
import numpy as np
import sklearn.feature_extraction
import sklearn.metrics
import vectorizers
import vectorizers.transformers
from fast_hdbscan import boruvka, cluster_trees, hdbscan, numba_kdtree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

import jinja2

_PROMPT_TEMPLATES = {
    "remedy": jinja2.Template(
        """
You are an expert in {{larger_topic}} and have been asked to provide a more specific name for a group of 
{{document_type}} from {{corpus_description}}. The group of {{document_type}} has been described as having a topic of one of 
{{attempted_topic_names}}. These topic names were not specific enough.

The other groups of {{document_type}} that can be confused with this topic are:

{% for topic in matching_topics  %}
{{topic}}:
 - Keywords: {{", ".join(matching_topic_keywords[topic])}}
{% if matching_topic_subtopics[topic] %}
 - Subtopics: {{", ".join(matching_topic_subtopics[topic])}}
{% endif %}
 - Sample {{document_type}}:
{% for sentence in matching_topic_sentences[topic] %}
      - "{{sentence}}"
{% endfor %}
{% endfor %}

As an expert in {{larger_topic}}, you need to provide a more specific name for this group of {{document_type}}:
 - Keywords: {{", ".join(cluster_keywords)}}
{% if cluster_subtopics %}
 - Subtopics: {{", ".join(cluster_subtopics)}}
{% endif %}
 - Sample {{document_type}}:
{% for sentence in cluster_sentences %}
      - "{{sentence}}"
{% endfor %}

You should make use of the relative relationships between these topics as well as the keywords
and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new 
better and more *specific* topic name.

The response should be only JSON with no preamble formatted as 
  {"topic_name":<NAME>, "less_specific_topic_name":<NAME>, "topic_specificity":<SCORE>} 
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble.
"""
    ),
    "distinguish_base_layer_topics": jinja2.Template(
        """
You are an expert in {{larger_topic}}, and have been asked to provide a more specific names for various groups of
{{document_type}} from {{corpus_description}} that have been assigned overly similar auto-generated topic names.

Below are the auto-generated topic names, along with some keywords associated to each topic, and a sampling of {{self.document_type}} from the topic area.

{% for topic, keywords, sentences in base_layer_topic_data %}
"{{loop.index}}. {{topic}}":
 - Keywords: {{", ".join(keywords)}}
 - Sample {{document_type}}:
{% for sentence in sentences %}
      - "{{sentence}}"
{% endfor %}
{% endfor %}

Your should make use of the relative relationships between these topics as well as the keywords
and {{self.document_type}} information and your expertise in {{larger_topic}} to generate new
better and more distinguishing topic names. There should be no duplicate topic names in the final list.

The response should be formatted as JSON in the format 
    {"topic_names": [<NAME1>, <NAME2>, ...], "previous_topic_names": [<OLD_NAME1>, <OLD_NAME2>, ...], topic_specificity": [<SCORE1>, <SCORE2>, ...]}
where SCORE is a value in the range 0 to 1.
The response must contain only JSON with no preamble and must have one entry for each topic to be renamed.
"""
    ),
}

COSINE_DISTANCE_EPSILON = 1e-6


@numba.njit(fastmath=True)
def layer_from_clustering(
    point_vectors,
    point_locations,
    cluster_label_vector,
    cluster_membership_vector,
    base_clusters,
    membership_strength_threshold=0.2,
):
    n_clusters = len(set(cluster_label_vector)) - 1

    average_vectors = np.zeros((n_clusters, point_vectors.shape[1]), dtype=np.float32)
    average_locations = np.zeros(
        (n_clusters, point_locations.shape[1]), dtype=np.float32
    )
    total_weights = np.zeros(n_clusters, dtype=np.float32)
    pointsets = [set([-1 for i in range(0)]) for i in range(n_clusters)]
    metaclusters = [set([-1 for i in range(0)]) for i in range(n_clusters)]

    for i in range(cluster_label_vector.shape[0]):
        cluster_num = cluster_label_vector[i]
        if cluster_num >= 0:
            average_vectors[cluster_num] += (
                cluster_membership_vector[i] * point_vectors[i]
            )
            average_locations[cluster_num] += (
                cluster_membership_vector[i] * point_locations[i]
            )
            total_weights[cluster_num] += cluster_membership_vector[i]

            if cluster_membership_vector[i] > membership_strength_threshold:
                pointsets[cluster_num].add(i)
                sub_cluster = base_clusters[i]
                if sub_cluster != -1:
                    metaclusters[cluster_num].add(sub_cluster)

    for c in range(n_clusters):
        average_vectors[c] /= total_weights[c]
        average_locations[c] /= total_weights[c]

    return average_vectors, average_locations, pointsets, metaclusters


def build_cluster_layers(
    point_vectors,
    point_locations,
    *,
    min_clusters=2,
    min_samples=5,
    base_min_cluster_size=10,
    membership_strength_threshold=0.2,
    next_cluster_size_quantile=0.8,
    verbose=False,
):
    vector_layers = []
    location_layers = []
    pointset_layers = []
    metacluster_layers = []

    min_cluster_size = base_min_cluster_size

    sklearn_tree = hdbscan.KDTree(point_locations)
    numba_tree = numba_kdtree.kdtree_to_numba(sklearn_tree)
    edges = boruvka.parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = cluster_trees.mst_to_linkage_tree(sorted_mst)
    new_tree = cluster_trees.condense_tree(uncondensed_tree, base_min_cluster_size)
    leaves = cluster_trees.extract_leaves(new_tree)
    clusters = cluster_trees.get_cluster_label_vector(
        new_tree, leaves, 0.0, len(point_locations)
    )
    point_probs = cluster_trees.get_point_membership_strength_vector(
        new_tree, leaves, clusters
    )

    cluster_ids = np.unique(clusters[clusters >= 0])
    base_clusters = clusters.copy()
    n_clusters_in_layer = cluster_ids.shape[0]

    base_layer = True

    while n_clusters_in_layer >= min_clusters:

        layer_vectors, layer_locations, layer_pointsets, layer_metaclusters = (
            layer_from_clustering(
                point_vectors,
                point_locations,
                clusters,
                point_probs,
                base_clusters,
                membership_strength_threshold,
            )
        )

        if not base_layer:
            layer_metacluster_selection = np.asarray(
                [len(x) > 1 for x in layer_metaclusters]
            )
            layer_metaclusters = [
                list(x)
                for x, select in zip(layer_metaclusters, layer_metacluster_selection)
                if select
            ]
            layer_pointsets = [
                list(x)
                for x, select in zip(layer_pointsets, layer_metacluster_selection)
                if select
            ]
            layer_vectors = layer_vectors[layer_metacluster_selection]
            layer_locations = layer_locations[layer_metacluster_selection]

        vector_layers.append(layer_vectors)
        location_layers.append(layer_locations)
        pointset_layers.append(layer_pointsets)
        metacluster_layers.append(layer_metaclusters)

        last_min_cluster_size = min_cluster_size
        min_cluster_size = int(
            np.quantile([len(x) for x in layer_pointsets], next_cluster_size_quantile)
        )
        if verbose:
            print(
                f"cluster={len(layer_vectors)}, last_min_cluster_size={last_min_cluster_size}, min_cluster_size={min_cluster_size}"
            )

        new_tree = cluster_trees.condense_tree(uncondensed_tree, min_cluster_size)
        leaves = cluster_trees.extract_leaves(new_tree)
        clusters = cluster_trees.get_cluster_label_vector(
            new_tree, leaves, 0.0, len(point_locations)
        )
        point_probs = cluster_trees.get_point_membership_strength_vector(
            new_tree, leaves, clusters
        )

        cluster_ids = np.unique(clusters[clusters >= 0])
        n_clusters_in_layer = np.max(clusters) + 1
        base_layer = False

    pointset_layers = [
        [list(pointset) for pointset in layer] for layer in pointset_layers
    ]
    return vector_layers, location_layers, pointset_layers, metacluster_layers


@numba.njit()
def _build_cluster_tree(labels):
    mapping = [(-1, -1, -1, -1) for i in range(0)]
    found = [set([-1]) for i in range(len(labels))]
    mapping_idx = 0
    for upper_layer in range(1, len(labels)):
        upper_layer_unique_labels = np.unique(labels[upper_layer])
        for lower_layer in range(upper_layer - 1, -1, -1):
            upper_cluster_order = np.argsort(labels[upper_layer])
            cluster_groups = np.split(
                labels[lower_layer][upper_cluster_order],
                np.cumsum(np.bincount(labels[upper_layer] + 1))[:-1],
            )
            for i, label in enumerate(upper_layer_unique_labels):
                if label >= 0:
                    for child in cluster_groups[i]:
                        if child >= 0 and child not in found[lower_layer]:
                            mapping.append((upper_layer, label, lower_layer, child))
                            found[lower_layer].add(child)

    for lower_layer in range(len(labels) - 1, -1, -1):
        for child in range(labels[lower_layer].max() + 1):
            if child >= 0 and child not in found[lower_layer]:
                mapping.append((len(labels), 0, lower_layer, child))

    return mapping


def build_cluster_tree(labels):
    result = {}
    raw_mapping = _build_cluster_tree(labels)
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result


def pointsets_to_label_layers(pointsets_layers, n_points):
    result = []
    for pointsets in pointsets_layers:
        label_layer = np.full(n_points, -1)
        for i, pointset in enumerate(pointsets):
            label_layer[pointset] = i
        result.append(label_layer)
    return result


def diversify(query_vector, candidate_neighbor_vectors, alpha=1.0, max_candidates=16):
    distance_to_query = np.squeeze(
        sklearn.metrics.pairwise_distances(
            [query_vector], candidate_neighbor_vectors, metric="cosine"
        )
    )

    retained_neighbor_indices = [0]
    for i, vector in enumerate(candidate_neighbor_vectors[1:], 1):
        retained_neighbor_distances = sklearn.metrics.pairwise_distances(
            [vector],
            candidate_neighbor_vectors[retained_neighbor_indices],
            metric="cosine",
        )[0]
        for j in range(retained_neighbor_distances.shape[0]):
            if alpha * distance_to_query[i] > retained_neighbor_distances[j]:
                break
        else:
            retained_neighbor_indices.append(i)
            if len(retained_neighbor_indices) >= max_candidates:
                return retained_neighbor_indices

    return retained_neighbor_indices


def topical_sentences_for_cluster(
    docs, vector_array, pointset, centroid_vector, n_sentence_examples=16
):
    sentences = docs.values[pointset]

    sent_vectors = vector_array[pointset]
    candidate_neighbor_indices = np.argsort(
        np.squeeze(
            sklearn.metrics.pairwise_distances(
                [centroid_vector], sent_vectors, metric="cosine"
            )
        )
    )
    candidate_neighbors = sent_vectors[candidate_neighbor_indices]
    topical_sentence_indices = candidate_neighbor_indices[
        diversify(centroid_vector, candidate_neighbors)[:n_sentence_examples]
    ]
    topical_sentences = [sentences[i] for i in topical_sentence_indices]
    return topical_sentences


def distinctive_sentences_for_cluster(
    cluster_num,
    docs,
    vector_array,
    pointset_layer,
    cluster_neighbors,
    n_sentence_examples=16,
):
    pointset = pointset_layer[cluster_num]
    sentences = docs.values[pointset]

    local_vectors = vector_array[
        sum([pointset_layer[x] for x in cluster_neighbors], [])
    ]
    vectors_for_svd = normalize(local_vectors - local_vectors.mean(axis=0))
    U, S, Vh = randomized_svd(
        vectors_for_svd, min(int(np.sqrt(vectors_for_svd.shape[0])), 64)
    )
    transformed_docs = local_vectors @ Vh.T
    transformed_docs = np.maximum(transformed_docs, 0)
    class_labels = np.repeat(
        np.arange(len(cluster_neighbors)),
        [len(pointset_layer[x]) for x in cluster_neighbors],
    )
    iwt = vectorizers.transformers.InformationWeightTransformer().fit(
        transformed_docs, class_labels
    )
    sentence_weights = np.sum(
        transformed_docs[: len(pointset)] * iwt.information_weights_, axis=1
    )
    distinctive_sentence_indices = np.argsort(sentence_weights)[
        : n_sentence_examples * 3
    ]
    distinctive_sentence_vectors = vector_array[distinctive_sentence_indices]
    diversified_candidates = diversify(
        vector_array[pointset_layer[cluster_num]].mean(axis=0),
        distinctive_sentence_vectors,
    )
    distinctive_sentence_indices = distinctive_sentence_indices[
        diversified_candidates[:n_sentence_examples]
    ]
    distinctive_sentences = [sentences[i] for i in distinctive_sentence_indices]
    return distinctive_sentences


def longest_keyphrases(candidate_keyphrases):
    result = []
    for i, phrase in enumerate(candidate_keyphrases):
        for other in candidate_keyphrases:
            if f" {phrase}" in other or f"{phrase} " in other:
                phrase = other

        if phrase not in result:
            candidate_keyphrases[i] = phrase
            result.append(phrase)

    return result


def create_distinguish_base_layer_topics_prompt(
    topic_indices,
    attempted_topic_names,
    representations,
    document_type,
    corpus_description,
    max_keywords=16,
    max_sentences=16,
):
    template = _PROMPT_TEMPLATES["distinguish_base_layer_topics"]

    unique_topic_names = list(set(attempted_topic_names))
    if len(unique_topic_names) == 1:
        larger_topic = unique_topic_names[0]
    else:
        larger_topic = (
            ", ".join(unique_topic_names[:-1]) + " and " + unique_topic_names[-1]
        )

    keywords_per_topic = [
        representations["contrastive"][i][:max_keywords] for i in topic_indices
    ]
    sentences_per_topic = [
        representations["topical"][i][:max_sentences] for i in topic_indices
    ]

    base_layer_topic_data = list(zip(attempted_topic_names, keywords_per_topic, sentences_per_topic))

    prompt_text = template.render(
        larger_topic=larger_topic,
        base_layer_topic_data=base_layer_topic_data,
        document_type=document_type,
        corpus_description=corpus_description,
    )
    return prompt_text


def create_topic_discernment_prompt(
    layer,
    topic_index,
    attempted_topic_names,
    matching_topics,
    representations,
    subtopic_layers,
    document_type,
    corpus_description,
    max_keywords=16,
    max_subtopics=16,
    max_sentences=16,
):
    template = _PROMPT_TEMPLATES["remedy"]
    larger_topic = attempted_topic_names[0]
    cluster_keywords = representations["contrastive"][layer][topic_index][:max_keywords]
    cluster_subtopics = (
        subtopic_layers[layer - 1][topic_index][:max_subtopics] if layer > 0 else None
    )
    cluster_sentences = representations["topical"][layer][topic_index][:max_sentences]
    matching_topic_keywords = {
        (matching_topic_layer, matching_topic_index): representations["contrastive"][
            matching_topic_layer
        ][matching_topic_index][:max_keywords]
        for matching_topic_layer, matching_topic_index in matching_topics
    }
    matching_topic_subtopics = {
        (matching_topic_layer, matching_topic_index): subtopic_layers[
            matching_topic_layer - 1
        ][matching_topic_index][:max_subtopics]
        for matching_topic_layer, matching_topic_index in matching_topics
        if matching_topic_layer > 0
    }
    matching_topic_sentences = {
        (matching_topic_layer, matching_topic_index): representations["topical"][
            matching_topic_layer
        ][matching_topic_index][:max_sentences]
        for matching_topic_layer, matching_topic_index in matching_topics
    }
    prompt_text = template.render(
        larger_topic=larger_topic,
        attempted_topic_names=attempted_topic_names,
        matching_topics=matching_topics,
        matching_topic_keywords=matching_topic_keywords,
        matching_topic_subtopics=matching_topic_subtopics,
        matching_topic_sentences=matching_topic_sentences,
        cluster_keywords=cluster_keywords,
        cluster_subtopics=cluster_subtopics,
        cluster_sentences=cluster_sentences,
        document_type=document_type,
        corpus_description=corpus_description,
    )
    return prompt_text


def find_threshold_for_max_cluster_size(distances, max_cluster_size=16):
    n_samples = distances.shape[0]
    clustering = AgglomerativeClustering(
        n_clusters=2,
        distance_threshold=None,
        compute_full_tree=True,
        compute_distances=True,
        metric="precomputed",
        linkage="complete",
    )
    clustering.fit(distances)
    cluster_sizes = defaultdict(lambda: 1)
    merge_distances = clustering.distances_

    for i, (cluster1, cluster2) in enumerate(clustering.children_):
        new_size = cluster_sizes[cluster1] + cluster_sizes[cluster2]
        if (
            new_size > max_cluster_size
            and merge_distances[i - 1] > COSINE_DISTANCE_EPSILON
        ):
            return merge_distances[i - 1] if i > 0 else merge_distances[0]

        cluster_sizes[n_samples + i] = new_size
        del cluster_sizes[cluster1]
        del cluster_sizes[cluster2]

    return merge_distances[-1]


@dataclass
class ClusterLayers:
    """Class for keeping track of cluster layer information"""

    vector_layers: list[list[list]]
    location_layers: list[list[list]]
    pointset_layers: list[list[list]]
    metacluster_layers: list[list[list]]
    layer_cluster_neighbours: list[list[list]]


class Toponymy:
    """
    documents: list of strings
        A list of objects to topic model.  Our current LLM topic naming functions currently presume these to be strings.
    document_vectors: numpy array
        A numpy array of shape number_of_objects by features.  These are vectors which encode the semantic similarity of our
        documents being topic modeled.
    document_map: numpy array
        A numpy array of shape number_of_objects by 2 (or 3).  These are two dimensional vectors often corresponding
        to a 2 dimensional umap of the document_vectors.
    cluster_layers: list of lists (optional, default None):
        A list with one element for each layer in your hierarchical clustering.
        Each layer is a list
    representative_sentences: dict (optional, default None):
        A dictionary from one of a set of ways to represent a document cluster to a the cluster representation.
    trim_percentile: int (between 0 and 100)
        Trim any document with a token length longer than the 99th percentile. This prevents very long outlier documents from swamping our prompts.
        The trim length will be the maximum of this value and trim_length.  Set to 100 if you don't want any trimming.
    trim_length: int
        Maximum number of tokens to keep from each document. This prevents very long outlier documents from swamping our prompts.
        The trim length will be the maximum of this value and trim_length. Set to None if you don't want any trimming.
    """

    def __init__(
        self,
        documents,
        document_vectors,
        document_map,
        llm,
        embedding_model=None,  # The embedding model that the document_vectors was constructed with.
        cluster_layers=None,  # ClusterLayer dataclass
        representation_techniques=["topical", "contrastive"],
        document_type="titles",
        corpus_description="academic articles",
        verbose=True,
        trim_percentile=99,
        trim_length=100,
        keyphrase_min_occurrences=25,
        keyphrase_ngram_range=(1, 4),
        n_sentence_examples_per_cluster=16,
        n_keyphrases_per_cluster=16,
        max_subtopics_per_cluster=32,
        max_neighbors_per_cluster=6,
    ):
        self.documents = documents
        self.document_vectors = document_vectors
        self.document_map = document_map
        if (cluster_layers is not None) and not isinstance(
            cluster_layers, ClusterLayers
        ):
            raise ValueError(
                f"cluster_layers must be of type ClusterLayers not {type(cluster_layers)}"
            )
        if cluster_layers:
            self.cluster_layers_ = cluster_layers
        self.representation_techniques = representation_techniques
        self.embedding_model = embedding_model
        # Check that this is either None or has an embed function.
        self.document_type = document_type
        self.corpus_description = corpus_description
        self.llm = llm
        self.verbose = verbose
        self.trim_percentile = trim_percentile
        self.trim_length = trim_length
        self.keyphrase_min_occurrences = keyphrase_min_occurrences
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.n_sentence_examples_per_cluster = n_sentence_examples_per_cluster
        self.n_keyphrases_per_cluster = n_keyphrases_per_cluster
        self.max_subtopics_per_cluster = max_subtopics_per_cluster
        self.max_neighbors_per_cluster = max_neighbors_per_cluster

    def fit_clusters(self, base_min_cluster_size=100, min_clusters=6):
        """
        Constructs a layered hierarchical clustering well suited for layered topic modeling.
        TODO: Add a check to ensure that there were any cluster generated at the specified base_min_cluster_size.
        """
        if self.verbose:
            print("Constructing cluster layers")
        self.base_min_cluster_size_ = base_min_cluster_size
        self.min_clusters_ = min_clusters

        vector_layers, location_layers, pointset_layers, metacluster_layers = (
            build_cluster_layers(
                self.document_vectors,
                self.document_map,
                base_min_cluster_size=base_min_cluster_size,
                min_clusters=min_clusters,
                verbose=self.verbose,
            )
        )

        layer_cluster_neighbours = [
            np.argsort(
                sklearn.metrics.pairwise_distances(layer, metric="cosine"), axis=1
            )[:, : self.max_neighbors_per_cluster]
            for layer in vector_layers
        ]
        self.cluster_layers_ = ClusterLayers(
            vector_layers,
            location_layers,
            pointset_layers,
            metacluster_layers,
            layer_cluster_neighbours,
        )

        clustering_label_layers = pointsets_to_label_layers(
            pointset_layers, self.document_vectors.shape[0]
        )
        self.cluster_tree_ = build_cluster_tree(clustering_label_layers)

    def get_topical_layers(self, n_sentence_examples=16):
        """
        Fits a set of topical documents to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.
        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()

        topical_sentences_per_cluster = [
            [
                topical_sentences_for_cluster(
                    self.documents,
                    self.document_vectors,
                    pointset,
                    cluster_vector,
                    n_sentence_examples=n_sentence_examples,
                )
                for pointset, cluster_vector in tqdm(
                    zip(
                        self.cluster_layers_.pointset_layers[i],
                        self.cluster_layers_.vector_layers[i],
                    ),
                    desc=f"Topical sentences for layer {i}",
                    total=len(self.cluster_layers_.pointset_layers[i]),
                    disable=(not self.verbose),
                )
            ]
            for i in range(len(self.cluster_layers_.pointset_layers))
        ]
        return topical_sentences_per_cluster

    def get_distinctive_layers(self, n_sentence_examples=16):
        """
        Fits a set of distincts documents to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()

        distinctive_sentences_per_cluster = [
            [
                distinctive_sentences_for_cluster(
                    topic_num,
                    self.documents,
                    self.document_vectors,
                    self.cluster_layers_.pointset_layers[i],
                    self.cluster_layers_.layer_cluster_neighbours[i][topic_num],
                    n_sentence_examples=n_sentence_examples,
                )
                for topic_num in tqdm(
                    range(len(self.cluster_layers_.pointset_layers[i])),
                    desc=f"Distinctive sentences for layer {i}",
                    disable=(not self.verbose),
                )
            ]
            for i in range(len(self.cluster_layers_.pointset_layers))
        ]
        return distinctive_sentences_per_cluster

    def _contrastive_keywords_for_layer(
        self,
        layer_num,
        full_count_matrix,
        inverse_vocab,
        vocab_vectors,
        n_keywords=16,
        prior_strength=0.1,
        weight_power=2.0,
    ):
        pointset_layer = self.cluster_layers_.pointset_layers[layer_num]
        count_matrix = full_count_matrix[sum(pointset_layer, []), :]
        column_mask = np.squeeze(np.asarray(count_matrix.sum(axis=0))) > 0.0
        count_matrix = count_matrix[:, column_mask]
        column_map = np.arange(full_count_matrix.shape[1])[column_mask]
        row_mask = np.squeeze(np.asarray(count_matrix.sum(axis=1))) > 0.0
        count_matrix = count_matrix[row_mask, :]
        bad_rows = set(np.where(~row_mask)[0])

        class_labels = np.repeat(
            np.arange(len(pointset_layer)), [len(x) for x in pointset_layer]
        )[row_mask]
        iwt = vectorizers.transformers.InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        ).fit(count_matrix, class_labels)
        count_matrix.data = np.log(count_matrix.data + 1)
        count_matrix.eliminate_zeros()

        weighted_matrix = iwt.transform(count_matrix)

        contrastive_keyword_layer = []

        from_row = 0

        for i in range(len(pointset_layer)):
            if i in bad_rows:
                contrastive_keyword_layer.append(["no keywords were found"])
            else:
                to_row = from_row + len(pointset_layer[i])
                contrastive_scores = np.squeeze(
                    np.asarray(weighted_matrix[from_row:to_row].sum(axis=0))
                )
                contrastive_keyword_indices = np.argsort(contrastive_scores)[
                    -4 * n_keywords :
                ]
                contrastive_keywords = [
                    inverse_vocab[column_map[j]]
                    for j in reversed(contrastive_keyword_indices)
                ]
                contrastive_keywords = longest_keyphrases(contrastive_keywords)

                centroid_vector = np.mean(
                    self.document_vectors[pointset_layer[i]], axis=0
                )
                keyword_vectors = np.asarray(
                    [vocab_vectors[word] for word in contrastive_keywords]
                )
                chosen_indices = diversify(
                    centroid_vector, keyword_vectors, alpha=0.66
                )[:n_keywords]
                contrastive_keywords = [contrastive_keywords[j] for j in chosen_indices]

                contrastive_keyword_layer.append(contrastive_keywords)
                from_row = to_row

        return contrastive_keyword_layer

    def get_contrastive_keyword_layers(self, n_keyphrases_per_cluster=16):
        """
        Fits a set of contrastive keywords to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # TODO: count_vectorizer: CountVectorizer might be passed in at some point but for now is hard coded.
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()
        # Check if embedding_model is set and has an encode function

        cv = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=True,
            min_df=self.keyphrase_min_occurrences,
            token_pattern="(?u)\\b\\w[-'\\w]+\\b",
            ngram_range=self.keyphrase_ngram_range,
        )
        full_count_matrix = cv.fit_transform(self.documents)
        acceptable_vocab = [
            v
            for v in cv.vocabulary_
            if v.split()[0] not in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
            and v.split()[-1] not in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
        ]
        acceptable_indices = [cv.vocabulary_[v] for v in acceptable_vocab]
        full_count_matrix = full_count_matrix[:, acceptable_indices]
        inverse_vocab = {i: w for i, w in enumerate(acceptable_vocab)}
        vocab = acceptable_vocab

        if self.verbose:
            print(
                f"Created a potential keyphrase vocabulary of {len(vocab)} potential keyphrases"
            )

        vocab_vectors = dict(
            zip(
                vocab,
                self.embedding_model.encode(vocab, show_progress_bar=self.verbose),
            )
        )

        contrastive_keyword_layers = [
            self._contrastive_keywords_for_layer(
                layer_num,
                full_count_matrix,
                inverse_vocab,
                vocab_vectors,
                n_keywords=n_keyphrases_per_cluster,
            )
            for layer_num in tqdm(
                range(len(self.cluster_layers_.pointset_layers)),
                desc="Finding contrastive keywords",
                disable=(not self.verbose),
            )
        ]
        return contrastive_keyword_layers

    # Might use a dict for handling options
    def fit_representation(self):
        """
        Samples topical_layers, distincive_layers and contrastive_keyword layers depending on which methods have been included in the representation_techniques.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()
        if self.verbose:
            print("Sampling documents per cluster")
        self.representation_ = dict()
        for rep in self.representation_techniques:
            if rep == "topical":
                self.representation_[rep] = self.get_topical_layers(
                    n_sentence_examples=self.n_sentence_examples_per_cluster
                )
            elif rep == "distinctive":
                self.representation_[rep] = self.get_distinctive_layers(
                    n_sentence_examples=self.n_sentence_examples_per_cluster
                )
            elif rep == "contrastive":
                self.representation_[rep] = self.get_contrastive_keyword_layers(
                    n_keyphrases_per_cluster=self.n_keyphrases_per_cluster
                )
            else:
                warnings.warn(f"{rep} is not a supported representation")
        return None

    def build_base_prompt(
        self,
        cluster_id,
        layer_id=0,
        max_docs_per_cluster=100,
        llm_instruction="The short distinguising topic name is:",
    ):
        """
        Take a cluster_id and layer_id and extracts the relevant information from the representation_ and cluster_layers_ properties to
        construct a representative prompt to present to a large langauge model.

        Each represenative is trimmed to be at most self.token_trim_length tokens in size.
        """
        prompt_text = f"Below is a information about a group of {self.document_type} from {self.corpus_description}:\n\n"

        # TODO: Add some random sentences

        # Add some contrastive keywords (might drop this in favor of the last one. Let the experiments commence!)
        if "contrastive" in self.representation_techniques:
            prompt_text += (
                'keywords for this group:\n - "'
                + ", ".join(self.representation_["contrastive"][layer_id][cluster_id])
                + '"\n'
            )
        # Add some topical documents
        if "topical" in self.representation_techniques:
            prompt_text += (
                f"\nSample topical {self.document_type} from the group include:\n"
            )
            for text in self.representation_["topical"][layer_id][cluster_id][
                :max_docs_per_cluster
            ]:
                prompt_text += f' - "{text}"\n'
        if "distinctive" in self.representation_techniques:
            prompt_text += (
                f"\nSample distinctive {self.document_type} from the group include:\n"
            )
            for text in self.representation_["distinctive"][layer_id][cluster_id][
                :max_docs_per_cluster
            ]:
                prompt_text += f' - "{text}"\n'
        prompt_text += "\n\n" + llm_instruction
        return prompt_text

    def fit_base_level_prompts(
        self,
        layer_id=0,
        max_docs_per_cluster=100,
    ):
        """
        This returns a list of prompts for the layer_id independent of any other layer.
        This is commonly used for the base layer of a hierarchical topic clustering (hence the layer_id=0)

        If any of the prompt lengths (in llm tokenze) are longere than the max tokens for our llm (as defined by llm.n_ctx)
        then we reduce the maximum documents sampled from each cluster by a half and try again.  If we ever have to sample
        a single document per cluster we will declaire failure and raise and error.

        If the representation_ have not yet been generated or is None it will generate them as necessary.

        FUTURE: We hope to include improved subsampling and document partitioning method in future releases to allow
            for more representative sampling and prompt engineering.
        """
        max_docs_per_cluster = min(
            max_docs_per_cluster, self.n_sentence_examples_per_cluster
        )
        if self.verbose:
            print(
                f"generating base layer topic names with at most {max_docs_per_cluster} {self.document_type} per cluster."
            )
        if getattr(self, "representation_", None) is None:
            self.fit_representation()
        layer_size = len(self.cluster_layers_.location_layers[layer_id])
        prompts = []
        for cluster_id in tqdm(
            range(layer_size),
            desc="Generating base layer prompts",
            disable=(not self.verbose),
        ):
            prompt = self.build_base_prompt(
                cluster_id,
                layer_id,
                max_docs_per_cluster,
                llm_instruction=self.llm.llm_instruction(kind="base_layer"),
            )
            prompts.append(prompt)
        self.base_layer_prompts_ = prompts
        return None

    def _get_topic_name(self, prompt_layer, layer_num):
        """
        Takes a prompt layer and applies an llm to convert these prompts into topics.
        """
        topic_names = []
        for i in tqdm(
            range(len(prompt_layer)),
            desc=f"Generating topics for layer {layer_num}",
            disable=(not self.verbose),
        ):
            if prompt_layer[i].startswith("SKIP"):
                topic_names.append(prompt_layer[i].split(":")[1].strip())
                continue
            topic_name = self.llm.generate_topic_name(prompt_layer[i])
            topic_names.append(topic_name)
        return topic_names

    def distinguish_base_layer_topics(self):
        if getattr(self, "base_layer_topics_", None) is None:
            self.fit_base_layer_topics()

        import pandas as pd
        new_topic_names = self.base_layer_topics_[:]
        base_layer_topic_embedding = self.embedding_model.encode(
            self.base_layer_topics_, show_progress_bar=True
        )
        base_layer_topic_distances = pairwise_distances(
            base_layer_topic_embedding, metric="cosine"
        )
        distance_threshold = find_threshold_for_max_cluster_size(
            base_layer_topic_distances
        )
        cls = AgglomerativeClustering(
            n_clusters=None,
            compute_full_tree=True,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="complete",
        )
        print("Distance threshold", distance_threshold)
        cls.fit(base_layer_topic_distances)
        cluster_sizes = np.bincount(cls.labels_)
        print("Cluster sizes", np.sort(cluster_sizes)[::-1])
        print(pd.Series(new_topic_names).value_counts())
        clusters_for_renaming = np.where(cluster_sizes >= 2)[0]
        for c in tqdm(
            clusters_for_renaming,
            desc="Distinguishing similar topics",
            disable=(not self.verbose),
        ):
            label_indices = np.where(cls.labels_ == c)[0]
            prompt = create_distinguish_base_layer_topics_prompt(
                label_indices,
                [self.base_layer_topics_[x] for x in label_indices],
                {
                    "topical": self.representation_["topical"][0],
                    "contrastive": self.representation_["contrastive"][0],
                },
                self.document_type,
                self.corpus_description,
            )
            cluster_topic_names = self.llm.generate_topic_cluster_names(
                prompt,
                [self.base_layer_topics_[x] for x in label_indices],
                temperature=1.0,
            )
            for new_topic_name, topic_index in zip(cluster_topic_names, label_indices):
                new_topic_names[topic_index] = new_topic_name

        print(pd.Series(new_topic_names).value_counts())
        self.base_layer_topics_ = new_topic_names

    def fit_base_layer_topics(self):
        """
        Uses the llm to fit a topic name for each base level cluster based on the base_layer_prompts_
        If the base_layer_prompts_ have not yet been generated or is None it will generate them as necessary.
        """
        if getattr(self, "base_layer_prompts_", None) is None:
            self.fit_base_level_prompts()
        self.base_layer_topics_ = self._get_topic_name(self.base_layer_prompts_, 0)
        self.distinguish_base_layer_topics()
        return None

    def _topical_subtopics_for_cluster(
        self,
        layer_num,
        cluster_num,
        n_subtopics=32,
    ):
        metacluster = self.cluster_layers_.metacluster_layers[layer_num][cluster_num]
        pointset = self.cluster_layers_.pointset_layers[layer_num][cluster_num]

        centroid_vector = np.mean(self.document_vectors[pointset], axis=0)
        subtopic_vectors = np.asarray(
            [
                np.mean(
                    self.document_vectors[self.cluster_layers_.pointset_layers[0][n]],
                    axis=0,
                )
                for n in metacluster
            ]
        )
        candidate_neighbor_indices = np.argsort(
            np.squeeze(
                sklearn.metrics.pairwise_distances(
                    [centroid_vector], subtopic_vectors, metric="cosine"
                )
            )
        )[: 2 * n_subtopics]
        candidate_neighbors = subtopic_vectors[candidate_neighbor_indices]
        topical_subtopic_indices = candidate_neighbor_indices[
            diversify(
                centroid_vector,
                candidate_neighbors,
                alpha=0.66,
                max_candidates=n_subtopics,
            )
        ][:n_subtopics]
        topical_subtopics = [
            self.base_layer_topics_[metacluster[i]] for i in topical_subtopic_indices
        ]
        return topical_subtopics

    def _distinctive_subtopics_for_cluster(
        self,
        layer_num,
        cluster_num,
        base_layer_topic_embeddings,
        n_subtopics=24,
    ):

        layer_neighbors = self.cluster_layers_.layer_cluster_neighbours[layer_num]
        cluster_neighbors = layer_neighbors[cluster_num]
        meta_clusters = self.cluster_layers_.metacluster_layers[layer_num]
        cluster_subclusters = meta_clusters[cluster_neighbors[0]]
        topic_names = [self.base_layer_topics_[x] for x in cluster_subclusters]
        local_vectors = base_layer_topic_embeddings[
            sum([meta_clusters[x] for x in cluster_neighbors], [])
        ]
        U, S, Vh = np.linalg.svd(local_vectors - local_vectors.mean(axis=0))
        transformed_docs = local_vectors @ Vh.T
        transformed_docs = np.where(transformed_docs > 0, transformed_docs, 0)
        class_labels = np.repeat(
            np.arange(len(cluster_neighbors)),
            [len(meta_clusters[x]) for x in cluster_neighbors],
        )
        iwt = vectorizers.transformers.InformationWeightTransformer().fit(
            transformed_docs, class_labels
        )
        topic_name_weights = np.sum(
            transformed_docs[: len(topic_names)] * iwt.information_weights_, axis=1
        )
        distinctive_topic_indices = np.argsort(topic_name_weights)[: n_subtopics * 3]
        distinctive_topic_vectors = base_layer_topic_embeddings[
            distinctive_topic_indices
        ]
        diversified_candidates = diversify(
            base_layer_topic_embeddings[cluster_subclusters].mean(axis=0),
            distinctive_topic_vectors,
        )
        distinctive_topic_indices = distinctive_topic_indices[
            diversified_candidates[:n_subtopics]
        ]
        distinctive_sentences = [topic_names[i] for i in distinctive_topic_indices]
        return distinctive_sentences

    def fit_subtopic_layers(self, max_subtopics_per_cluster=32):
        """
        Fits the topical and contrastive subtopics for each intermadiate topic.
        If the base_layer_topics_ have not yet been generated or is None it will generate them as necessary.
        """
        if getattr(self, "base_layer_topics_", None) is None:
            self.fit_base_layer_topics()
        base_layer_topic_embedding = self.embedding_model.encode(
            self.base_layer_topics_, show_progress_bar=True
        )
        self.subtopic_layers_ = dict()
        self.subtopic_layers_["topical"] = [
            [
                self._topical_subtopics_for_cluster(
                    layer_num,
                    cluster_num,
                    n_subtopics=max_subtopics_per_cluster,
                )
                for cluster_num in range(
                    len(self.cluster_layers_.metacluster_layers[layer_num])
                )
            ]
            for layer_num in tqdm(
                range(1, len(self.cluster_layers_.metacluster_layers)),
                desc="Finding topical subtopics",
                disable=(not self.verbose),
            )
        ]
        return None

    def _create_prompt_from_subtopics(
        self,
        previous_layer_topics,  # Need to find the previous layer topic that contained each topic.
        layer_id,
        max_subtopics=12,
        max_docs_per_cluster=12,
        max_adjacent_clusters=3,
        max_adjacent_docs=2,
        llm_instruction="The short distinguising topic name is:",
    ):
        if getattr(self, "subtopic_layers_", None) is None:
            self.fit_subtopic_layers(self.max_subtopics_per_cluster)
        layer_size = len(self.cluster_layers_.location_layers[layer_id])
        prompts = []
        for cluster_id in tqdm(
            range(layer_size),
            desc=f"Generating prompts for layer {layer_id}",
            disable=(not self.verbose),
        ):
            prompt_text = f"Below is a information about a group of {self.document_type} from {self.corpus_description} that are all on the same topic:\n\n"
            # Add some contrastive keywords
            if "contrastive" in self.representation_techniques:
                prompt_text += (
                    'Keywords for this group:\n - "'
                    + ", ".join(
                        self.representation_["contrastive"][layer_id][cluster_id]
                    )
                    + '"\n'
                )
            # Get tree based subtopics
            tree_subtopics = self.cluster_tree_[(layer_id, cluster_id)]

            if len(tree_subtopics) == 1:
                prompts.append(
                    f"SKIP: {self.topic_name_layers_[tree_subtopics[0][0]][tree_subtopics[0][1]]}"
                )
                continue

            # Subtopics one layer down are major subtopics; two layers down are minor
            major_subtopics = [x[1] for x in tree_subtopics if x[0] == layer_id - 1]
            minor_subtopics = [x[1] for x in tree_subtopics if x[0] == layer_id - 2]
            other_subtopics = [x for x in tree_subtopics if x[0] < layer_id - 2]

            if len(major_subtopics) > 0:
                prompt_text += "\nMajor sub-topics for this group are:\n"
                for subtopic_id in major_subtopics:
                    prompt_text += (
                        f'- "{self.topic_name_layers_[layer_id - 1][subtopic_id]}"\n'
                    )

            if len(minor_subtopics) > 0:
                prompt_text += "\nMinor sub-topics for this group are:\n"
                for subtopic_id in minor_subtopics:
                    prompt_text += (
                        f'- "{self.topic_name_layers_[layer_id - 2][subtopic_id]}"\n'
                    )

            if len(other_subtopics) > 0:
                prompt_text += "\nOther sub-topics for this group not included in major or minor sub-topics are:\n"
                for layer_num, subtopic_id in other_subtopics[:max_subtopics]:
                    prompt_text += (
                        f'- "{self.topic_name_layers_[layer_num][subtopic_id]}"\n'
                    )

            if len(tree_subtopics) < max_subtopics:
                # Use the previous layer information to inject knowledge into this cluster.
                prompt_text += (
                    "\nA sampling of detailed sub-topics from the group include:\n"
                )
                for text in previous_layer_topics[cluster_id][
                    : (max_subtopics - len(tree_subtopics))
                ]:
                    prompt_text += f'- "{text}"\n'

            # Add some topical documents if we don't have many subtopics
            if len(tree_subtopics) < max_subtopics:
                if "topical" in self.representation_techniques:
                    prompt_text += f"\nSample topical {self.document_type} from the group include:\n"
                    for text in self.representation_["topical"][layer_id][cluster_id][
                        : max_docs_per_cluster - len(tree_subtopics)
                    ]:
                        prompt_text += f' - "{text}"\n'

            prompt_text += "\n" + llm_instruction
            prompts.append(prompt_text)

        return prompts

    def distinguish_intermediate_layer_topics(
        self, layer_id, previous_layer_topics, max_subtopics=12
    ):
        new_topic_names = self.topic_name_layers_[layer_id][:]
        layer_topic_embedding = self.embedding_model.encode(
            self.topic_name_layers_[layer_id], show_progress_bar=True
        )
        layer_topic_distances = pairwise_distances(
            layer_topic_embedding, metric="cosine"
        )
        distance_threshold = find_threshold_for_max_cluster_size(
            layer_topic_distances, max_cluster_size=8
        )
        cls = AgglomerativeClustering(
            n_clusters=None,
            compute_full_tree=True,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="complete",
        )
        cls.fit(layer_topic_distances)
        cluster_sizes = np.bincount(cls.labels_)
        clusters_for_renaming = np.where(cluster_sizes >= 2)[0]
        for c in tqdm(
            clusters_for_renaming,
            desc=f"Distinguishing similar topics in layer {layer_id}",
            disable=(not self.verbose),
        ):
            label_indices = np.where(cls.labels_ == c)[0]
            prompt = f"There are collections of {self.corpus_description} with somewhat similar auto-generated topic names, all in your field of expertise.\n"
            prompt += "Below are the auto-generated topic names, along with some keywords associated to each topic, and sub-topics from the topic area."
            for x in label_indices:
                prompt += f"\n\n**{self.topic_name_layers_[layer_id][x]}**\n"
                prompt += (
                    "    - keywords: "
                    + ", ".join(self.representation_["contrastive"][layer_id][x])
                    + "\n"
                )
                # Get tree based subtopics
                tree_subtopics = self.cluster_tree_[(layer_id, x)]

                # Subtopics one layer down are major subtopics; two layers down are minor
                major_subtopics = [a[1] for a in tree_subtopics if a[0] == layer_id - 1]
                minor_subtopics = [a[1] for a in tree_subtopics if a[0] == layer_id - 2]
                other_subtopics = [a for a in tree_subtopics if a[0] < layer_id - 2]

                if len(major_subtopics) > 0:
                    prompt += "\nMajor sub-topics for this group are:\n"
                    for subtopic_id in major_subtopics:
                        prompt += f'- "{self.topic_name_layers_[layer_id - 1][subtopic_id]}"\n'

                if len(minor_subtopics) > 0:
                    prompt += "\nMinor sub-topics for this group are:\n"
                    for subtopic_id in minor_subtopics:
                        prompt += f'- "{self.topic_name_layers_[layer_id - 2][subtopic_id]}"\n'

                if len(other_subtopics) > 0:
                    prompt += "\nOther sub-topics for this group not included in major or minor sub-topics are:\n"
                    for layer_num, subtopic_id in other_subtopics[:max_subtopics]:
                        prompt += (
                            f'- "{self.topic_name_layers_[layer_num][subtopic_id]}"\n'
                        )

                if len(tree_subtopics) < max_subtopics:
                    # Use the previous layer information to inject knowledge into this cluster.
                    prompt += (
                        "\nA sampling of detailed sub-topics from the group include:\n"
                    )
                    for text in previous_layer_topics[x][
                        : (max_subtopics - len(tree_subtopics))
                    ]:
                        prompt += f'- "{text}"\n'

                # Add some topical documents if we don't have many subtopics
                if len(tree_subtopics) < max_subtopics:
                    prompt += f"    - sample {self.document_type}:\n"
                    for text in self.representation_["topical"][0][x][
                        : max_subtopics - len(tree_subtopics)
                    ]:
                        prompt += f'        + "{text}"\n'

            prompt += "\n\nYou should make use of the relative relationships between these topics as well as the keywords and sub-topic information to generate new topic names."
            prompt += "\nStrive to provide the simplest possible topic name (ideally a few words) that distinguishes a given topic from the other topics listed."
            prompt += "\nPlease provide new names for the topics that differentiate among them. The result should be formatted as JSON in the format [{<OLD_TOPIC_NAME1>: <NEW_TOPIC_NAME>}, {<OLD_TOPIC_NAME2>: <NEW_TOPIC_NAME>}, ...].\n"
            prompt += "The result must contain only JSON with no preamble and must have one entry for each topic to be renamed\n"

            cluster_topic_names = self.llm.generate_topic_cluster_names(
                prompt,
                [self.topic_name_layers_[layer_id][x] for x in label_indices],
                temperature=0.8,
            )
            for new_topic_name, topic_index in zip(cluster_topic_names, label_indices):
                new_topic_names[topic_index] = new_topic_name

        self.topic_name_layers_[layer_id] = new_topic_names

    def fit_layers(self):
        """
        Constructs prompts and topic names for intermediate subtopic layers.
        If the subtopic_layers_ have not yet been generated or is None it will generate them as necessary.
        """

        if getattr(self, "subtopic_layers_", None) is None:
            self.fit_subtopic_layers(self.max_subtopics_per_cluster)
        if self.verbose:
            print("Fitting intermediate layers")
        self.topic_prompt_layers_ = [self.base_layer_prompts_]
        self.topic_name_layers_ = [self.base_layer_topics_]

        for layer_id in range(1, len(self.cluster_layers_.metacluster_layers)):
            subtopics_layer = self.subtopic_layers_["topical"][layer_id - 1]
            topic_naming_prompts = self._create_prompt_from_subtopics(
                subtopics_layer,
                layer_id,
                llm_instruction=self.llm.llm_instruction(kind="intermediate_layer"),
            )
            self.topic_prompt_layers_.append(topic_naming_prompts)
            topic_names = self._get_topic_name(topic_naming_prompts, layer_id)
            self.topic_name_layers_.append(topic_names)
            self.distinguish_intermediate_layer_topics(
                layer_id, subtopics_layer, max_subtopics=12
            )

        return None

    def _get_singleton_subclusters(self):
        result_list = []
        result_dict = {}
        for parent in self.cluster_tree_:
            if len(self.cluster_tree_[parent]) == 1:
                singleton_child = self.cluster_tree_[parent][0]
                result_list.append((parent, singleton_child))
                result_dict[singleton_child] = parent
        return result_list, result_dict

    def clean_topic_names(self):
        """
        Cleans up the prompts from the top down in order to remove topic name duplication.
        If previous properties have not yet been generated will generate them as necessary.
        This can be the only function called.
        """
        if getattr(self, "topic_name_layers_", None) is None:
            self.fit_layers()
        if self.verbose:
            print("Cleaning up topic names\n")
        self.layer_clusters = [
            np.full(self.document_map.shape[0], "Unlabelled", dtype=object)
            for i in range(len(self.topic_name_layers_))
        ]
        unique_names = {
            "": (-1, -1)
        }  # Start with empty string so we fix any topics that failed to get a name

        # Find the singletons so we can skip them

        # Find the singletons so we can skip them
        singleton_subclusters, singleton_dict = self._get_singleton_subclusters()
        for n in range(len(self.topic_name_layers_) - 1, -1, -1):
            for i, (name, indices) in tqdm(
                enumerate(
                    zip(
                        self.topic_name_layers_[n],
                        self.cluster_layers_.pointset_layers[n],
                    )
                ),
                total=len(self.topic_name_layers_[n]),
                desc=f"Cleaning layer topic names for layer {n}",
                disable=(not self.verbose),
            ):
                n_attempts = 0
                unique_name = name
                original_topic_names = [unique_name]
                if (n, i) in singleton_dict:
                    # This is a singleton cluster, and doesn't need a name
                    continue

                if (n, i) in singleton_dict:
                    continue

                matching_topics = []
                while unique_name in unique_names and n_attempts < 3:
                    matching_topics.append(unique_names[unique_name])
                    prompt_text = create_topic_discernment_prompt(
                        layer=n,
                        topic_index=i,
                        attempted_topic_names=original_topic_names,
                        matching_topics=matching_topics,
                        representations=self.representation_,
                        subtopic_layers=self.subtopic_layers_["topical"],
                        document_type=self.document_type,
                        corpus_description=self.corpus_description,
                    )
                    unique_name = self.llm.generate_topic_name(prompt_text)
                    original_topic_names.append(unique_name)
                    n_attempts += 1

                if n_attempts > 0 and self.verbose:
                    print(f"{name} --> {unique_name} after {n_attempts} attempts")
                if unique_name not in unique_names:
                    unique_names[unique_name] = (n, i)

                if unique_name != "":
                    self.layer_clusters[n][indices] = unique_name
                else:
                    self.layer_clusters[n][indices] = name # If we failed to get a name, keep the old one
