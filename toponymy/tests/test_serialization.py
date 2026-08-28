import os, shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from toponymy import TopicModel, Toponymy, ToponymyClusterer, KeyphraseBuilder
from toponymy.serialization import topic_uid
from toponymy.topic_tree import TopicTree


def is_equal(model1, model2):
    equal = np.allclose(model1.embedding_vectors, model2.embedding_vectors)
    if not equal:
        print("embedding vectors not equal")
    equal &= np.allclose(model1.reduced_vectors, model2.reduced_vectors)
    if not equal:
        print("reduced vectors not equal")
    equal &= model1.cluster_tree == model2.cluster_tree
    if not equal:
        print("cluster trees not equal")
    for layer in range(len(model1.cluster_layers)):
        equal &= (model1.cluster_layers[layer] != model2.cluster_layers[layer]).nnz == 0
    if not equal:
        print("cluster layers not equal")
    equal &= model1.topic_df.equals(model2.topic_df)
    if not equal:
        print("topic_dfs not equal")
    equal &= model1.document_df.equals(model2.document_df)
    if not equal:
        print("document_dfs not equal")
    # Check topic_sizes if size column exists
    if "size" in model1.topic_df.columns:
        equal &= model1.topic_sizes == model2.topic_sizes
        if not equal:
            print("topic_sizes not equal")
    return equal


def mock_data_model():
    tree_dict = {
        (3, 0): [(2, 0), (2, 1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        [
            "Subtopic C1",
            "Subtopic C2",
            "Subtopic C3",
            "Subtopic C4",
            "Subtopic C5",
            "Subtopic C6",
            "Subtopic C7",
            "Subtopic C8",
        ],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
        ["Root"],
    ]

    data = []
    for tup in tree_dict.keys():
        l, c = tup
        data.append(
            {"layer": l, "cluster": c, "name": topics[l][c], "uid": topic_uid(tup)}
        )
    # Also add the leaf nodes (layer 0), which aren't keys in tree_dict
    for c in range(8):
        data.append(
            {"layer": 0, "cluster": c, "name": topics[0][c], "uid": topic_uid((0, c))}
        )
    topic_df = pd.DataFrame(data)

    n_samples = 100
    n_clusters_per_layer = [8, 4, 2, 1]

    # Assign each document to one leaf cluster (layer 0), evenly distributed
    rng = np.random.default_rng(3141)
    leaf_assignments = rng.integers(0, 8, size=n_samples)

    child_to_parent = {}
    for parent, children in tree_dict.items():
        for child in children:
            child_to_parent[child] = parent

    def find_ancestor_at_layer(leaf_cluster, target_layer):
        node = (0, leaf_cluster)
        while node[0] < target_layer:
            node = child_to_parent[node]
        return node[1]

    matrices = []
    for layer in range(3):
        n_clusters = n_clusters_per_layer[layer]
        matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8)
        for doc_idx, leaf_cluster in enumerate(leaf_assignments):
            cluster = find_ancestor_at_layer(leaf_cluster, layer)
            matrix[doc_idx, cluster] = 255
        matrices.append(sp.csr_matrix(matrix))

    document_df = pd.DataFrame({"item_num": range(n_samples)})

    return TopicModel(
        topic_df=topic_df,
        document_df=document_df,
        cluster_tree=tree_dict,
        cluster_layers=matrices,
        embedding_vectors=rng.standard_normal((n_samples, 32)).astype(np.float32),
        reduced_vectors=rng.standard_normal((n_samples, 2)).astype(np.float32),
    )


@pytest.fixture(scope="function")
def fitted_toponymy_newsgroups():
    """Fixture for a fitted Toponymy object with 20newsgroups data."""
    newsgroups_df = pd.read_parquet(
        "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
    )
    embeddings = np.stack(newsgroups_df["embedding"].values)
    projection = np.stack(newsgroups_df["map"].values)
    metadata = newsgroups_df[["post", "newsgroup"]]
    clusterer = ToponymyClusterer(verbose=True, base_min_cluster_size=25)

    toponymy = Toponymy(
        llm_wrapper=None,
        text_embedding_model=None,
        clusterer=clusterer,
        object_description="newsgroup posts",
        corpus_description="20-newsgroups dataset",
    )
    toponymy.cluster_layers_, toponymy.cluster_tree_ = clusterer.fit_predict(
        projection, embeddings
    )
    toponymy.embedding_vectors_ = embeddings
    toponymy.clusterable_vectors_ = projection

    # Build topic_names and topic_sizes
    topic_names = []
    topic_sizes = []
    for layer in range(len(toponymy.cluster_layers_)):
        layer_names = []
        keyphrases = []
        layer_sizes = []
        for cluster in np.unique(toponymy.cluster_layers_[layer].cluster_labels):
            if cluster >= 0:  # Skip noise
                layer_names.append(f"{layer},{cluster}")
                keyphrases.append(np.array(["lorem", "ipsum"], dtype=object))
                # Count documents in this cluster
                size = np.sum(toponymy.cluster_layers_[layer].cluster_labels == cluster)
                layer_sizes.append(int(size))
        toponymy.cluster_layers_[layer].keyphrases = keyphrases
        topic_names.append(layer_names)
        topic_sizes.append(layer_sizes)

    toponymy.topic_names_ = topic_names
    toponymy.topic_sizes_ = topic_sizes

    return toponymy, metadata


def test_round_trip_lance():
    path = "test_model_lance"
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    model = mock_data_model()
    model.to_lance(path)
    model2 = model.from_lance(path)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    assert is_equal(model, model2)


def test_round_trip_zip():
    path = "test_model.tm.zip"
    model = mock_data_model()
    model.to_file(path)
    model2 = model.from_file(path)
    if os.path.exists(path):
        os.remove(path)
    assert is_equal(model, model2)


def test_topic_names():
    model = mock_data_model()
    topics = [
        [
            "Subtopic C1",
            "Subtopic C2",
            "Subtopic C3",
            "Subtopic C4",
            "Subtopic C5",
            "Subtopic C6",
            "Subtopic C7",
            "Subtopic C8",
        ],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    assert model.topic_names == topics


def test_topic_tree_property():
    """Test that TopicModel.topic_tree() method returns a valid TopicTree."""
    model = mock_data_model()

    # Get the topic tree
    topic_tree = model.topic_tree()

    # Verify it's a TopicTree instance
    assert isinstance(topic_tree, TopicTree)

    # Verify it has the correct attributes
    assert topic_tree.tree == model.cluster_tree
    assert topic_tree.topics == model.topic_names
    assert topic_tree.topic_sizes == model.topic_sizes
    assert topic_tree.n_objects == model.embedding_vectors.shape[0]

    # Verify it can be converted to string (print functionality)
    tree_string = str(topic_tree)
    assert "Topic tree:" in tree_string
    assert "Topic A" in tree_string
    assert "Topic B" in tree_string
    assert "Subtopic" in tree_string

    # Verify it can be rendered to HTML
    html_output = topic_tree._repr_html_()
    assert '<div class="topic-tree">' in html_output
    assert "Topic A" in html_output
    assert "Topic B" in html_output

    # Verify print method works
    topic_tree.print()  # Should not raise an exception

    # Test with prune_duplicates parameter
    topic_tree_no_prune = model.topic_tree(prune_duplicates=False)
    assert isinstance(topic_tree_no_prune, TopicTree)


def test_from_toponymy(fitted_toponymy_newsgroups, premade_topic_model_path):
    """Test that TopicModel.from_toponymy correctly extracts all properties including topic_sizes."""
    toponymy, metadata = fitted_toponymy_newsgroups

    test_model = TopicModel.from_toponymy(toponymy, document_df=metadata)
    ## This doesn't seem to work on Azure, but it does work locally.
    # good_model = TopicModel.from_file(premade_topic_model_path)
    ## instead we can just test the test_model has correct properties
    n_topics = sum([len(x) for x in toponymy.topic_names_])
    assert (test_model.embedding_vectors == toponymy.embedding_vectors_).all()
    assert (test_model.reduced_vectors == toponymy.clusterable_vectors_).all()
    assert test_model.cluster_tree == toponymy.cluster_tree_
    assert len(test_model.topic_df) == n_topics
    assert len(test_model.document_df) == test_model.embedding_vectors.shape[0]

    # Verify topic_sizes were properly serialized
    assert "size" in test_model.topic_df.columns, "Size column should exist in topic_df"
    assert (
        test_model.topic_sizes == toponymy.topic_sizes_
    ), "Topic sizes should match the original"


def test_topic_tree_from_fitted_toponymy(fitted_toponymy_newsgroups):
    """Test that topic_tree() method works with models created from fitted Toponymy."""
    toponymy, metadata = fitted_toponymy_newsgroups

    # Create TopicModel from fitted Toponymy
    model = TopicModel.from_toponymy(toponymy, document_df=metadata)

    # Get the topic tree
    topic_tree = model.topic_tree()

    # Verify it's a TopicTree instance
    assert isinstance(topic_tree, TopicTree)

    # Verify it has the correct number of objects
    assert topic_tree.n_objects == model.embedding_vectors.shape[0]

    # Verify topics match
    assert topic_tree.topics == model.topic_names

    # Verify it can be converted to string
    tree_string = str(topic_tree)
    assert "Topic tree:" in tree_string

    # Verify it can be rendered to HTML
    html_output = topic_tree._repr_html_()
    assert '<div class="topic-tree">' in html_output


def test_round_trip_zip_with_size_column():
    """Test zip serialization with size column (newer models with topic_sizes)."""
    path = "test_model_sizes.tm.zip"
    model = mock_data_model()

    # Add size column to topic_df (simulating newer models)
    expected_sizes = []
    for layer_matrix in model.cluster_layers:
        if sp.issparse(layer_matrix):
            sizes = np.asarray(layer_matrix.sum(axis=0)).ravel().tolist()
        else:
            sizes = layer_matrix.sum(axis=0).tolist()
        expected_sizes.append([int(s // 255) for s in sizes])

    size_map = {}
    for layer_idx, layer_sizes in enumerate(expected_sizes):
        for cluster_idx, size in enumerate(layer_sizes):
            size_map[(layer_idx, cluster_idx)] = size

    model.topic_df["size"] = model.topic_df.apply(
        lambda row: size_map.get((row["layer"], row["cluster"]), 0), axis=1
    )

    model.to_file(path)
    model2 = model.from_file(path)

    if os.path.exists(path):
        os.remove(path)

    # is_equal checks everything including topic_sizes when size column exists
    assert is_equal(model, model2)


def test_round_trip_lance_with_size_column():
    """Test lance serialization with size column (newer models with topic_sizes)."""
    path = "test_model_sizes_lance"
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

    model = mock_data_model()

    # Add size column to topic_df (simulating newer models)
    expected_sizes = []
    for layer_matrix in model.cluster_layers:
        if sp.issparse(layer_matrix):
            sizes = np.asarray(layer_matrix.sum(axis=0)).ravel().tolist()
        else:
            sizes = layer_matrix.sum(axis=0).tolist()
        expected_sizes.append([int(s // 255) for s in sizes])

    size_map = {}
    for layer_idx, layer_sizes in enumerate(expected_sizes):
        for cluster_idx, size in enumerate(layer_sizes):
            size_map[(layer_idx, cluster_idx)] = size

    model.topic_df["size"] = model.topic_df.apply(
        lambda row: size_map.get((row["layer"], row["cluster"]), 0), axis=1
    )

    model.to_lance(path)
    model2 = model.from_lance(path)

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

    # is_equal checks everything including topic_sizes when size column exists
    assert is_equal(model, model2)


def test_topic_sizes_fallback_without_size_column():
    """Test that topic_sizes can be reconstructed from cluster_layers if size column is missing."""
    model = mock_data_model()

    # Compute expected sizes from cluster_layers
    expected_sizes = []
    for layer_matrix in model.cluster_layers:
        if sp.issparse(layer_matrix):
            sizes = np.asarray(layer_matrix.sum(axis=0)).ravel().tolist()
        else:
            sizes = layer_matrix.sum(axis=0).tolist()
        expected_sizes.append([int(s // 255) for s in sizes])

    # Access topic_sizes without size column (should use fallback)
    sizes = model.topic_sizes

    assert (
        sizes == expected_sizes
    ), "topic_sizes should be computed from cluster_layers as fallback"
