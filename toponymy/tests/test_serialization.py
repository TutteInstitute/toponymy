import numpy as np
import pandas as pd

from toponymy import TopicModel
from toponymy.serialization import topic_uid

def is_equal(model1, model2):
  equal = np.allclose(model1.embedding_vectors,model2.embedding_vectors)
  equal &= np.allclose(model1.reduced_vectors,model2.reduced_vectors)
  equal &= model1.cluster_tree==model2.cluster_tree
  for layer in range(len(model1.cluster_layers)):
        equal &= ((model1.cluster_layers[layer] != model2.cluster_layers[layer]).nnz == 0)
  equal &= model1.topic_df.equals(model2.topic_df)
  equal &= model1.document_df.equals(model2.document_df)
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
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4",
         "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
        ["Root"],
    ]

    data = []
    for tup in tree_dict.keys():
        l, c = tup
        data.append({
            'layer': l,
            'cluster': c,
            'name': topics[l][c], 
            'uid': topic_uid(tup)
        })
    # Also add the leaf nodes (layer 0), which aren't keys in tree_dict
    for c in range(8):
        data.append({
            'layer': 0,
            'cluster': c,
            'name': topics[0][c],
            'uid': topic_uid((0, c))
        })
    topic_df = pd.DataFrame(data)
    topic_df.set_index('uid', inplace=True)

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
    for layer in range(4):
        n_clusters = n_clusters_per_layer[layer]
        matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8)
        for doc_idx, leaf_cluster in enumerate(leaf_assignments):
            cluster = find_ancestor_at_layer(leaf_cluster, layer)
            matrix[doc_idx, cluster] = 255
        matrices.append(sp.csr_matrix(matrix))

    document_df = pd.DataFrame({'item_num':range(n_samples)})

    return TopicModel(
        topic_df=topic_df,
        document_df=document_df,
        cluster_tree=tree_dict,
        cluster_layers=matrices,
        embedding_vectors=rng.standard_normal((n_samples, 32)).astype(np.float32),
        reduced_vectors=rng.standard_normal((n_samples, 2)).astype(np.float32),
    )

def test_round_trip_lance():
    path = 'tmp/test_model_lance'
    model = mock_data_model()
    model.to_lance(path)
    model2 = model.from_lance(path)
    assert is_equal(model, model2)

def test_round_trip_zip():
    path = 'tmp/test_model.tm.zip'
    model = mock_data_model()
    model.to_file(path)
    model2 = model.from_file(path)
    assert is_equal(model, model2)
