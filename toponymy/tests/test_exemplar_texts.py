from typing import Literal
import pytest
import json
import numpy as np
from toponymy.exemplar_texts import random_exemplars, diverse_exemplars
from pathlib import Path
import sentence_transformers

from toponymy.clustering import (
    centroids_from_labels,
)

EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
 
TEST_OBJECTS = json.load(open(Path(__file__).parent / "test_objects.json", "r"))
TOPIC_OBJECTS = json.load(open(Path(__file__).parent / "topic_objects.json", "r"))
ALL_TOPIC_OBJECTS = sum([x["paragraphs"] for x in TOPIC_OBJECTS], [])
CLUSTER_LAYER = np.concatenate([np.arange(10).repeat(10), np.full(10, -1)])
TOPIC_VECTORS = EMBEDDER.encode(ALL_TOPIC_OBJECTS)
CENTROID_VECTORS = centroids_from_labels(CLUSTER_LAYER, TOPIC_VECTORS)


def test_json_load():
    assert len(TEST_OBJECTS)==125

@pytest.mark.parametrize("n_exemplars", [4, 15])
def test_random_exemplar(n_exemplars):
    #ALL_TOPIC_OBJECTS -> paragraphs = np.asarray(sum([x['paragraphs'] for x in TOPIC_OBJECTS], []))  
    #CLUSTER_LAYER -> cluster_label_vector = np.concatenate([[i]*len(x['paragraphs']) for i,x in enumerate(TOPIC_OBJECTS)])  
    exemplars = random_exemplars(CLUSTER_LAYER, ALL_TOPIC_OBJECTS, n_exemplars=n_exemplars)
    assert len(exemplars)==CLUSTER_LAYER.max()+1
    
    for i in range(CLUSTER_LAYER.max()+1):
        assert len(exemplars[i])==min(n_exemplars, len(TOPIC_OBJECTS[i]['paragraphs']))

@pytest.mark.parametrize("n_exemplars", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("method", ['centroid','random'])
def test_diverse_exemplar_result_sizes(n_exemplars: Literal[3] | Literal[5], diversify_alpha: float, method: str):
    exemplar_results, exemplar_indices = diverse_exemplars(
        cluster_label_vector = CLUSTER_LAYER,
        objects = ALL_TOPIC_OBJECTS,
        object_vectors = TOPIC_VECTORS,
        centroid_vectors = CENTROID_VECTORS,
        n_exemplars= n_exemplars,
        diversify_alpha = diversify_alpha,
        method=method,
    )
    assert len(exemplar_results) == len(np.unique(CLUSTER_LAYER)) - 1
    if diversify_alpha == 1.0:
        print(exemplar_results)
    print([len(x) for x in exemplar_results])
    assert all([len(x) == n_exemplars for x in exemplar_results]) and exemplar_results
    assert all([len(set(x)) == n_exemplars for x in exemplar_results])


@pytest.mark.parametrize("method", ['centroid','random'])
def test_empty_cluster_diverse(method: str):
    new_clustering = CLUSTER_LAYER
    new_clustering[new_clustering==0] = 9
    exemplar_results, exemplar_indices = diverse_exemplars(
        cluster_label_vector = new_clustering,
        objects = ALL_TOPIC_OBJECTS,
        object_vectors = TOPIC_VECTORS,
        centroid_vectors = CENTROID_VECTORS,
        method=method,
    )
    assert len(exemplar_results[0])==0

def test_empty_cluster_random():
    new_clustering = CLUSTER_LAYER
    new_clustering[new_clustering==0] = 9
    exemplar_results, exemplar_indices = random_exemplars(
        cluster_label_vector = new_clustering,
        objects = ALL_TOPIC_OBJECTS,
    )
    assert len(exemplar_results[0])==0