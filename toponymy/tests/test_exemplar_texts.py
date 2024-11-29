import pytest
import json
import numpy as np
from toponymy.exemplar_texts import random_sample_exemplar
from pathlib import Path

#import sentence_transformers
#EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
 
TEST_OBJECTS = json.load(open(Path(__file__).parent / "test_objects.json", "r"))
TOPIC_OBJECTS = json.load(open(Path(__file__).parent / "topic_objects.json", "r"))

def test_json_load():
    print()
    assert len(TEST_OBJECTS)==125

@pytest.mark.parametrize("n_samples", [4, 15])
def test_random_exemplar(n_samples):
    paragraphs = np.asarray(sum([x['paragraphs'] for x in TOPIC_OBJECTS], []))
    cluster_label_vector = np.concatenate([[i]*len(x['paragraphs']) for i,x in enumerate(TOPIC_OBJECTS)])
    exemplars = random_sample_exemplar(cluster_label_vector, paragraphs, n_samples=n_samples)
    assert len(exemplars)==len(TOPIC_OBJECTS)
    for i,x in enumerate(TOPIC_OBJECTS):
        #print(f'{len(exemplars[i])} {n_samples} {len(x['paragraphs'])}' )
        assert len(exemplars[i])==min(n_samples, len(x['paragraphs']))


    