"""Exercise the public pipeline with deterministic local data and responses.

Run ``python examples/local_pipeline.py`` from an installed source checkout.
Use ``--output topics.toponymy`` to also save and reload the result. The names
are scripted examples, not evidence of semantic quality from a language model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from toponymy import PrecomputedClusterer, Toponymy, TopicModel
from toponymy.templates import Prompt


class LocalNamer:
    """Deterministic fake at the provider boundary; makes no network requests."""

    def __init__(self):
        self.calls = 0

    def generate_topic_name(self, prompt: Prompt, *, response_parser):
        self.calls += 1
        words = prompt.user.lower()
        subjects = [word for word in ("orchard", "river") if word in words]
        response = {
            "topic_name": " and ".join(subjects).title(),
            "topic_specificity": 0.8,
        }
        return response_parser(json.dumps(response))

    def generate_topic_cluster_names(self, prompt, old_names, *, response_parser):
        self.calls += 1
        response = {
            "new_topic_name_mapping": {
                str(i): f"{name} ({i})" for i, name in enumerate(old_names, 1)
            },
            "topic_specificities": [0.8] * len(old_names),
        }
        return response_parser(json.dumps(response))


def sample_data():
    objects = [
        "An orchard grows apples.",
        "The orchard harvest begins in autumn.",
        "A river flows toward the coast.",
        "Rain raises the river level.",
    ]
    semantic_vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = [np.array([10, 10, 42, 42]), np.array([7, 7, 7, 7])]
    return objects, semantic_vectors, labels


def run_example(output: Path | None = None):
    objects, semantic_vectors, labels = sample_data()
    provider = LocalNamer()
    pipeline = Toponymy(
        provider,
        clusterer=PrecomputedClusterer(labels),
        object_description="notes",
        corpus_description="four short notes",
    )
    pipeline.prepare(objects, semantic_vectors)
    assert provider.calls == 0
    assert isinstance(pipeline.topics_[(0, 10)].prompt, Prompt)
    print("Prepared topic keys:", list(pipeline.topics_))
    pipeline.name_topics()
    print("Names by original cluster ID:", pipeline.topic_names_)
    print("Provider calls:", provider.calls)

    if output is not None:
        pipeline.topic_model_.to_file(str(output))
        loaded = TopicModel.from_file(str(output))
        assert loaded.topic_names == pipeline.topic_names_
        assert set(loaded.topics) == set(pipeline.topics_)
        print("Reloaded:", output)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run_example(parser.parse_args().output)
