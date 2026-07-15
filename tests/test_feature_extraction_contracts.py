from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from toponymy.new_feature_extractor import TextExemplarExtractor, TextKeyphraseExtractor


def layers(*labels):
    return [SimpleNamespace(labels=np.asarray(values, dtype=np.int64)) for values in labels]


def test_features_are_fit_results_and_refit_changes_objects():
    extractor = TextExemplarExtractor(
        "random", n_exemplars=1, object_to_text_function=lambda values: values
    )
    with pytest.raises(NotFittedError):
        extractor.predict()
    clustering = layers([11, -1, 200])
    assert extractor.fit_predict(["first", "noise", "last"], clustering) == [
        [["first"], ["last"]]
    ]
    assert extractor.features is extractor.predict()
    assert extractor.fit_predict(["new", "ignored", "other"], clustering) == [
        [["new"], ["other"]]
    ]
    assert not hasattr(clustering[0], "features")


@pytest.mark.parametrize("labels", [[], [-1, -1]])
@pytest.mark.parametrize("method", TextExemplarExtractor.supported_selection_methods)
def test_empty_clusterings_need_no_embeddings(labels, method):
    objects = ["ignored"] * len(labels)
    assert TextExemplarExtractor(method).fit_predict(objects, layers(labels)) == [[]]
    assert TextKeyphraseExtractor().fit_predict(objects, layers(labels)) == [[]]


def test_invalid_configuration_and_labels_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        TextExemplarExtractor("unknown").fit_predict(["a"], layers([0]))
    with pytest.raises(ValueError, match="positive"):
        TextExemplarExtractor(n_exemplars=0).fit_predict(["a"], layers([0]))
    with pytest.raises(ValueError, match="match"):
        TextExemplarExtractor().fit_predict(["a"], layers([0, 1]))
    with pytest.raises(ValueError, match="integers"):
        TextExemplarExtractor().fit_predict(["a"], [SimpleNamespace(labels=[0.5])])


def test_failed_refit_does_not_expose_previous_results():
    extractor = TextExemplarExtractor("random", object_to_text_function=list)
    extractor.fit_predict(["old"], layers([0]))
    with pytest.raises(ValueError):
        extractor.fit_predict(["new"], layers([0]), selection_method="unknown")
    with pytest.raises(NotFittedError):
        extractor.predict()
