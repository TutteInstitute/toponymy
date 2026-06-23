import numpy as np
import pytest

from toponymy.feature_extractor import centroids_from_labels, FeatureExtractorBase


def test_cannot_create_abstract_feature_extractor():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class FeatureExtractorBase without an implementation for abstract methods 'fit', 'get_cluster_features'",
    ):
        FeatureExtractorBase()
