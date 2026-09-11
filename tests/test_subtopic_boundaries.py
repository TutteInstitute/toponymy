"""Selection oracles independent of provider output and document frequency."""

import numpy as np
import pytest

from toponymy import subtopics as st

NAMES = ["East", "North", "Northeast", "West", "South", "Southwest"]
VECTORS = np.array([[3, 0], [0, 3], [2, 2], [-3, 0], [0, -3], [-2, -2]], dtype=float)
PARENTS = np.array([2, 2, 2, 7, 7, 7])
CHILDREN = np.arange(6)


class ForbiddenEmbedder:
    def encode(self, texts):
        raise AssertionError("No encoding is needed for this boundary")


@pytest.mark.parametrize("extra_rows", [False, True])
def test_centering_preserves_tiny_directions_after_large_offset_cancels(extra_rows):
    vectors = np.array(
        [[1e300, -1e-300], [1e300, 2e-300], [1e300, 3e-300], [1e300, -4e-300]]
    )
    parents = [0, 0, 0, 1]
    if extra_rows:
        vectors = np.vstack(
            (
                vectors,
                [[np.nextafter(1e300, np.inf), 0], [np.nextafter(1e300, -np.inf), 0]],
            )
        )
        parents.extend([1, 1])
    names = [str(i) for i in range(len(vectors))]
    selected = st.central_subtopics(
        np.array(parents), names, np.arange(len(names)), vectors, n_subtopics=1
    )
    assert selected[0] == ["1"]  # +y query; child 0 points in the opposite direction.


def test_unsigned_parent_labels_keep_finite_legacy_slots():
    assert st.central_subtopics(
        np.array([2], dtype=np.uint8),
        ["Only"],
        np.array([0], dtype=np.uint8),
        np.array([[1.0, 2.0]]),
    ) == [[], [], ["Only"]]


def test_all_candidate_query_keeps_small_true_child_direction():
    assert st.central_subtopics_from_all_subtopics(
        np.array([0]),
        ["Huge unrelated", "Tiny child"],
        np.array([1]),
        np.array([[0.0, 1e300], [1e-300, 0.0]]),
        n_subtopics=1,
    ) == [["Tiny child"]]


@pytest.mark.parametrize("score", [0.0, np.nan])
def test_unavailable_information_falls_back_before_weighted_average(monkeypatch, score):
    from scipy import sparse

    class Coder:
        def __init__(self, **options):
            pass

        def fit(self, vectors):
            return self

        def transform(self, vectors):
            return np.ones((len(vectors), 2))

    class Uninformative:
        def __init__(self, **options):
            pass

        def fit_transform(self, coding, y):
            return sparse.csr_array(np.full(coding.shape, score))

    monkeypatch.setattr(st, "DictionaryLearning", Coder)
    monkeypatch.setattr(st, "InformationWeightTransformer", Uninformative)
    assert st.information_weighted_subtopics(
        PARENTS, NAMES, CHILDREN, VECTORS, n_subtopics=1
    ) == [[], [], ["Northeast"], [], [], [], [], ["Southwest"]]


def test_information_scores_preserve_raw_dictionary_input_and_rank_strength(
    monkeypatch,
):
    seen = []
    codes = np.array(
        [[5, 0, 1], [3, 0, 1], [1, 0, 1], [0, 5, 1], [0, 3, 1], [0, 1, 1]], dtype=float
    )

    class Coder:
        def __init__(self, **options):
            assert options["transform_alpha"] == 0.25
            assert options["positive_code"] is True

        def fit(self, vectors):
            seen.append(vectors.copy())
            return self

        def transform(self, vectors):
            np.testing.assert_array_equal(vectors, seen[-1])
            return codes

    monkeypatch.setattr(st, "DictionaryLearning", Coder)
    for scale in (1.0, 7.0):
        vectors = VECTORS * scale
        selected = st.information_weighted_subtopics(
            PARENTS,
            NAMES,
            CHILDREN,
            vectors,
            n_subtopics=1,
            diversify_alpha=0,
            coding_transform_alpha=0.25,
        )
        np.testing.assert_array_equal(seen[-1], vectors)
        assert selected == [[], [], ["East"], [], [], [], [], ["West"]]
    assert len(seen) == 2


def test_information_extreme_dictionary_magnitude_fails_before_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid squared objective must be rejected before fit")

    monkeypatch.setattr(st, "DictionaryLearning", forbidden)
    with pytest.raises(ValueError, match="numerical range of dictionary"):
        st.information_weighted_subtopics(PARENTS, NAMES, CHILDREN, VECTORS * 1e300)


@pytest.mark.parametrize(
    "dtype,scale",
    [
        (np.float32, 1),
        (np.float32, 1e35),
        (np.float32, 1e-35),
        (np.float64, 1e300),
        (np.float64, 1e-300),
    ],
)
def test_central_direction_oracle_sparse_parents_readonly(dtype, scale):
    backing = np.zeros((6, 4), dtype=dtype)
    backing[:, ::2] = VECTORS * scale
    vectors = backing[:, ::2]
    vectors.flags.writeable = False
    before = backing.copy()
    actual = st.central_subtopics(
        PARENTS, NAMES, CHILDREN, vectors, n_subtopics=1, diversify_alpha=0.8
    )
    assert actual == [[], [], ["Northeast"], [], [], [], [], ["Southwest"]]
    np.testing.assert_array_equal(backing, before)


def test_all_candidates_centroid_excludes_noise_before_indexing():
    actual = st.central_subtopics_from_all_subtopics(
        np.array([0, 0]),
        ["True child", "Unrelated diagonal", "Last row"],
        np.array([0, -1]),
        np.array([[1.0, 0.0], [1.0, 3.0], [0.0, 3.0]]),
        n_subtopics=1,
    )
    assert actual == [["True child"]]


@pytest.mark.parametrize(
    "select",
    [
        st.central_subtopics,
        st.submodular_subtopics,
        st.information_weighted_subtopics,
        st.central_subtopics_from_all_subtopics,
    ],
)
@pytest.mark.parametrize(
    "parents,children,expected",
    [([], [], []), ([-1], [-1], []), ([3], [-1], [[], [], [], []])],
)
def test_empty_groups_keep_legacy_slots_without_embedding(
    select, parents, children, expected
):
    assert (
        select(
            np.array(parents, dtype=int),
            [],
            np.array(children, dtype=int),
            embedding_model=ForbiddenEmbedder(),
        )
        == expected
    )


@pytest.mark.parametrize(
    "select",
    [
        st.central_summary_subtopics,
        st.submodular_summary_subtopics,
        st.information_weighted_summary_subtopics,
    ],
)
def test_empty_summary_slots_are_strings(select):
    assert select(
        np.array([3]), [], [], [], np.array([-1]), embedding_model=ForbiddenEmbedder()
    ) == ["", "", "", ""]


def test_summary_selection_keeps_exact_unicode_bundle_order():
    summaries = [f"Résumé {i}\nmore" for i in range(6)]
    explanations = [f"原因 {i}" for i in range(6)]
    actual = st.central_summary_subtopics(
        PARENTS, NAMES, summaries, explanations, CHILDREN, VECTORS, n_subtopics=1
    )
    assert actual == [
        "",
        "",
        "Northeast\nRésumé 2\nmore\n原因 2",
        "",
        "",
        "",
        "",
        "Southwest\nRésumé 5\nmore\n原因 5",
    ]


def test_information_crossing_child_rejected_before_embedding():
    with pytest.raises(ValueError, match="multiple|more than one"):
        st.information_weighted_subtopics(
            np.array([2, 7]),
            ["Shared"],
            np.array([0, 0]),
            embedding_model=ForbiddenEmbedder(),
        )


@pytest.mark.parametrize("vectors", [np.zeros((1, 2)), np.array([[1.0, 2.0]])])
def test_information_singleton_has_representative_fallback(vectors):
    assert st.information_weighted_subtopics(
        np.array([2, 7]), ["Only child"], np.array([0, -1]), vectors
    ) == [[], [], ["Only child"], [], [], [], [], []]


@pytest.mark.parametrize("method", ["facility_location", "saturated_coverage"])
@pytest.mark.parametrize("count", [1, 3, 4])
def test_submodular_selection_is_invariant_to_common_finite_scale(method, count):
    options = dict(n_subtopics=count, submodular_function=method)
    expected = st.submodular_subtopics(PARENTS, NAMES, CHILDREN, VECTORS, **options)
    for scale in (1e-300, 1e300):
        actual = st.submodular_subtopics(
            PARENTS, NAMES, CHILDREN, VECTORS * scale, **options
        )
        assert actual == expected
    for parent, allowed in ((2, NAMES[:3]), (7, NAMES[3:])):
        assert len(expected[parent]) == min(count, 3)
        assert set(expected[parent]) <= set(allowed)
