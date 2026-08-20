from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from toponymy.evaluation import WayfindingLineup


def fixture_data():
    names = {(0, 11): "apple", (0, 80): "banana", (0, 201): "cherry"}
    documents = [f"{name} document {i}" for name in names.values() for i in range(4)]
    vectors = np.array([[value, 0.1 * i] for value in (0, 2, 20) for i in range(4)])
    model = SimpleNamespace(
        topics={
            key: SimpleNamespace(members=np.arange(index * 4, (index + 1) * 4))
            for index, key in enumerate(names)
        }
    )
    return model, documents, vectors, names


def make_lineup(**kwargs):
    model, documents, vectors, names = fixture_data()
    lineup = WayfindingLineup.from_topic_model(
        model, documents, vectors, exemplar_indices={(0, 11): [0]}, **kwargs
    )
    return lineup, names


def exact_judge(label, groups):
    return [float(all(text.startswith(label) for text in group)) for group in groups]


def test_samples_are_held_out_and_inputs_are_not_retained():
    model, documents, vectors, _ = fixture_data()
    original = vectors.copy()
    lineup = WayfindingLineup.from_topic_model(
        model,
        documents,
        vectors,
        exemplar_indices={(0, 11): [0, 1], (0, 80): [4]},
        documents_per_candidate=10,
    )
    assert len(lineup.trials) == 3
    assert not lineup.excluded_topics
    for trial in lineup.trials:
        for indices, texts in zip(trial.document_indices, trial.documents):
            assert not set(indices).intersection({0, 1, 4})
            assert len(indices) == len(set(indices))
            assert texts == tuple(documents[index] for index in indices)
    np.testing.assert_array_equal(vectors, original)
    frozen = lineup.trials[0].documents
    documents[:] = ["changed"] * len(documents)
    vectors[:] = 100
    model.topics[(0, 11)].members[:] = 0
    assert lineup.trials[0].documents == frozen


def test_seed_and_order_are_deterministic_and_sparse_ids_preserved():
    first, _ = make_lineup(seed=42, documents_per_candidate=1)
    second, _ = make_lineup(seed=42, documents_per_candidate=1)
    other, _ = make_lineup(seed=7, documents_per_candidate=1)
    assert first == second
    assert first.lineup_id != other.lineup_id
    assert {trial.topic for trial in first.trials} == {(0, 11), (0, 80), (0, 201)}


def test_distractors_are_nearest_semantic_centroids_within_layer():
    model, documents, vectors, _ = fixture_data()
    model.topics[(1, 4)] = SimpleNamespace(members=np.arange(8))
    model.topics[(1, 8)] = SimpleNamespace(members=np.arange(8, 12))
    lineup = WayfindingLineup.from_topic_model(
        model,
        documents,
        vectors,
        exemplar_indices={},
        candidates=2,
    )
    trials = {trial.topic: trial for trial in lineup.trials}
    assert set(trials[(0, 11)].candidates) == {(0, 11), (0, 80)}
    assert set(trials[(0, 201)].candidates) == {(0, 80), (0, 201)}
    assert set(trials[(1, 4)].candidates) == {(1, 4), (1, 8)}
    assert all(
        key[0] == trial.topic[0] for trial in lineup.trials for key in trial.candidates
    )


def test_no_exemplar_metadata_requires_explicit_decision():
    model, documents, vectors, _ = fixture_data()
    with pytest.raises(TypeError, match="exemplar_indices"):
        WayfindingLineup.from_topic_model(model, documents, vectors)


def test_topics_without_held_out_data_or_distractors_are_explicit():
    model, documents, vectors, names = fixture_data()
    lineup = WayfindingLineup.from_topic_model(
        model,
        documents,
        vectors,
        exemplar_indices={(0, 11): [0, 1, 2, 3], (0, 80): [4, 5, 6, 7]},
    )
    assert lineup.trials == ()
    assert lineup.excluded_topics == (
        ((0, 11), "no held-out documents"),
        ((0, 80), "no held-out documents"),
        ((0, 201), "no same-layer distractor"),
    )
    calls = []
    scores = lineup.score(names, lambda *args: calls.append(args))
    assert calls == []
    assert scores.accuracy is None
    assert lineup.compare(scores, scores).overall.conclusion == "no_trials"


def test_empty_model_produces_no_evaluation_claim():
    lineup = WayfindingLineup.from_topic_model(
        SimpleNamespace(topics={}),
        [],
        np.empty((0, 2)),
        exemplar_indices={},
    )
    assert lineup.trials == ()
    assert lineup.excluded_topics == ()


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros(12),
        np.zeros((11, 2)),
        np.zeros((12, 0)),
        np.full((12, 2), np.nan),
        np.full((12, 2), np.inf),
        np.ones((12, 2), dtype=complex),
    ],
)
def test_invalid_embeddings(vectors):
    model, documents, _, _ = fixture_data()
    with pytest.raises(ValueError, match="embedding_vectors"):
        WayfindingLineup.from_topic_model(
            model, documents, vectors, exemplar_indices={}
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("candidates", 1),
        ("candidates", True),
        ("documents_per_candidate", 0),
        ("documents_per_candidate", 1.5),
        ("seed", -1),
    ],
)
def test_invalid_configuration(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        make_lineup(**{parameter: value})


@pytest.mark.parametrize(
    "members,match",
    [
        ([0, 0], "duplicate"),
        ([0.2], "integer"),
        ([-1], "out-of-range"),
        ([12], "out-of-range"),
        ([[1]], "integer"),
        ([4], "overlap"),
    ],
)
def test_invalid_members(members, match):
    model, documents, vectors, _ = fixture_data()
    model.topics[(0, 11)].members = members
    with pytest.raises(ValueError, match=match):
        WayfindingLineup.from_topic_model(
            model, documents, vectors, exemplar_indices={}
        )


@pytest.mark.parametrize("key", [(0, -1), "a", (0,), (True, 1)])
def test_invalid_topic_keys(key):
    model, documents, vectors, _ = fixture_data()
    model.topics[key] = model.topics.pop((0, 11))
    with pytest.raises(ValueError, match="topic keys"):
        WayfindingLineup.from_topic_model(
            model, documents, vectors, exemplar_indices={}
        )


@pytest.mark.parametrize(
    "exclusions,match",
    [
        ({(4, 9): [0]}, "unknown"),
        ({(0, 11): [5]}, "belong"),
        ({(0, 11): [0, 0]}, "duplicate"),
    ],
)
def test_invalid_exemplar_exclusions(exclusions, match):
    model, documents, vectors, _ = fixture_data()
    with pytest.raises(ValueError, match=match):
        WayfindingLineup.from_topic_model(
            model, documents, vectors, exemplar_indices=exclusions
        )


def test_invalid_document_type():
    model, documents, vectors, _ = fixture_data()
    documents[0] = 42
    with pytest.raises(ValueError, match="strings"):
        WayfindingLineup.from_topic_model(
            model, documents, vectors, exemplar_indices={}
        )


def test_large_finite_embeddings_preserve_nearest_order():
    model, documents, vectors, _ = fixture_data()
    vectors *= 1e306
    lineup = WayfindingLineup.from_topic_model(
        model,
        documents,
        vectors,
        exemplar_indices={},
        candidates=2,
    )
    assert set(lineup.trials[0].candidates) == {(0, 11), (0, 80)}


def test_score_has_exact_call_count_and_anonymous_immutable_groups():
    lineup, names = make_lineup()
    calls = []

    def judge(label, groups):
        assert isinstance(label, str)
        assert isinstance(groups, tuple)
        assert all(isinstance(group, tuple) for group in groups)
        calls.append((label, groups))
        return exact_judge(label, groups)

    result = lineup.score(names, judge, repeats=2)
    assert result.accuracy == 1.0
    assert result.call_count == len(calls) == 6
    assert result.tie_count == 0
    assert calls[:3] == calls[3:]


def test_ties_are_fractional_and_visible():
    lineup, names = make_lineup()
    result = lineup.score(names, lambda label, groups: [7] * len(groups))
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.tie_count == 3
    assert result.runs[0][0].winners == (0, 1, 2)


@pytest.mark.parametrize(
    "values",
    [
        [1],
        [[1, 2, 3]],
        [1, 2, np.nan],
        [1, np.inf, 2],
        [True, False, False],
        ["a", "b", "c"],
    ],
)
def test_invalid_judge_scores_do_not_become_success(values):
    lineup, names = make_lineup()
    with pytest.raises(ValueError, match="finite numeric score"):
        lineup.score(names, lambda label, groups: values)


def test_missing_name_is_rejected_before_any_judge_calls():
    lineup, names = make_lineup()
    names.pop((0, 201))
    calls = []
    with pytest.raises(ValueError, match="name"):
        lineup.score(names, lambda *args: calls.append(args))
    assert not calls


def test_judge_failure_propagates_without_retry():
    lineup, names = make_lineup()
    calls = []

    def failing_judge(*args):
        calls.append(args)
        raise PermissionError("access denied")

    with pytest.raises(PermissionError, match="access denied"):
        lineup.score(names, failing_judge, repeats=3)
    assert len(calls) == 1


def test_paired_comparison_detects_known_bad_labels_without_claiming_measured_noise():
    lineup, names = make_lineup()
    swapped = {(0, 11): "banana", (0, 80): "cherry", (0, 201): "apple"}
    bad = lineup.score(swapped, exact_judge)
    good = lineup.score(names, exact_judge)
    comparison = lineup.compare(bad, good)
    assert comparison.overall.delta == 1.0
    assert comparison.band is None
    assert comparison.overall.conclusion == "repeat_noise_not_measured"
    assert comparison.overall.paired_trials == 3
    assert comparison.by_layer[0].delta == 1.0


def test_independent_repeat_band_and_direction():
    lineup, names = make_lineup()
    swapped = {(0, 11): "banana", (0, 80): "cherry", (0, 201): "apple"}
    bad = lineup.score(swapped, exact_judge, repeats=2)
    good = lineup.score(names, exact_judge, repeats=2)
    comparison = lineup.compare(bad, good)
    assert comparison.band == 0.0
    assert comparison.overall.conclusion == "higher_accuracy"
    assert lineup.compare(good, bad).overall.conclusion == "lower_accuracy"
    assert comparison.calls_a == comparison.calls_b == 6


def test_observed_noise_can_cover_configuration_difference():
    lineup, names = make_lineup()
    swapped = {(0, 11): "banana", (0, 80): "cherry", (0, 201): "apple"}
    good = lineup.score(names, exact_judge)
    bad = lineup.score(swapped, exact_judge)
    noisy_repeat = lineup.score(
        names, lambda label, groups: [-score for score in exact_judge(label, groups)]
    )
    comparison = lineup.compare(bad, good, repeat_scores=[noisy_repeat])
    assert comparison.band == 1.0
    assert comparison.overall.conclusion == "inside_repeat_band"
    assert comparison.repeat_calls == 3


def test_reusing_same_result_is_not_an_independent_repeat():
    lineup, names = make_lineup()
    scores = lineup.score(names, exact_judge)
    comparison = lineup.compare(scores, scores, repeat_scores=[scores])
    assert comparison.band is None
    assert comparison.repeat_calls == 0
    independent = lineup.score(names, exact_judge)
    assert lineup.compare(scores, independent).band == 0
    repeated = lineup.compare(scores, scores, repeat_scores=[independent, independent])
    assert repeated.repeat_calls == 3


def test_comparisons_require_identical_frozen_trials_and_repeat_labels():
    first, names = make_lineup(seed=0)
    second, _ = make_lineup(seed=1)
    score_a = first.score(names, exact_judge)
    score_b = second.score(names, exact_judge)
    with pytest.raises(ValueError, match="identical frozen lineup"):
        first.compare(score_a, score_b)
    third = first.score({key: "unknown" for key in names}, exact_judge)
    with pytest.raises(ValueError, match="labels"):
        first.compare(score_a, score_a, repeat_scores=[third])
    with pytest.raises(ValueError, match="identical frozen lineup"):
        first.compare(score_a, replace(score_a, runs=()))


def test_repeats_validation():
    lineup, names = make_lineup()
    with pytest.raises(ValueError, match="repeats"):
        lineup.score(names, exact_judge, repeats=0)
