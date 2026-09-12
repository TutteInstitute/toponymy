"""Opt-in, paired label evaluation using frozen held-out document lineups.

No model is loaded or called automatically. A caller supplies a judge returning
one finite score per anonymous candidate group; larger scores indicate a better
match. Tied winners receive fractional credit and are reported separately.
The empirical repeat band describes observed rerun variation, not a confidence
interval or evidence that a particular judge measures semantic quality.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol

import numpy as np

TopicKey = tuple[int, int]
DocumentGroups = tuple[tuple[str, ...], ...]
Judge = Callable[[str, DocumentGroups], Sequence[float]]


class _Topic(Protocol):
    members: Sequence[int]


class _TopicModel(Protocol):
    topics: Mapping[TopicKey, _Topic]


@dataclass(frozen=True)
class LineupTrial:
    topic: TopicKey
    candidates: tuple[TopicKey, ...]
    document_indices: tuple[tuple[int, ...], ...]
    documents: DocumentGroups

    @property
    def target_position(self) -> int:
        return self.candidates.index(self.topic)


@dataclass(frozen=True)
class LineupJudgment:
    scores: tuple[float, ...]
    winners: tuple[int, ...]
    credit: float

    @property
    def tied(self) -> bool:
        return len(self.winners) > 1


@dataclass(frozen=True)
class LineupScores:
    lineup_id: str
    names: tuple[str, ...]
    runs: tuple[tuple[LineupJudgment, ...], ...]

    @property
    def call_count(self) -> int:
        return sum(len(run) for run in self.runs)

    @property
    def tie_count(self) -> int:
        return sum(item.tied for run in self.runs for item in run)

    @property
    def accuracy(self) -> float | None:
        credits = [item.credit for run in self.runs for item in run]
        return float(np.mean(credits)) if credits else None


@dataclass(frozen=True)
class PairedResult:
    layer: int | None
    paired_trials: int
    accuracy_a: float | None
    accuracy_b: float | None
    delta: float | None
    band: float | None
    conclusion: str


@dataclass(frozen=True)
class LineupComparison:
    overall: PairedResult
    by_layer: tuple[PairedResult, ...]
    calls_a: int
    calls_b: int
    repeat_calls: int
    ties_a: int
    ties_b: int

    @property
    def band(self) -> float | None:
        return self.overall.band


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _indices(values: Sequence[int], length: int, name: str) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must contain integer indices")
    if array.size and (np.any(array < 0) or np.any(array >= length)):
        raise ValueError(f"{name} contains out-of-range indices")
    result = tuple(int(value) for value in array)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return result


@dataclass(frozen=True)
class WayfindingLineup:
    """A reusable evaluation dataset; construction performs no judge calls.

    Exemplar exclusions must be supplied explicitly. For comparisons with
    different naming examples, pass the union of examples used by all arms.
    An empty mapping asserts that no naming exemplars were used. Exclusions are
    global across layers, preventing an exemplar from reappearing as evidence in
    another trial. Topics without held-out documents or a distractor are recorded
    in ``excluded_topics`` and do not produce vacuous one-candidate trials.
    """

    trials: tuple[LineupTrial, ...]
    excluded_topics: tuple[tuple[TopicKey, str], ...]
    lineup_id: str

    @classmethod
    def from_topic_model(
        cls,
        topic_model: _TopicModel,
        objects: Sequence[str],
        embedding_vectors: np.ndarray,
        *,
        exemplar_indices: Mapping[TopicKey, Sequence[int]],
        candidates: int = 3,
        documents_per_candidate: int = 3,
        seed: int = 0,
    ) -> "WayfindingLineup":
        """Freeze samples and nearest same-layer Euclidean centroid distractors.

        ``embedding_vectors`` must be semantic object embeddings with one finite
        row per document. Full cluster membership determines the centroids; only
        the frozen held-out samples are shown to the judge. Source arrays, topic
        objects and document lists are never retained or modified.
        """
        _positive_integer(candidates, "candidates", minimum=2)
        _positive_integer(documents_per_candidate, "documents_per_candidate")
        _positive_integer(seed, "seed", minimum=0)
        documents = tuple(objects)
        if not all(isinstance(item, str) for item in documents):
            raise ValueError("objects must contain strings")
        vectors = np.asarray(embedding_vectors)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(documents)
            or vectors.shape[1] == 0
            or vectors.dtype.kind not in "iuf"
            or not np.all(np.isfinite(vectors))
        ):
            raise ValueError("embedding_vectors must be finite rows aligned to objects")
        keys = list(topic_model.topics)
        for key in keys:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or value < 0
                    for value in key
                )
            ):
                raise ValueError(
                    "topic keys must be non-negative (layer, cluster) pairs"
                )
        keys = sorted((int(key[0]), int(key[1])) for key in keys)
        members = {
            key: _indices(topic_model.topics[key].members, len(documents), "members")
            for key in keys
        }
        occupied: dict[int, set[int]] = {}
        for key in keys:
            previous = occupied.setdefault(key[0], set())
            if previous.intersection(members[key]):
                raise ValueError("topic members overlap within a layer")
            previous.update(members[key])
        exclusions: set[int] = set()
        for key, values in exemplar_indices.items():
            if key not in members:
                raise ValueError("exemplar_indices contains an unknown topic")
            indices = _indices(values, len(documents), "exemplar_indices")
            if not set(indices).issubset(members[key]):
                raise ValueError("exemplar indices must belong to their topic")
            exclusions.update(indices)

        rng = np.random.default_rng(seed)
        scale = (
            max(float(vectors.max()), -float(vectors.min()), 1.0)
            if vectors.size
            else 1.0
        )
        samples: dict[TopicKey, tuple[int, ...]] = {}
        centroids: dict[TopicKey, np.ndarray] = {}
        excluded = []
        for key in keys:
            available = sorted(set(members[key]) - exclusions)
            if not available:
                excluded.append((key, "no held-out documents"))
                continue
            size = min(documents_per_candidate, len(available))
            samples[key] = tuple(
                int(index) for index in rng.choice(available, size, replace=False)
            )
            centroids[key] = (
                vectors[list(members[key])].astype(np.float64) / scale
            ).mean(axis=0)

        trials = []
        for key in keys:
            if key not in samples:
                continue
            peers = [other for other in samples if other[0] == key[0] and other != key]
            if not peers:
                excluded.append((key, "no same-layer distractor"))
                continue
            peers.sort(
                key=lambda other: (
                    float(np.hypot.reduce(centroids[key] - centroids[other])),
                    other,
                )
            )
            selected = [key, *peers[: candidates - 1]]
            order = rng.permutation(len(selected))
            selected = tuple(selected[int(position)] for position in order)
            indices = tuple(samples[other] for other in selected)
            groups = tuple(
                tuple(documents[index] for index in group) for group in indices
            )
            trials.append(LineupTrial(key, selected, indices, groups))
        trial_tuple = tuple(trials)
        payload = [
            (trial.topic, trial.candidates, trial.document_indices, trial.documents)
            for trial in trial_tuple
        ]
        lineage = json.dumps([payload, sorted(excluded)], ensure_ascii=False).encode(
            "utf-8"
        )
        return cls(
            trial_tuple, tuple(sorted(excluded)), hashlib.sha256(lineage).hexdigest()
        )

    def score(
        self,
        names: Mapping[TopicKey, str],
        judge: Judge,
        *,
        repeats: int = 1,
    ) -> LineupScores:
        """Judge every frozen trial once per repeat without retries or fallbacks.

        Exceptions propagate immediately. The judge sees a label and anonymous
        document groups, never cluster identities or the correct position.
        ``call_count`` is exactly ``len(trials) * repeats`` on success.
        """
        _positive_integer(repeats, "repeats")
        labels = []
        for trial in self.trials:
            label = names.get(trial.topic)
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"A non-empty name is required for topic {trial.topic}"
                )
            labels.append(label)
        runs = []
        for _ in range(repeats):
            judgments = []
            for trial, label in zip(self.trials, labels):
                values = np.asarray(judge(label, trial.documents))
                if (
                    values.shape != (len(trial.candidates),)
                    or values.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError(
                        "Judge must return one finite numeric score per candidate"
                    )
                winners = tuple(
                    int(index) for index in np.flatnonzero(values == values.max())
                )
                credit = 1.0 / len(winners) if trial.target_position in winners else 0.0
                judgments.append(
                    LineupJudgment(tuple(map(float, values)), winners, credit)
                )
            runs.append(tuple(judgments))
        return LineupScores(self.lineup_id, tuple(labels), tuple(runs))

    def compare(
        self,
        scores_a: LineupScores,
        scores_b: LineupScores,
        *,
        repeat_scores: Sequence[LineupScores] = (),
    ) -> LineupComparison:
        """Compare paired trial means; positive deltas favor the second arm.

        A repeat band requires two runs of the *same* labels, either through
        ``score(repeats=...)`` or additional ``repeat_scores``. The band is the
        largest observed within-arm accuracy range. It is never inferred from
        the difference between the two configurations being compared.
        """
        extras = tuple(repeat_scores)
        for scores in (scores_a, scores_b, *extras):
            if (
                scores.lineup_id != self.lineup_id
                or len(scores.names) != len(self.trials)
                or not scores.runs
                or any(len(run) != len(self.trials) for run in scores.runs)
            ):
                raise ValueError("Scores must come from this identical frozen lineup")
        for scores in extras:
            if scores.names not in (scores_a.names, scores_b.names):
                raise ValueError("Repeat scores must use the labels of a compared arm")
        independent = []
        for scores in (scores_a, scores_b, *extras):
            if not any(scores is previous for previous in independent):
                independent.append(scores)

        def summarize(layer: int | None) -> PairedResult:
            positions = [
                index
                for index, trial in enumerate(self.trials)
                if layer is None or trial.topic[0] == layer
            ]
            if not positions:
                return PairedResult(layer, 0, None, None, None, None, "no_trials")

            def run_means(scores: LineupScores) -> list[float]:
                return [
                    float(np.mean([run[index].credit for index in positions]))
                    for run in scores.runs
                ]

            mean_a = float(np.mean(run_means(scores_a)))
            mean_b = float(np.mean(run_means(scores_b)))
            bands = []
            for arm in (scores_a, scores_b):
                repetitions = [
                    value
                    for scores in independent
                    if scores.names == arm.names
                    for value in run_means(scores)
                ]
                if len(repetitions) > 1:
                    bands.append(max(repetitions) - min(repetitions))
            band = max(bands) if bands else None
            delta = mean_b - mean_a
            if band is None:
                conclusion = "repeat_noise_not_measured"
            elif abs(delta) <= band:
                conclusion = "inside_repeat_band"
            else:
                conclusion = "higher_accuracy" if delta > 0 else "lower_accuracy"
            return PairedResult(
                layer, len(positions), mean_a, mean_b, delta, band, conclusion
            )

        return LineupComparison(
            summarize(None),
            tuple(
                summarize(layer)
                for layer in sorted({trial.topic[0] for trial in self.trials})
            ),
            scores_a.call_count,
            scores_b.call_count,
            sum(
                scores.call_count
                for scores in independent
                if scores is not scores_a and scores is not scores_b
            ),
            scores_a.tie_count,
            scores_b.tie_count,
        )
