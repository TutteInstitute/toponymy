"""Semantic selection of child-topic evidence.

Legacy labels index output slots and the supplied vocabulary. Cosine methods
preserve centered directions; dictionary learning keeps raw magnitudes.
"""

import numpy as np
import scipy.sparse
from typing import List, Optional
from sklearn.decomposition import DictionaryLearning
from vectorizers.transformers import InformationWeightTransformer
from toponymy.utility_functions import diversify_max_alpha as diversify
from toponymy.utility_functions import (
    _mean_vector,
    _center_vectors,
    _normalize_rows,
    distance_to_vector,
)
from toponymy.exemplar_texts import (
    FacilityLocationSelection,
    SaturatedCoverageSelection,
)
from toponymy._utils import handle_verbose_params
from toponymy.embedding_wrappers import TextEmbedderProtocol
from tqdm.auto import tqdm


def _subtopic_groups(parent_labels, child_labels, n_names):
    parents, children = np.asarray(parent_labels), np.asarray(child_labels)
    if parents.ndim != 1 or children.shape != parents.shape:
        raise ValueError("Parent and subtopic labels must be aligned vectors")
    for labels in (parents, children):
        if labels.size and (labels.dtype.kind not in "iu" or np.any(labels < -1)):
            raise ValueError("Labels must be integers with only -1 for noise")
    if np.any(children >= n_names):
        raise ValueError("Subtopic labels must index the supplied vocabulary")
    return [
        np.unique(children[(parents == parent) & (children >= 0)])
        for parent in range((int(parents.max()) if parents.size else -1) + 1)
    ]


def _parent_classes(groups, n_names):
    classes = np.full(n_names, -1, dtype=np.int64)
    for parent, indices in enumerate(groups):
        if np.any(classes[indices] >= 0):
            raise ValueError(
                "Information weighting cannot assign a subtopic to multiple parents"
            )
        classes[indices] = parent
    return classes


def _selection_vectors(names, vectors, embedding_model, groups, n_subtopics, alpha=0):
    if (
        isinstance(n_subtopics, (bool, np.bool_))
        or not isinstance(n_subtopics, (int, np.integer))
        or n_subtopics < 1
    ):
        raise ValueError("n_subtopics must be a positive integer")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("diversify_alpha must be finite and nonnegative")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("Subtopic names must be strings")
    if vectors is None:
        if not any(len(indices) for indices in groups):
            return None
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_vectors or embedding_model must be provided"
            )
        vectors = subtopic_embeddings(names, embedding_model)
    vectors = np.asarray(vectors)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(names)
        or (len(names) and not vectors.shape[1])
        or vectors.dtype.kind not in "fiu"
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(
            "Subtopic vectors must be a finite real matrix aligned with names"
        )
    return vectors


def _progress(groups, description, verbose, show_progress_bar):
    show, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )
    return tqdm(groups, desc=description, disable=not show, leave=False, unit="cluster")


def _central_indices(
    indices, vectors, n_subtopics, alpha, center, all_candidates=False
):
    if not len(indices):
        return []
    query = _center_vectors(_mean_vector(vectors[indices])[None, :], center)[0]
    candidates = np.arange(len(vectors)) if all_candidates else indices
    candidate_vectors = _center_vectors(vectors[candidates], center)
    distances = distance_to_vector(query, candidate_vectors)
    order = np.argsort(distances, kind="stable")[: n_subtopics**2]
    candidates = candidates[order]
    chosen = diversify(query, candidate_vectors[order], n_subtopics, max_alpha=alpha)
    return candidates[chosen[:n_subtopics]].tolist()


def _central_selections(groups, vectors, n_subtopics, alpha, all_candidates=False):
    center = 0.0
    if vectors is not None and len(vectors):
        if not all_candidates:
            center = _mean_vector(vectors)
    return [
        _central_indices(indices, vectors, n_subtopics, alpha, center, all_candidates)
        for indices in groups
    ]


def _submodular_selections(groups, vectors, n_subtopics, method):
    if method == "facility_location":
        selector = FacilityLocationSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    elif method == "saturated_coverage":
        selector = SaturatedCoverageSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    else:
        raise ValueError(
            "submodular_function must be facility_location or saturated_coverage"
        )
    if vectors is not None and len(vectors):
        vectors = _normalize_rows(_center_vectors(vectors, _mean_vector(vectors)))
    result = []
    for indices in groups:
        if len(indices) >= n_subtopics:
            _, chosen = selector.fit_transform(vectors[indices], y=indices)
        else:
            chosen = indices
        result.append(chosen.tolist())
    return result


def _information_selections(
    groups,
    classes,
    vectors,
    n_subtopics,
    alpha,
    prior_strength=0.1,
    weight_power=2.0,
    n_dictionary_vectors=512,
    coding_transform_alpha=0.1,
    n_jobs=-1,
):
    if not any(len(indices) for indices in groups):
        return [[] for _ in groups]
    center = _mean_vector(vectors)
    if len(vectors) < 2 or len(np.unique(classes)) < 2 or not np.any(vectors):
        return [
            _central_indices(ids, vectors, n_subtopics, alpha, center) for ids in groups
        ]
    # The L1 objective depends on magnitude: never normalize dictionary inputs.
    raw = np.asarray(vectors, dtype=np.float64)
    if np.max(np.abs(raw)) > np.sqrt(np.finfo(np.float64).max / raw.size):
        raise ValueError(
            "Subtopic magnitudes exceed the numerical range of dictionary learning"
        )
    coder = DictionaryLearning(
        n_components=min(n_dictionary_vectors, len(vectors) // 2),
        positive_code=True,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        n_jobs=n_jobs,
        max_iter=100_000,
        transform_alpha=coding_transform_alpha,
        transform_max_iter=100_000,
    ).fit(raw)
    coding = scipy.sparse.csr_array(coder.transform(raw))
    if not np.isfinite(coding.data).all() or np.any(coding.data < 0):
        raise ValueError("Dictionary learning returned invalid nonnegative codes")
    if not coding.nnz or not np.any(coding.data):
        return [
            _central_indices(ids, vectors, n_subtopics, alpha, center) for ids in groups
        ]
    weighted = InformationWeightTransformer(
        prior_strength=prior_strength, weight_power=weight_power
    ).fit_transform(coding, y=classes)
    scores = np.asarray(weighted.sum(axis=1)).ravel()
    result = []
    for indices in groups:
        if not len(indices):
            result.append([])
            continue
        weights = scores[indices]
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.max() <= 0:
            result.append(
                _central_indices(indices, vectors, n_subtopics, alpha, center)
            )
            continue
        weights = weights / weights.max()
        query = _mean_vector(vectors[indices], weights=weights)
        order = np.argsort(-weights, kind="stable")[: n_subtopics**2]
        candidates = indices[order]
        chosen = diversify(
            query,
            np.asarray(vectors[candidates], dtype=np.float64),
            n_subtopics,
            max_alpha=alpha,
        )
        result.append(candidates[chosen[:n_subtopics]].tolist())
    return result


def _summary_fields(names, summaries, explanations):
    if any(
        len(values) != len(names) or not all(isinstance(value, str) for value in values)
        for values in (summaries, explanations)
    ):
        raise ValueError(
            "Summaries and explanations must be strings aligned with names"
        )


def _render_summaries(selections, names, summaries, explanations):
    return [
        "\n--\n".join(
            "\n".join((names[i], summaries[i], explanations[i])) for i in indices
        )
        for indices in selections
    ]


def subtopic_embeddings(
    subtopics: List[str], embedding_model: TextEmbedderProtocol
) -> np.ndarray:
    return embedding_model.encode(subtopics)


def central_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    vectors = _selection_vectors(
        subtopics,
        subtopic_vectors,
        embedding_model,
        groups,
        n_subtopics,
        diversify_alpha,
    )
    groups = _progress(
        groups, "Selecting central subtopics", verbose, show_progress_bar
    )
    selections = _central_selections(
        groups, vectors, n_subtopics, diversify_alpha, all_candidates=False
    )
    return [[subtopics[i] for i in indices] for indices in selections]


def submodular_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    submodular_function: str = "facility_location",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    vectors = _selection_vectors(
        subtopics, subtopic_vectors, embedding_model, groups, n_subtopics, 0
    )
    groups = _progress(
        groups, "Selecting submodular subtopics", verbose, show_progress_bar
    )
    selections = _submodular_selections(
        groups, vectors, n_subtopics, submodular_function
    )
    return [[subtopics[i] for i in indices] for indices in selections]


def central_subtopics_from_all_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    vectors = _selection_vectors(
        subtopics,
        subtopic_vectors,
        embedding_model,
        groups,
        n_subtopics,
        diversify_alpha,
    )
    groups = _progress(
        groups, "Selecting central subtopics", verbose, show_progress_bar
    )
    selections = _central_selections(
        groups, vectors, n_subtopics, diversify_alpha, all_candidates=True
    )
    return [[subtopics[i] for i in indices] for indices in selections]


def information_weighted_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    n_dictionary_vectors: int = 512,
    coding_transform_alpha: float = 0.1,
    n_jobs=-1,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    classes = _parent_classes(groups, len(subtopics))
    vectors = _selection_vectors(
        subtopics,
        subtopic_vectors,
        embedding_model,
        groups,
        n_subtopics,
        diversify_alpha,
    )
    selections = _information_selections(
        groups,
        classes,
        vectors,
        n_subtopics,
        diversify_alpha,
        prior_strength,
        weight_power,
        n_dictionary_vectors,
        coding_transform_alpha,
        n_jobs,
    )
    selections = _progress(
        selections, "Selecting informative subtopics", verbose, show_progress_bar
    )
    return [[subtopics[i] for i in indices] for indices in selections]


def central_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[str]:
    _summary_fields(subtopics, summaries, explanations)
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    vectors = _selection_vectors(
        subtopics,
        subtopic_vectors,
        embedding_model,
        groups,
        n_subtopics,
        diversify_alpha,
    )
    groups = _progress(
        groups, "Selecting central subtopics", verbose, show_progress_bar
    )
    selections = _central_selections(
        groups, vectors, n_subtopics, diversify_alpha, all_candidates=False
    )
    return _render_summaries(selections, subtopics, summaries, explanations)


def submodular_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    submodular_function: str = "facility_location",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[str]:
    _summary_fields(subtopics, summaries, explanations)
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    vectors = _selection_vectors(
        subtopics, subtopic_vectors, embedding_model, groups, n_subtopics, 0
    )
    groups = _progress(
        groups, "Selecting submodular subtopics", verbose, show_progress_bar
    )
    selections = _submodular_selections(
        groups, vectors, n_subtopics, submodular_function
    )
    return _render_summaries(selections, subtopics, summaries, explanations)


def information_weighted_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    n_dictionary_vectors: int = 512,
    coding_transform_alpha: float = 0.1,
    n_jobs=-1,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[str]:
    _summary_fields(subtopics, summaries, explanations)
    groups = _subtopic_groups(
        cluster_label_vector, subtopic_label_vector, len(subtopics)
    )
    classes = _parent_classes(groups, len(subtopics))
    vectors = _selection_vectors(
        subtopics,
        subtopic_vectors,
        embedding_model,
        groups,
        n_subtopics,
        diversify_alpha,
    )
    selections = _information_selections(
        groups,
        classes,
        vectors,
        n_subtopics,
        diversify_alpha,
        prior_strength,
        weight_power,
        n_dictionary_vectors,
        coding_transform_alpha,
        n_jobs,
    )
    selections = _progress(
        selections, "Selecting informative subtopics", verbose, show_progress_bar
    )
    return _render_summaries(selections, subtopics, summaries, explanations)
