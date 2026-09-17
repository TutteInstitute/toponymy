"""Inspect topic features, prompts and naming results without provider calls."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Optional, List

import pandas as pd

_AUDIT_COLUMNS = [
    "layer",
    "cluster_id",
    "num_documents",
    "top_5_keyphrases",
    "all_keyphrases",
    "num_keyphrases",
    "num_exemplars",
    "first_exemplar",
    "subtopics_list",
    "subtopics_text",
    "prompt_preview",
    "prompt_length",
    "llm_topic_name",
    "document_indices",
]


def _result(instance):
    result = getattr(instance, "topic_model_", instance)
    if not isinstance(getattr(result, "topics", None), Mapping):
        raise ValueError(
            "Audit requires a fitted Toponymy or TopicModel with topic metadata"
        )
    return result


def _layer_topics(instance, layer_index):
    result = _result(instance)
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < len(result.cluster_layers)
    ):
        raise ValueError("layer_index is outside the fitted model")
    return [
        topic
        for (layer, _), topic in sorted(result.topics.items())
        if layer == layer_index
    ]


def _limit(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None")


def _document_texts(instance, original_texts):
    if original_texts is None or len(original_texts) != len(
        _result(instance).embedding_vectors
    ):
        raise ValueError("original_texts must contain one entry per fitted document")


def _prompt_text(prompt):
    if prompt is None:
        return ""
    if isinstance(prompt, Mapping):
        if prompt.get("combined") is not None:
            return prompt["combined"]
        return "\n\n".join(
            prompt.get(key, "") for key in ("system", "user") if prompt.get(key)
        )
    if hasattr(prompt, "system") and hasattr(prompt, "user"):
        if getattr(prompt, "combined", None) is not None:
            return prompt.combined
        return "\n\n".join(part for part in (prompt.system, prompt.user) if part)
    return str(prompt)


def _subtopics(topic):
    values = topic.features.get("cluster_subtopics", [])
    if isinstance(values, Mapping):
        return [name for group in values.values() for name in group]
    return list(values)


def create_cluster_audit_df(
    toponymy_instance,
    layer_index: int = 0,
    include_all_docs: bool = False,
    max_docs_per_cluster: Optional[int] = None,
    original_texts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return one row per original cluster ID from learned topic state.

    Original texts must align to all fitted documents when requested. A zero
    sample limit is valid. Returned feature lists are independent copies.
    """
    topics = _layer_topics(toponymy_instance, layer_index)
    _limit(max_docs_per_cluster, "max_docs_per_cluster")
    if include_all_docs:
        _document_texts(toponymy_instance, original_texts)
    rows = []
    for topic in topics:
        keyphrases = deepcopy(topic.features.get("cluster_keywords", []))
        exemplars = topic.features.get("cluster_sentences", [])
        subtopics = _subtopics(topic)
        prompt = _prompt_text(topic.prompt)
        indices = [int(index) for index in topic.members]
        first = str(exemplars[0]) if exemplars else ""
        row = {
            "layer": layer_index,
            "cluster_id": topic.label,
            "num_documents": len(indices),
            "top_5_keyphrases": ", ".join(keyphrases[:5]),
            "all_keyphrases": keyphrases,
            "num_keyphrases": len(keyphrases),
            "num_exemplars": len(exemplars),
            "first_exemplar": first[:300] + "..." if len(first) > 300 else first,
            "subtopics_list": subtopics[:5],
            "subtopics_text": ", ".join(subtopics[:5]),
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "prompt_length": len(prompt),
            "llm_topic_name": topic.name or "",
            "document_indices": indices,
        }
        if include_all_docs:
            texts = [original_texts[index] for index in indices]
            if max_docs_per_cluster is not None and len(texts) > max_docs_per_cluster:
                row["document_sample"] = texts[:max_docs_per_cluster]
                row["total_docs_in_cluster"] = len(texts)
            else:
                row["document_texts"] = texts
        rows.append(row)
    columns = list(_AUDIT_COLUMNS)
    for optional in ("document_sample", "total_docs_in_cluster", "document_texts"):
        if any(optional in row for row in rows):
            columns.append(optional)
    return pd.DataFrame(rows, columns=columns)


def create_audit_df(
    toponymy_instance,
    layer_index=None,
    include_all_docs=False,
    max_docs_per_cluster=None,
    original_texts=None,
):
    """Inspect one layer or all layers, including a model with no topic layers."""
    if layer_index is not None:
        return create_cluster_audit_df(
            toponymy_instance,
            layer_index,
            include_all_docs,
            max_docs_per_cluster,
            original_texts,
        )
    result = _result(toponymy_instance)
    _limit(max_docs_per_cluster, "max_docs_per_cluster")
    if include_all_docs:
        _document_texts(toponymy_instance, original_texts)
    frames = [
        create_cluster_audit_df(
            toponymy_instance,
            layer,
            include_all_docs,
            max_docs_per_cluster,
            original_texts,
        )
        for layer in range(len(result.cluster_layers))
    ]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_AUDIT_COLUMNS)
    )


def create_comparison_df(toponymy_instance, layer_index=0):
    """Show extracted evidence and final topic names side by side."""
    frame = create_cluster_audit_df(toponymy_instance, layer_index)[
        [
            "cluster_id",
            "num_documents",
            "top_5_keyphrases",
            "num_exemplars",
            "subtopics_text",
            "llm_topic_name",
        ]
    ].copy()
    frame.columns = [
        "Cluster ID",
        "Document Count",
        "Extracted Keyphrases (Top 5)",
        "Exemplar Count",
        "Child Subtopics",
        "Final LLM Topic Name",
    ]
    return frame


def create_keyphrase_analysis_df(toponymy_instance, layer_index=0):
    """Match each of the first ten keyphrases against its generated topic name."""
    rows = []
    for topic in _layer_topics(toponymy_instance, layer_index):
        name = topic.name or ""
        for phrase in topic.features.get("cluster_keywords", [])[:10]:
            rows.append(
                {
                    "cluster_id": topic.label,
                    "keyphrase": phrase,
                    "llm_topic_name": name,
                    "keyphrase_in_topic": phrase.lower() in name.lower(),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["cluster_id", "keyphrase", "llm_topic_name", "keyphrase_in_topic"],
    )


def create_prompt_analysis_df(toponymy_instance):
    """Describe stored prompts without invoking or selecting a provider."""
    rows = []
    for (layer, label), topic in sorted(_result(toponymy_instance).topics.items()):
        if topic.prompt is None:
            continue
        prompt = _prompt_text(topic.prompt)
        name = topic.name or ""
        rows.append(
            {
                "layer": layer,
                "cluster_id": label,
                "prompt_length": len(prompt),
                "num_exemplars_in_prompt": sum(
                    str(text) in prompt
                    for text in topic.features.get("cluster_sentences", [])
                ),
                "num_keyphrases_in_prompt": sum(
                    phrase.lower() in prompt.lower()
                    for phrase in topic.features.get("cluster_keywords", [])[:10]
                ),
                "topic_name": name,
                "topic_name_length": len(name),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "layer",
            "cluster_id",
            "prompt_length",
            "num_exemplars_in_prompt",
            "num_keyphrases_in_prompt",
            "topic_name",
            "topic_name_length",
        ],
    )


def create_layer_summary_df(toponymy_instance):
    """Summarize real topic membership, excluding noise and absent IDs."""
    rows = []
    for layer in range(len(_result(toponymy_instance).cluster_layers)):
        topics = _layer_topics(toponymy_instance, layer)
        sizes = [len(topic.members) for topic in topics]
        names = [topic.name for topic in topics if topic.name is not None]
        rows.append(
            {
                "layer": layer,
                "num_clusters": len(topics),
                "avg_cluster_size": sum(sizes) / len(sizes) if sizes else 0,
                "min_cluster_size": min(sizes, default=0),
                "max_cluster_size": max(sizes, default=0),
                "unique_topic_names": len(set(names)),
                "duplicate_topic_names": len(names) - len(set(names)),
                "has_subtopics": any(bool(_subtopics(topic)) for topic in topics),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "layer",
            "num_clusters",
            "avg_cluster_size",
            "min_cluster_size",
            "max_cluster_size",
            "unique_topic_names",
            "duplicate_topic_names",
            "has_subtopics",
        ],
    )


def export_audit_excel(toponymy_instance, filename="toponymy_audit.xlsx"):
    """Write inspection tables using the optional openpyxl Excel engine."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        create_layer_summary_df(toponymy_instance).to_excel(
            writer, sheet_name="Layer Summary", index=False
        )
        create_audit_df(toponymy_instance).to_excel(
            writer, sheet_name="Full Audit", index=False
        )
        for layer in range(len(_result(toponymy_instance).cluster_layers)):
            create_comparison_df(toponymy_instance, layer).to_excel(
                writer, sheet_name=f"Layer {layer} Comparison", index=False
            )
            if layer < 3:
                create_keyphrase_analysis_df(toponymy_instance, layer).to_excel(
                    writer, sheet_name=f"Layer {layer} Keyphrases", index=False
                )
        create_prompt_analysis_df(toponymy_instance).to_excel(
            writer, sheet_name="Prompt Analysis", index=False
        )
    print(f"Audit data exported to {filename}")


def get_cluster_documents(
    toponymy_instance, layer_index, cluster_id, original_texts, max_docs=None
):
    """Return document indices and texts for an original cluster ID."""
    _layer_topics(toponymy_instance, layer_index)
    _document_texts(toponymy_instance, original_texts)
    _limit(max_docs, "max_docs")
    topic = _result(toponymy_instance).topics[(layer_index, cluster_id)]
    indices = [int(index) for index in topic.members]
    selected = indices if max_docs is None else indices[:max_docs]
    return {
        "indices": selected,
        "texts": [original_texts[index] for index in selected],
        "total_count": len(indices),
    }


def get_cluster_details(toponymy_instance, layer_index, cluster_id):
    """Return copied intermediate data for one original cluster ID."""
    _layer_topics(toponymy_instance, layer_index)
    topic = _result(toponymy_instance).topics[(layer_index, cluster_id)]
    details = {
        "layer": layer_index,
        "cluster_id": cluster_id,
        "num_documents": len(topic.members),
        "topic_name": topic.name or "",
    }
    for source, output in (
        ("cluster_keywords", "keyphrases"),
        ("cluster_sentences", "exemplars"),
        ("exemplar_indices", "exemplar_indices"),
        ("cluster_subtopics", "subtopics"),
    ):
        if source in topic.features:
            details[output] = deepcopy(topic.features[source])
    if topic.prompt is not None:
        details["prompt"] = deepcopy(topic.prompt)
    return details
