"""
Audit functionality for Toponymy - Compare intermediate results with LLM outputs
for transparency and debugging purposes.
"""

import pandas as pd
from typing import Optional, List, Dict, Union


def create_cluster_audit_df(
    toponymy_instance,
    layer_index: int = 0,
    include_all_docs: bool = False,
    max_docs_per_cluster: Optional[int] = None,
    original_texts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a DataFrame comparing intermediate results with LLM outputs for a specific layer.

    Each row represents one cluster with columns showing:
    - Cluster metadata (id, size)
    - Intermediate results (keyphrases, exemplars, subtopics)
    - LLM results (final topic name)
    - Optional: document indices and texts for full traceability

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Which layer to audit (default: 0)
        include_all_docs: If True, includes document texts for each cluster (default: False)
        max_docs_per_cluster: Maximum number of documents to show per cluster (default: None)
        original_texts: Original document texts for traceability (required if include_all_docs=True)

    Returns:
        pd.DataFrame with audit information
    """
    layer = toponymy_instance.cluster_layers_[layer_index]

    audit_data = []

    for cluster_idx in range(len(layer.topic_names)):
        # Extract intermediate data
        keyphrases = (
            layer.keyphrases[cluster_idx] if cluster_idx < len(layer.keyphrases) else []
        )
        exemplars = (
            layer.exemplars[cluster_idx] if cluster_idx < len(layer.exemplars) else []
        )

        # Handle subtopics (only available for layers > 0)
        subtopics = []
        if (
            hasattr(layer, "subtopics")
            and layer.subtopics
            and cluster_idx < len(layer.subtopics)
        ):
            subtopics = layer.subtopics[cluster_idx]

        # Count documents in cluster
        num_docs = (layer.cluster_labels == cluster_idx).sum()

        # Get prompt info
        prompt = ""
        if hasattr(layer, "prompts") and cluster_idx < len(layer.prompts):
            prompt = str(layer.prompts[cluster_idx])

        # Get document indices for this cluster
        document_indices = [
            i for i, label in enumerate(layer.cluster_labels) if label == cluster_idx
        ]

        # Build the audit row data
        row_data = {
            # Metadata
            "layer": layer_index,
            "cluster_id": cluster_idx,
            "num_documents": num_docs,
            # Intermediate results
            "top_5_keyphrases": ", ".join(keyphrases[:5]),
            "all_keyphrases": keyphrases,
            "num_keyphrases": len(keyphrases),
            "num_exemplars": len(exemplars),
            "first_exemplar": exemplars[0][:300] + "..." if exemplars else "",
            "subtopics_list": subtopics[:5],
            "subtopics_text": ", ".join(subtopics[:5]),
            # LLM input/output
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "prompt_length": len(prompt),
            "llm_topic_name": (
                layer.topic_names[cluster_idx]
                if cluster_idx < len(layer.topic_names)
                else ""
            ),
            # Document traceability
            "document_indices": document_indices,
        }

        # Add document texts if requested
        if include_all_docs and original_texts:
            doc_texts = [original_texts[i] for i in document_indices]
            if max_docs_per_cluster and len(doc_texts) > max_docs_per_cluster:
                row_data["document_sample"] = doc_texts[:max_docs_per_cluster]
                row_data["total_docs_in_cluster"] = len(doc_texts)
            else:
                row_data["document_texts"] = doc_texts

        audit_data.append(row_data)

    return pd.DataFrame(audit_data)


def create_audit_df(
    toponymy_instance,
    layer_index: Optional[int] = None,
    include_all_docs: bool = False,
    max_docs_per_cluster: Optional[int] = None,
    original_texts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create audit DataFrame for one or all layers.

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Specific layer to audit. If None, audits all layers.
        include_all_docs: If True, includes document texts for each cluster (default: False)
        max_docs_per_cluster: Maximum number of documents to show per cluster (default: None)
        original_texts: Original document texts for traceability (required if include_all_docs=True)

    Returns:
        pd.DataFrame with audit information

    Example:
        >>> from toponymy import Toponymy
        >>> from toponymy.audit import create_audit_df
        >>> # After fitting a Toponymy model
        >>> audit_df = create_audit_df(topic_model, layer_index=0)
        >>> print(audit_df[['cluster_id', 'num_documents', 'llm_topic_name']].head())
           cluster_id  num_documents          llm_topic_name
        0           0            245    NHL Hockey Discussion
        1           1            189  Windows Software Support
        
        >>> # Include all documents
        >>> audit_df = create_audit_df(topic_model, include_all_docs=True, original_texts=documents)
        >>> print(audit_df['document_indices'][0])  # List of document indices for cluster 0
        [2, 15, 27, 45, ...]
    """
    if layer_index is not None:
        return create_cluster_audit_df(
            toponymy_instance,
            layer_index,
            include_all_docs=include_all_docs,
            max_docs_per_cluster=max_docs_per_cluster,
            original_texts=original_texts,
        )

    # Audit all layers
    all_layers = []
    for layer_idx in range(len(toponymy_instance.cluster_layers_)):
        layer_df = create_cluster_audit_df(
            toponymy_instance,
            layer_idx,
            include_all_docs=include_all_docs,
            max_docs_per_cluster=max_docs_per_cluster,
            original_texts=original_texts,
        )
        all_layers.append(layer_df)

    return pd.concat(all_layers, ignore_index=True)


def create_comparison_df(toponymy_instance, layer_index: int = 0) -> pd.DataFrame:
    """
    Create a simplified DataFrame showing intermediate vs LLM results side by side.

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Which layer to analyze

    Returns:
        pd.DataFrame with side-by-side comparison
    """
    audit_df = create_cluster_audit_df(toponymy_instance, layer_index)

    # Select key columns for comparison
    comparison_df = audit_df[
        [
            "cluster_id",
            "num_documents",
            "top_5_keyphrases",
            "num_exemplars",
            "subtopics_text",
            "llm_topic_name",
        ]
    ].copy()

    # Rename for clarity
    comparison_df.columns = [
        "Cluster ID",
        "Document Count",
        "Extracted Keyphrases (Top 5)",
        "Exemplar Count",
        "Child Subtopics",
        "Final LLM Topic Name",
    ]

    return comparison_df


def create_keyphrase_analysis_df(
    toponymy_instance, layer_index: int = 0
) -> pd.DataFrame:
    """
    Analyze how keyphrases relate to final topic names.

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Which layer to analyze

    Returns:
        pd.DataFrame showing keyphrase to topic name mapping
    """
    layer = toponymy_instance.cluster_layers_[layer_index]

    keyphrase_data = []

    for cluster_idx in range(len(layer.topic_names)):
        if cluster_idx < len(layer.keyphrases):
            keyphrases = layer.keyphrases[cluster_idx]
            topic_name = layer.topic_names[cluster_idx]

            for kp in keyphrases[:10]:  # Top 10 keyphrases per cluster
                keyphrase_data.append(
                    {
                        "cluster_id": cluster_idx,
                        "keyphrase": kp,
                        "llm_topic_name": topic_name,
                        "keyphrase_in_topic": kp.lower() in topic_name.lower(),
                    }
                )

    return pd.DataFrame(keyphrase_data)


def create_prompt_analysis_df(toponymy_instance) -> pd.DataFrame:
    """
    Analyze prompt characteristics across all layers and clusters.

    Args:
        toponymy_instance: Fitted Toponymy model

    Returns:
        pd.DataFrame with prompt statistics
    """
    prompt_data = []

    for layer_idx, layer in enumerate(toponymy_instance.cluster_layers_):
        if hasattr(layer, "prompts"):
            for cluster_idx, prompt in enumerate(layer.prompts):
                if cluster_idx < len(layer.topic_names):
                    prompt_str = str(prompt)

                    # Count keyphrases that appear in prompt
                    keyphrases_in_prompt = 0
                    if cluster_idx < len(layer.keyphrases):
                        keyphrases_in_prompt = sum(
                            1
                            for kp in layer.keyphrases[cluster_idx][:10]
                            if kp.lower() in prompt_str.lower()
                        )

                    prompt_data.append(
                        {
                            "layer": layer_idx,
                            "cluster_id": cluster_idx,
                            "prompt_length": len(prompt_str),
                            "num_exemplars_in_prompt": prompt_str.count("EXAMPLE"),
                            "num_keyphrases_in_prompt": keyphrases_in_prompt,
                            "topic_name": layer.topic_names[cluster_idx],
                            "topic_name_length": len(layer.topic_names[cluster_idx]),
                        }
                    )

    return pd.DataFrame(prompt_data)


def create_layer_summary_df(toponymy_instance) -> pd.DataFrame:
    """
    Create a summary DataFrame showing statistics for each layer.

    Args:
        toponymy_instance: Fitted Toponymy model

    Returns:
        pd.DataFrame with layer-level statistics
    """
    summary_data = []

    for layer_idx, layer in enumerate(toponymy_instance.cluster_layers_):
        # Count unique clusters (excluding -1 for unclustered)
        unique_clusters = len(set(layer.cluster_labels)) - (
            1 if -1 in layer.cluster_labels else 0
        )

        # Calculate average cluster size
        cluster_sizes = [
            (layer.cluster_labels == i).sum() for i in range(unique_clusters)
        ]

        summary_data.append(
            {
                "layer": layer_idx,
                "num_clusters": unique_clusters,
                "avg_cluster_size": (
                    sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
                ),
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "unique_topic_names": len(set(layer.topic_names)),
                "duplicate_topic_names": len(layer.topic_names)
                - len(set(layer.topic_names)),
                "has_subtopics": hasattr(layer, "subtopics")
                and layer.subtopics is not None,
            }
        )

    return pd.DataFrame(summary_data)


def export_audit_excel(toponymy_instance, filename: str = "toponymy_audit.xlsx"):
    """
    Export comprehensive audit data to an Excel file with multiple sheets.

    Args:
        toponymy_instance: Fitted Toponymy model
        filename: Output Excel filename
    """
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Summary sheet
        summary_df = create_layer_summary_df(toponymy_instance)
        summary_df.to_excel(writer, sheet_name="Layer Summary", index=False)

        # Full audit data
        full_audit_df = create_audit_df(toponymy_instance)
        full_audit_df.to_excel(writer, sheet_name="Full Audit", index=False)

        # Comparison for each layer
        for layer_idx in range(len(toponymy_instance.cluster_layers_)):
            comparison_df = create_comparison_df(toponymy_instance, layer_idx)
            comparison_df.to_excel(
                writer, sheet_name=f"Layer {layer_idx} Comparison", index=False
            )

            # Add keyphrase analysis for first few layers
            if layer_idx < 3:
                kp_df = create_keyphrase_analysis_df(toponymy_instance, layer_idx)
                kp_df.to_excel(
                    writer, sheet_name=f"Layer {layer_idx} Keyphrases", index=False
                )

        # Prompt analysis
        prompt_df = create_prompt_analysis_df(toponymy_instance)
        prompt_df.to_excel(writer, sheet_name="Prompt Analysis", index=False)

    print(f"Audit data exported to {filename}")


def get_cluster_documents(
    toponymy_instance,
    layer_index: int,
    cluster_id: int,
    original_texts: List[str],
    max_docs: Optional[int] = None,
) -> Dict[str, Union[List[int], List[str]]]:
    """
    Get all documents belonging to a specific cluster.

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Layer containing the cluster
        cluster_id: ID of the cluster
        original_texts: Original document texts
        max_docs: Maximum number of documents to return (default: None returns all)

    Returns:
        Dictionary containing:
        - 'indices': List of document indices in the cluster
        - 'texts': List of document texts
        - 'total_count': Total number of documents in cluster

    Example:
        >>> cluster_docs = get_cluster_documents(model, 0, 5, documents)
        >>> print(f"Cluster 5 has {cluster_docs['total_count']} documents")
        >>> for idx, text in zip(cluster_docs['indices'][:3], cluster_docs['texts'][:3]):
        ...     print(f"Doc {idx}: {text[:50]}...")
    """
    layer = toponymy_instance.cluster_layers_[layer_index]
    
    # Get all document indices for this cluster
    doc_indices = [
        i for i, label in enumerate(layer.cluster_labels) if label == cluster_id
    ]
    
    # Apply max_docs limit if specified
    if max_docs and len(doc_indices) > max_docs:
        limited_indices = doc_indices[:max_docs]
        doc_texts = [original_texts[i] for i in limited_indices]
        return {
            'indices': limited_indices,
            'texts': doc_texts,
            'total_count': len(doc_indices)
        }
    
    doc_texts = [original_texts[i] for i in doc_indices]
    return {
        'indices': doc_indices,
        'texts': doc_texts,
        'total_count': len(doc_indices)
    }


def get_cluster_details(toponymy_instance, layer_index: int, cluster_id: int) -> Dict:
    """
    Get detailed audit information for a specific cluster.

    Args:
        toponymy_instance: Fitted Toponymy model
        layer_index: Layer containing the cluster
        cluster_id: ID of the cluster to inspect

    Returns:
        Dictionary with detailed cluster information
    """
    layer = toponymy_instance.cluster_layers_[layer_index]

    details = {
        "layer": layer_index,
        "cluster_id": cluster_id,
        "num_documents": (layer.cluster_labels == cluster_id).sum(),
        "topic_name": (
            layer.topic_names[cluster_id]
            if cluster_id < len(layer.topic_names)
            else "N/A"
        ),
    }

    # Add intermediate data
    if cluster_id < len(layer.keyphrases):
        details["keyphrases"] = layer.keyphrases[cluster_id]

    if cluster_id < len(layer.exemplars):
        details["exemplars"] = layer.exemplars[cluster_id]
        details["exemplar_indices"] = layer.exemplar_indices[cluster_id]

    if (
        hasattr(layer, "subtopics")
        and layer.subtopics
        and cluster_id < len(layer.subtopics)
    ):
        details["subtopics"] = layer.subtopics[cluster_id]

    if hasattr(layer, "prompts") and cluster_id < len(layer.prompts):
        details["prompt"] = layer.prompts[cluster_id]

    return details
