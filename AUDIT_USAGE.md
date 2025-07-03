# Audit Functionality for Toponymy

The audit module provides transparency into how Toponymy generates topic names by comparing intermediate clustering results with final LLM-generated names.

## Installation

The audit functionality is included in the main Toponymy package. No additional installation required.

## Basic Usage

```python
from toponymy import Toponymy
from toponymy.audit import create_audit_df, create_comparison_df, export_audit_excel

# After fitting your Toponymy model
topic_model = Toponymy(...)
topic_model.fit(texts, vectors, map)

# Create audit comparison
comparison_df = create_comparison_df(topic_model, layer_index=0)
print(comparison_df.head())
```

## Available Functions

### 1. `create_audit_df()`
Get comprehensive audit data including all intermediate results:

```python
audit_df = create_audit_df(topic_model)  # All layers
audit_df = create_audit_df(topic_model, layer_index=0)  # Specific layer
```

### 2. `create_comparison_df()`
Simple side-by-side comparison of intermediate vs final results:

```python
comparison_df = create_comparison_df(topic_model, layer_index=0)
```

Output columns:
- **Cluster ID**: Cluster identifier
- **Document Count**: Number of documents in cluster
- **Extracted Keyphrases (Top 5)**: Top keyphrases from the cluster
- **Exemplar Count**: Number of example documents used
- **Child Subtopics**: Subtopics from lower layers (if applicable)
- **Final LLM Topic Name**: The LLM-generated topic name

### 3. `create_layer_summary_df()`
Get statistics for each hierarchical layer:

```python
summary_df = create_layer_summary_df(topic_model)
```

### 4. `create_keyphrase_analysis_df()`
Analyze how keyphrases relate to topic names:

```python
keyphrase_df = create_keyphrase_analysis_df(topic_model, layer_index=0)
```

### 5. `export_audit_excel()`
Export all audit data to an Excel file:

```python
export_audit_excel(topic_model, "audit_results.xlsx")
```

## Use Cases

### Finding Mismatches
Identify clusters where keyphrases don't appear in the topic name:

```python
audit_df = create_audit_df(topic_model, layer_index=0)
for _, row in audit_df.iterrows():
    keyphrases = row['top_5_keyphrases'].lower()
    topic_name = row['llm_topic_name'].lower()
    if not any(kp in topic_name for kp in keyphrases.split(', ')):
        print(f"Mismatch - Cluster {row['cluster_id']}:")
        print(f"  Keyphrases: {row['top_5_keyphrases']}")
        print(f"  Topic: {row['llm_topic_name']}")
```

### Analyzing Topic Quality
Compare small vs large clusters:

```python
audit_df = create_audit_df(topic_model, layer_index=0)
small_clusters = audit_df[audit_df['num_documents'] < 20]
large_clusters = audit_df[audit_df['num_documents'] > 100]

print(f"Small clusters: {len(small_clusters)}")
print(f"Large clusters: {len(large_clusters)}")
```

## Example Output

```
   Cluster ID  Document Count                    Extracted Keyphrases (Top 5)  Final LLM Topic Name
0           0             342  machine learning, neural, AI, model, training  Machine Learning Research
1           1             278  climate, warming, carbon, renewable, energy    Climate Change & Energy
2           2             189  covid, pandemic, vaccine, health, virus        COVID-19 Response
```

## Requirements

- Pandas for DataFrame operations
- Openpyxl for Excel export (optional)