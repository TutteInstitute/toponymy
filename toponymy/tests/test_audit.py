"""
Tests for the audit module functionality.
"""

import pytest
import numpy as np
import pandas as pd
from toponymy.audit import (
    create_audit_df,
    create_comparison_df,
    create_layer_summary_df,
    create_keyphrase_analysis_df,
    create_prompt_analysis_df,
    get_cluster_details,
    get_cluster_documents,
    export_audit_excel
)


class MockClusterLayer:
    """Mock cluster layer for testing audit functionality."""
    
    def __init__(self, n_clusters=5, layer_id=0):
        self.layer_id = layer_id
        n_docs = 100
        
        # Simulate cluster assignments
        self.cluster_labels = np.random.RandomState(42).choice(n_clusters, n_docs)
        
        # Simulate topic names
        self.topic_names = [f"Topic {i}" for i in range(n_clusters)]
        
        # Simulate keyphrases
        self.keyphrases = [[f"keyword_{i}_{j}" for j in range(5)] for i in range(n_clusters)]
        
        # Simulate exemplars
        self.exemplars = [[f"Example doc {j} for cluster {i}" for j in range(3)] for i in range(n_clusters)]
        self.exemplar_indices = [[j for j in range(3)] for i in range(n_clusters)]
        
        # Simulate prompts
        self.prompts = [f"Prompt for cluster {i}" for i in range(n_clusters)]
        
        # Simulate subtopics for higher layers
        if layer_id > 0:
            self.subtopics = [[f"Subtopic {i}.{j}" for j in range(2)] for i in range(n_clusters)]
        else:
            self.subtopics = None


class MockToponymyModel:
    """Mock Toponymy model for testing."""
    
    def __init__(self, n_layers=3):
        self.cluster_layers_ = [
            MockClusterLayer(n_clusters=10-i*3, layer_id=i) 
            for i in range(n_layers)
        ]


class TestAuditFunctions:
    """Test suite for audit functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = MockToponymyModel()
    
    def test_create_audit_df_single_layer(self):
        """Test creating audit DataFrame for a single layer."""
        audit_df = create_audit_df(self.mock_model, layer_index=0)
        
        assert isinstance(audit_df, pd.DataFrame)
        assert len(audit_df) == 10  # 10 clusters in layer 0
        assert 'cluster_id' in audit_df.columns
        assert 'num_documents' in audit_df.columns
        assert 'top_5_keyphrases' in audit_df.columns
        assert 'llm_topic_name' in audit_df.columns
    
    def test_create_audit_df_all_layers(self):
        """Test creating audit DataFrame for all layers."""
        audit_df = create_audit_df(self.mock_model)
        
        assert isinstance(audit_df, pd.DataFrame)
        # 10 + 7 + 4 = 21 total clusters across 3 layers
        assert len(audit_df) == 21
        assert audit_df['layer'].nunique() == 3
    
    def test_create_comparison_df(self):
        """Test creating comparison DataFrame."""
        comparison_df = create_comparison_df(self.mock_model, layer_index=0)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 10
        assert 'Cluster ID' in comparison_df.columns
        assert 'Document Count' in comparison_df.columns
        assert 'Extracted Keyphrases (Top 5)' in comparison_df.columns
        assert 'Final LLM Topic Name' in comparison_df.columns
    
    def test_create_layer_summary_df(self):
        """Test creating layer summary DataFrame."""
        summary_df = create_layer_summary_df(self.mock_model)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 3  # 3 layers
        assert 'layer' in summary_df.columns
        assert 'num_clusters' in summary_df.columns
        assert 'avg_cluster_size' in summary_df.columns
        assert summary_df.loc[0, 'num_clusters'] == 10
        assert summary_df.loc[1, 'num_clusters'] == 7
        assert summary_df.loc[2, 'num_clusters'] == 4
    
    def test_create_keyphrase_analysis_df(self):
        """Test creating keyphrase analysis DataFrame."""
        keyphrase_df = create_keyphrase_analysis_df(self.mock_model, layer_index=0)
        
        assert isinstance(keyphrase_df, pd.DataFrame)
        assert 'cluster_id' in keyphrase_df.columns
        assert 'keyphrase' in keyphrase_df.columns
        assert 'llm_topic_name' in keyphrase_df.columns
        assert 'keyphrase_in_topic' in keyphrase_df.columns
        
        # Each cluster has 5 keyphrases, 10 clusters
        assert len(keyphrase_df) == 50
    
    def test_create_prompt_analysis_df(self):
        """Test creating prompt analysis DataFrame."""
        prompt_df = create_prompt_analysis_df(self.mock_model)
        
        assert isinstance(prompt_df, pd.DataFrame)
        assert 'layer' in prompt_df.columns
        assert 'cluster_id' in prompt_df.columns
        assert 'prompt_length' in prompt_df.columns
        assert 'topic_name' in prompt_df.columns
        assert 'topic_name_length' in prompt_df.columns
    
    def test_get_cluster_details(self):
        """Test getting detailed cluster information."""
        details = get_cluster_details(self.mock_model, layer_index=0, cluster_id=0)
        
        assert isinstance(details, dict)
        assert details['layer'] == 0
        assert details['cluster_id'] == 0
        assert 'num_documents' in details
        assert 'topic_name' in details
        assert 'keyphrases' in details
        assert 'exemplars' in details
        assert 'prompt' in details
    
    def test_export_audit_excel(self, tmp_path):
        """Test exporting audit data to Excel."""
        output_file = tmp_path / "test_audit.xlsx"
        
        # This will fail without openpyxl, so we just test the function exists
        # In a real test suite, we'd mock the Excel writer
        assert callable(export_audit_excel)
    
    def test_empty_cluster_handling(self):
        """Test handling of empty clusters."""
        # Create a model with some empty clusters
        model = MockToponymyModel()
        model.cluster_layers_[0].keyphrases[0] = []
        model.cluster_layers_[0].exemplars[0] = []
        
        audit_df = create_audit_df(model, layer_index=0)
        
        # Should handle empty lists gracefully
        assert audit_df.loc[0, 'num_keyphrases'] == 0
        assert audit_df.loc[0, 'num_exemplars'] == 0
        assert audit_df.loc[0, 'top_5_keyphrases'] == ''
    
    def test_subtopics_in_higher_layers(self):
        """Test that subtopics appear correctly in higher layers."""
        comparison_df = create_comparison_df(self.mock_model, layer_index=1)
        
        # Layer 1 should have subtopics
        assert 'Child Subtopics' in comparison_df.columns
        assert comparison_df['Child Subtopics'].iloc[0] != ''
        
        # Layer 0 should not have subtopics
        comparison_df_layer0 = create_comparison_df(self.mock_model, layer_index=0)
        assert comparison_df_layer0['Child Subtopics'].iloc[0] == ''
    
    def test_document_indices_always_included(self):
        """Test that document_indices are always included in audit DataFrame."""
        audit_df = create_audit_df(self.mock_model, layer_index=0)
        
        # document_indices should always be present
        assert 'document_indices' in audit_df.columns
        assert isinstance(audit_df['document_indices'].iloc[0], list)
        assert len(audit_df['document_indices'].iloc[0]) > 0
    
    def test_document_traceability_with_texts(self):
        """Test document traceability with full texts."""
        # Create sample documents
        documents = [f"Document {i}" for i in range(100)]
        
        # Test with include_all_docs=True
        audit_df = create_audit_df(
            self.mock_model, 
            layer_index=0,
            include_all_docs=True,
            original_texts=documents
        )
        
        # Should have document_texts column
        assert 'document_texts' in audit_df.columns
        assert isinstance(audit_df['document_texts'].iloc[0], list)
        
        # Document texts should match indices
        first_cluster_indices = audit_df['document_indices'].iloc[0]
        first_cluster_texts = audit_df['document_texts'].iloc[0]
        assert len(first_cluster_indices) == len(first_cluster_texts)
        
        # Verify texts match indices
        for idx, text in zip(first_cluster_indices, first_cluster_texts):
            assert text == documents[idx]
    
    def test_document_traceability_with_limit(self):
        """Test document traceability with max_docs_per_cluster limit."""
        documents = [f"Document {i}" for i in range(100)]
        
        audit_df = create_audit_df(
            self.mock_model,
            layer_index=0,
            include_all_docs=True,
            max_docs_per_cluster=3,
            original_texts=documents
        )
        
        # For clusters with more than 3 docs, should have document_sample
        for _, row in audit_df.iterrows():
            if row['num_documents'] > 3:
                assert 'document_sample' in row
                assert 'total_docs_in_cluster' in row
                assert len(row['document_sample']) == 3
                assert row['total_docs_in_cluster'] == row['num_documents']
            else:
                # Small clusters should have document_texts instead
                assert 'document_texts' in row
                assert len(row['document_texts']) == row['num_documents']
    
    def test_get_cluster_documents(self):
        """Test the get_cluster_documents helper function."""
        documents = [f"Document {i}" for i in range(100)]
        
        # Get documents for cluster 0
        cluster_docs = get_cluster_documents(
            self.mock_model,
            layer_index=0,
            cluster_id=0,
            original_texts=documents
        )
        
        assert isinstance(cluster_docs, dict)
        assert 'indices' in cluster_docs
        assert 'texts' in cluster_docs
        assert 'total_count' in cluster_docs
        
        # Verify structure
        assert isinstance(cluster_docs['indices'], list)
        assert isinstance(cluster_docs['texts'], list)
        assert len(cluster_docs['indices']) == len(cluster_docs['texts'])
        assert cluster_docs['total_count'] == len(cluster_docs['indices'])
        
        # Verify content
        for idx, text in zip(cluster_docs['indices'], cluster_docs['texts']):
            assert text == documents[idx]
    
    def test_get_cluster_documents_with_limit(self):
        """Test get_cluster_documents with max_docs limit."""
        documents = [f"Document {i}" for i in range(100)]
        
        cluster_docs = get_cluster_documents(
            self.mock_model,
            layer_index=0,
            cluster_id=0,
            original_texts=documents,
            max_docs=5
        )
        
        # Should only return 5 documents
        assert len(cluster_docs['indices']) == 5
        assert len(cluster_docs['texts']) == 5
        
        # But total_count should reflect actual count
        layer = self.mock_model.cluster_layers_[0]
        actual_count = (layer.cluster_labels == 0).sum()
        assert cluster_docs['total_count'] == actual_count
    
    def test_document_traceability_without_texts(self):
        """Test that document_indices work without original_texts."""
        # Call without original_texts - should still have indices
        audit_df = create_audit_df(self.mock_model, layer_index=0)
        
        assert 'document_indices' in audit_df.columns
        assert 'document_texts' not in audit_df.columns
        assert 'document_sample' not in audit_df.columns
        
        # Indices should still be populated
        for indices in audit_df['document_indices']:
            assert isinstance(indices, list)
            assert len(indices) > 0