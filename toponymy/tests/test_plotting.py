import numpy as np

from toponymy.plotting import TopicTreeSearch, construct_topic_hierarchy


class TestTopicHierarchySearchFacet:
    def test_aggregates_matches_at_each_topic_level(self):
        clusterer = type("Clusterer", (), {})()
        clusterer.cluster_tree_ = {
            (2, 0): [(1, 0), (1, 1)],
            (1, 0): [(0, 0)],
            (1, 1): [(0, 1)],
        }
        clusterer.cluster_layers_ = [
            type("Layer", (), {"cluster_labels": np.array([0, 0, 1, 1])})(),
            type("Layer", (), {"cluster_labels": np.array([0, 0, 1, 1])})(),
        ]

        hierarchy = construct_topic_hierarchy(
            clusterer,
            [["Leaf A", "Leaf B"], ["Topic A", "Topic B"]],
            matched_document_indices={0, 2, 3},
        )

        assert hierarchy["match_count"] == 3
        assert hierarchy["children"][0]["match_count"] == 1
        assert hierarchy["children"][0]["children"][0]["match_count"] == 1
        assert hierarchy["children"][1]["match_count"] == 2
        assert hierarchy["children"][1]["children"][0]["match_count"] == 2

    def test_builds_full_hierarchy_with_four_layers(self):
        """Test that hierarchy contains all 4 layers (root + 3 cluster layers)."""
        clusterer = type("Clusterer", (), {})()
        # Synthetic root at layer 3 connects down through layers 2, 1, 0
        # Layer structure: 6 documents → 3 clusters (layer 0) → 2 clusters (layer 1) → 1 cluster (layer 2) → root
        clusterer.cluster_tree_ = {
            (3, 0): [(2, 0)],                  # root → 1 cluster at layer 2
            (2, 0): [(1, 0), (1, 1)],          # layer 2 cluster 0 → 2 clusters at layer 1
            (1, 0): [(0, 0), (0, 1)],          # layer 1 cluster 0 → 2 clusters at layer 0
            (1, 1): [(0, 2)],                  # layer 1 cluster 1 → 1 cluster at layer 0
        }
        # 6 documents split across 3 clusters at layer 0
        clusterer.cluster_layers_ = [
            type("Layer", (), {"cluster_labels": np.array([0, 0, 1, 1, 2, 2])})(),
            type("Layer", (), {"cluster_labels": np.array([0, 0, 1, 1, 1, 1])})(),
            type("Layer", (), {"cluster_labels": np.array([0, 0, 0, 0, 0, 0])})(),
        ]

        hierarchy = construct_topic_hierarchy(
            clusterer,
            [
                ["Docs 0-1", "Docs 2-3", "Docs 4-5"],  # layer 0 topics
                ["Docs 0-3", "Docs 4-5"],               # layer 1 topics
                ["All Docs"],                           # layer 2 topics
            ],
            root_name="Root",
        )

        # Verify hierarchy structure
        assert hierarchy["name"] == "Root"
        assert len(hierarchy["children"]) == 1  # Should have 1 child from root (cluster at layer 2)
        assert hierarchy["children"][0]["name"] == "All Docs"

        # Verify depth is 4 levels (root + 3 layers)
        def count_depth(node):
            if "children" not in node or not node["children"]:
                return 1
            return 1 + max(count_depth(child) for child in node["children"])

        assert count_depth(hierarchy) == 4


class TestTopicTreeSearch:
    def test_keyword_search_is_case_insensitive(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["Neural Networks", "Network security", "Astronomy"],
        )

        np.testing.assert_array_equal(search.keyword("network"), [0, 1])
        np.testing.assert_array_equal(search.keyword(""), [0, 1, 2])

    def test_semantic_search_returns_nearest_documents(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["First", "Second", "Third"],
        )

        np.testing.assert_array_equal(search.semantic("first", n_document_results=2), [0, 1])

    def test_topic_keyword_search_returns_topic_documents(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["First", "Second", "Third"],
        )

        np.testing.assert_array_equal(search.topic_keyword("astronomy"), [2])

    def test_topic_semantic_search_embeds_and_caches_topic_names(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["First", "Second", "Third"],
        )

        np.testing.assert_array_equal(search.topic_semantic("astronomy"), [2])
        np.testing.assert_array_equal(search.topic_semantic("astronomy"), [2])
        assert search.toponymy.embedding_model.topic_name_encode_calls == 1

    def test_explorer_hides_semantic_search_without_embedder(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["First", "Second", "Third"],
        )

        keyword_only_explorer = search.explorer()
        semantic_explorer = search.explorer(embedder=search.toponymy.embedding_model)

        assert keyword_only_explorer.children[0].children[1].options == ("Keyword",)
        assert semantic_explorer.children[0].children[1].options == (
            "Keyword",
            "Semantic",
        )

    def test_html_adds_match_counts_and_matching_document_leaves(self):
        search = TopicTreeSearch(
            _toponymy_for_search(),
            ["First document", "Second document", "Third document"],
        )

        rendered_html = search.html([0, 1], show_documents=True)

        assert "Network methods (2 matches)" in rendered_html
        assert "Astronomy (0 matches)" in rendered_html
        assert "Document 0: First document" in rendered_html
        assert "Document 1: Second document" in rendered_html


def _toponymy_for_search():
    class Embedder:
        topic_name_encode_calls = 0

        def encode(self, texts, show_progress_bar=False):
            assert not show_progress_bar
            if texts == ["first"]:
                return np.array([[1.0, 0.0]])
            if texts == ["astronomy"]:
                return np.array([[0.0, 1.0]])
            self.topic_name_encode_calls += 1
            return np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
            )

    return type(
        "Toponymy",
        (),
        {
            "embedding_vectors_": np.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            ),
            "embedding_model": Embedder(),
            "cluster_tree_": {
                (2, 0): [(1, 0), (1, 1)],
                (1, 0): [(0, 0)],
                (1, 1): [(0, 1)],
            },
            "cluster_layers_": [
                type("Layer", (), {"cluster_labels": np.array([0, 0, 1])})(),
                type("Layer", (), {"cluster_labels": np.array([0, 1, 1])})(),
            ],
            "topic_names_": [
                ["Network methods", "Astronomy"],
                ["Computing", "Science"],
            ],
            "topic_sizes_": [[2, 1], [1, 2]],
        },
    )()