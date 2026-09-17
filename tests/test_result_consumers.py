"""Result consumers preserve original cluster identities and real member counts."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from toponymy.audit import (
    create_audit_df,
    create_cluster_audit_df,
    create_comparison_df,
    create_keyphrase_analysis_df,
    create_layer_summary_df,
    create_prompt_analysis_df,
    get_cluster_details,
    get_cluster_documents,
)
from toponymy.clustering import build_cluster_layers, build_cluster_tree
from toponymy.serialization import Topic, TopicModel
from toponymy.templates import Prompt
from toponymy.topic_tree import TopicTree, prune_duplicate_children, topic_name_string
from toponymy.treemap import treemap_dataframe


def model_fixture():
    labels = [
        np.array([90001, -1, 7, 90001, 7]),
        np.array([600001, -1, 600001, 600001, 600001]),
    ]
    layers = build_cluster_layers(labels)
    topics = {
        (layer.layer_index, cluster.label): Topic(
            layer.layer_index, cluster.label, cluster.members
        )
        for layer in layers
        for cluster in layer
    }
    topics[(0, 7)].name = "Apple & pear"
    topics[(0, 7)].features = {
        "cluster_keywords": ["apple", "orchard"],
        "cluster_sentences": ["apple document"],
        "exemplar_indices": [2],
    }
    topics[(0, 7)].prompt = Prompt("Name a fruit.", "apple document in an orchard")
    topics[(0, 90001)].name = "Banana"
    topics[(1, 600001)].name = "Fruit"
    topics[(1, 600001)].features = {
        "cluster_subtopics": {
            "major": ["Apple & pear", "Banana"],
            "minor": [],
            "misc": [],
        }
    }
    return TopicModel.from_topics(
        topics, layers, build_cluster_tree(labels), np.zeros((5, 2))
    )


def test_audit_uses_topic_state_and_sparse_ids():
    model = model_fixture()
    estimator = SimpleNamespace(topic_model_=model)
    frame = create_audit_df(estimator)
    assert frame["cluster_id"].tolist() == [7, 90001, 600001]
    assert frame["num_documents"].tolist() == [2, 2, 4]
    assert frame.iloc[0]["document_indices"] == [2, 4]
    assert frame.iloc[0]["first_exemplar"] == "apple document"
    assert frame.iloc[0]["all_keyphrases"] == ["apple", "orchard"]
    assert frame.iloc[2]["subtopics_list"] == ["Apple & pear", "Banana"]
    assert create_comparison_df(model)["Cluster ID"].tolist() == [7, 90001]
    frame.iloc[0]["all_keyphrases"].append("changed")
    assert model.topics[(0, 7)].features["cluster_keywords"] == ["apple", "orchard"]


def test_audit_details_prompt_and_summary_use_features():
    model = model_fixture()
    details = get_cluster_details(model, 0, 7)
    assert details["exemplar_indices"] == [2]
    assert details["prompt"] == model.topics[(0, 7)].prompt
    details["exemplars"].append("changed")
    assert model.topics[(0, 7)].features["cluster_sentences"] == ["apple document"]
    keyphrases = create_keyphrase_analysis_df(model)
    assert keyphrases["keyphrase_in_topic"].tolist() == [True, False]
    prompts = create_prompt_analysis_df(model)
    assert prompts["cluster_id"].tolist() == [7]
    assert prompts.iloc[0]["num_exemplars_in_prompt"] == 1
    assert prompts.iloc[0]["num_keyphrases_in_prompt"] == 2
    summary = create_layer_summary_df(model)
    assert summary["num_clusters"].tolist() == [2, 1]
    assert summary["avg_cluster_size"].tolist() == [2, 4]
    assert summary["has_subtopics"].tolist() == [False, True]


def test_document_sampling_zero_limit_and_alignment():
    model = model_fixture()
    texts = ["banana 0", "noise", "apple 2", "banana 3", "apple 4"]
    assert get_cluster_documents(model, 0, 90001, texts) == {
        "indices": [0, 3],
        "texts": ["banana 0", "banana 3"],
        "total_count": 2,
    }
    assert get_cluster_documents(model, 0, 90001, texts, max_docs=0) == {
        "indices": [],
        "texts": [],
        "total_count": 2,
    }
    frame = create_cluster_audit_df(
        model, include_all_docs=True, original_texts=texts, max_docs_per_cluster=0
    )
    assert frame["document_sample"].tolist() == [[], []]
    assert frame["total_docs_in_cluster"].tolist() == [2, 2]
    with pytest.raises(ValueError, match="original_texts"):
        create_audit_df(model, include_all_docs=True)
    with pytest.raises(ValueError, match="original_texts"):
        get_cluster_documents(model, 0, 7, texts[:-1])
    with pytest.raises(ValueError, match="max_docs"):
        get_cluster_documents(model, 0, 7, texts, max_docs=-1)
    with pytest.raises(KeyError):
        get_cluster_details(model, 0, 99)


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("combined", [None, "", "Only orchard evidence."])
def test_audit_preserves_explicit_combined_rendering(as_mapping, combined):
    model = model_fixture()
    prompt = Prompt("System apple", "User apple document", combined=combined)
    model.topics[(0, 7)].prompt = prompt._asdict() if as_mapping else prompt
    expected = "System apple\n\nUser apple document" if combined is None else combined
    audit = create_cluster_audit_df(model).iloc[0]
    assert audit["prompt_preview"] == expected
    assert audit["prompt_length"] == len(expected)
    analysis = create_prompt_analysis_df(model).iloc[0]
    assert analysis["prompt_length"] == len(expected)
    assert analysis["num_exemplars_in_prompt"] == (1 if combined is None else 0)
    assert analysis["num_keyphrases_in_prompt"] == (0 if combined == "" else 1)


def test_empty_audit_tables_have_stable_columns():
    empty = TopicModel.from_topics({}, [], {}, np.empty((0, 2)))
    assert create_audit_df(empty).empty
    assert "cluster_id" in create_audit_df(empty).columns
    assert "num_clusters" in create_layer_summary_df(empty).columns
    assert "prompt_length" in create_prompt_analysis_df(empty).columns
    all_noise = TopicModel.from_topics(
        {}, build_cluster_layers([np.array([-1, -1])]), {}, np.zeros((2, 2))
    )
    assert create_comparison_df(all_noise).empty
    assert create_keyphrase_analysis_df(all_noise).empty
    assert create_layer_summary_df(all_noise).iloc[0]["avg_cluster_size"] == 0
    with pytest.raises(ValueError, match="layer_index"):
        create_cluster_audit_df(empty)
    with pytest.raises(ValueError, match="fitted"):
        create_audit_df(SimpleNamespace())


def test_topic_tree_formats_mapping_values_and_propagates_options():
    tree = model_fixture().topic_tree(prune_duplicates=False)
    assert "Apple & pear" in str(tree)
    assert "Banana" in str(tree)
    html = tree.html(show_topic_id=True, cluster_percentage=True)
    assert "0_90001: Banana (2 objects) [40.00%]" in html
    assert "0_7: Apple &amp; pear (2 objects) [40.00%]" in html
    assert "1_600001: Fruit (4 objects) [80.00%]" in html


def test_topic_tree_keeps_legacy_list_values():
    tree = TopicTree(
        {(2, 0): [(1, 0)], (1, 0): [(0, 0), (0, 1)]},
        [["Apple", "Banana"], ["Fruit"]],
        [[2, 2], [4]],
        5,
    )
    assert "Apple" in str(tree)
    table = treemap_dataframe(tree)
    assert table.loc[table["id"] == "L0_C1", "value"].item() == 2


def test_treemap_preserves_large_ids_and_counts_without_dense_allocation():
    table = treemap_dataframe(model_fixture().topic_tree())
    assert len(table) == 4
    assert set(table["id"]) == {"L0_C7", "L0_C90001", "L1_C600001", "L2_C0"}
    assert table.loc[table["id"] == "L0_C90001", "value"].item() == 2
    root = table[table["parent"] == ""].iloc[0]
    assert root["value"] == 5
    assert root["layout_value"] == 5
    assert root["label"] == "Everything | 5 objects"


def test_empty_tree_and_zero_objects_render_without_division_errors(capsys):
    tree = TopicTree({}, [], [], 0)
    assert str(tree) == "Topic tree:\n"
    assert "Topic Tree" in tree.html(cluster_percentage=True)
    tree.print(cluster_percentage=True)
    assert "Topic tree:" in capsys.readouterr().out
    table = treemap_dataframe(tree)
    assert table.iloc[0]["value"] == 0
    assert table.iloc[0]["label"] == "Everything | 0 objects"
    assert "[0.00%]" in topic_name_string(
        [{90001: "Empty"}], 0, 90001, [{90001: 0}], 0, cluster_percentage=True
    )


def test_overlapping_root_topics_keep_counts_separate_from_layout_weights():
    tree = TopicTree(
        {}, [{7: "First grouping"}, {90001: "Second grouping"}], [{7: 3}, {90001: 3}], 4
    )
    table = treemap_dataframe(tree)
    root = table[table["parent"] == ""].iloc[0]
    assert root["value"] == 4
    assert root["layout_value"] == 6
    assert table.loc[table["id"] == "L0_C7", "value"].item() == 3


def test_pruning_sparse_duplicates_preserves_input_and_does_not_resurrect_nodes():
    source = {(3, 0): [(2, 6)], (2, 6): [(1, 90001)], (1, 90001): [(0, 7)]}
    names = [{7: "Leaf"}, {90001: "Parent"}, {6: "Parent"}]
    pruned = prune_duplicate_children(source, names)
    assert pruned[(2, 6)] == [(0, 7)]
    assert source[(2, 6)] == [(1, 90001)]
    tree = TopicTree(source, names, [{7: 2}, {90001: 2}, {6: 2}], 2)
    assert "L1_C90001" not in set(treemap_dataframe(tree)["id"])
    names[0][7] = "changed"
    assert "Leaf" in str(tree)


def test_unrooted_forest_keeps_unconnected_topics_and_rejects_cycles():
    tree = TopicTree({}, [{7: "Apple", 90001: "Banana"}], [{7: 2, 90001: 3}], 5)
    assert "Apple" in str(tree) and "Banana" in str(tree)
    assert len(treemap_dataframe(tree)) == 3
    with pytest.raises(ValueError, match="lower layers"):
        TopicTree({(0, 7): [(0, 7)]}, [{7: "Apple"}], [{7: 2}], 2)


def test_widget_hierarchy_uses_real_leaf_sizes_and_corpus_root():
    from toponymy.plotting import construct_topic_hierarchy

    model = model_fixture()
    hierarchy = construct_topic_hierarchy(model, model.topic_names)
    assert hierarchy["size"] == 5
    parent = hierarchy["children"][0]
    assert parent["name"] == "Fruit" and parent["size"] == 4
    assert {item["name"]: item["size"] for item in parent["children"]} == {
        "Apple & pear": 2,
        "Banana": 2,
    }
    empty = SimpleNamespace(cluster_tree_={}, cluster_layers_=())
    assert construct_topic_hierarchy(empty, []) == {"name": "Root", "size": 0}


def test_pure_plot_helpers_do_not_import_optional_widget_dependencies(monkeypatch):
    import toponymy.plotting as plotting

    monkeypatch.setitem(sys.modules, "anywidget", None)
    monkeypatch.setitem(sys.modules, "traitlets", None)
    importlib.reload(plotting)
    assert plotting.construct_topic_hierarchy(
        SimpleNamespace(cluster_tree_={}, cluster_layers_=()), []
    ) == {"name": "Root", "size": 0}


@pytest.mark.parametrize("consumer", ["topic_tree", "html", "widget"])
def test_disconnected_synthetic_roots_fail_instead_of_hiding_a_branch(consumer):
    from toponymy.plotting import construct_topic_hierarchy
    from toponymy.topic_tree import topic_tree_html

    tree = {(1, 0): [(0, 7)], (1, 1): [(0, 70)]}
    names = [{7: "Apple", 70: "Banana"}]
    sizes = [{7: 1, 70: 1}]
    with pytest.raises(ValueError, match="root"):
        if consumer == "topic_tree":
            TopicTree(tree, names, sizes, 2)
        elif consumer == "html":
            topic_tree_html(tree, names, sizes, 2)
        else:
            clusterer = SimpleNamespace(
                cluster_tree_=tree,
                cluster_layers_=build_cluster_layers([np.array([7, 70])]),
            )
            construct_topic_hierarchy(clusterer, names)


def test_one_legacy_display_root_remains_supported():
    tree = TopicTree({(3, 8): [(0, 7)]}, [{7: "Apple"}], [{7: 2}], 2)
    assert "Apple" in str(tree)
    assert set(treemap_dataframe(tree)["id"]) == {"L3_C8", "L0_C7"}


@pytest.mark.parametrize("name", ["IndentedTree", "RadialTidyTree", "CirclePacking"])
def test_widget_defaults_are_owned_valid_empty_hierarchies(name):
    pytest.importorskip("anywidget")
    import toponymy.plotting as plotting

    widget_type = getattr(plotting, name)
    first, second = widget_type(), widget_type()
    try:
        assert first.data == second.data == {"name": "Root", "size": 0}
        first.data["name"] = "Changed"
        assert second.data == {"name": "Root", "size": 0}
    finally:
        first.close()
        second.close()
