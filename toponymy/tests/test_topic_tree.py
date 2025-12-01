from toponymy import Toponymy
from toponymy.clustering import ToponymyClusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.llm_wrappers import HuggingFaceNamer

from toponymy.topic_tree import TopicTree, topic_tree_html, topic_tree_string_recursion, prune_duplicate_children
from sklearn.utils.validation import NotFittedError

import pytest

def test_topic_tree_string():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]

    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_string_recursion(tree_dict, (3, 0), topics, topic_sizes, n_objects)
    print(result)
    expected_result = (
        "Topic tree:\n"
        "   - Topic A\n"
        "     - Subtopic A1\n"
        "       - Subtopic C1\n"
        "       - Subtopic C2\n"
        "     - Subtopic A2\n"
        "       - Subtopic C3\n"
        "       - Subtopic C4\n"
        "   - Topic B\n"
        "     - Subtopic B1\n"
        "       - Subtopic C5\n"
        "       - Subtopic C6\n"
        "     - Subtopic B2\n"
        "       - Subtopic C7\n"
        "       - Subtopic C8\n"
    )
    assert result == expected_result

def test_topic_tree_string_sizes():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]

    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_string_recursion(tree_dict, (3, 0), topics, topic_sizes, n_objects, cluster_size=True, cluster_percentage=True)
    print(result)
    expected_result = (
        "Topic tree:\n"
        "   - Topic A (8 objects) [40.00%]\n"
        "     - Subtopic A1 (4 objects) [20.00%]\n"
        "       - Subtopic C1 (2 objects) [10.00%]\n"
        "       - Subtopic C2 (2 objects) [10.00%]\n"
        "     - Subtopic A2 (4 objects) [20.00%]\n"
        "       - Subtopic C3 (2 objects) [10.00%]\n"
        "       - Subtopic C4 (2 objects) [10.00%]\n"
        "   - Topic B (12 objects) [60.00%]\n"
        "     - Subtopic B1 (6 objects) [30.00%]\n"
        "       - Subtopic C5 (3 objects) [15.00%]\n"
        "       - Subtopic C6 (3 objects) [15.00%]\n"
        "     - Subtopic B2 (6 objects) [30.00%]\n"
        "       - Subtopic C7 (3 objects) [15.00%]\n"
        "       - Subtopic C8 (3 objects) [15.00%]\n"
    )
    assert result == expected_result

def test_topic_tree_html():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_html(tree_dict, topics, topic_sizes, n_objects)
    expected_result = """<div class="topic-tree">
    <style>
        .topic-tree ul {
            list-style-type: none;
            padding-left: 25px;
            margin-top: 5px;
        }
        .topic-tree li {
            list-style-type: none;
            margin-bottom: 3px;
            position: relative;
            padding-left: 15px;
        }
        .topic-tree li::before {
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
        }
        .topic-tree li.leaf-node::before {
            content: '□';
        }
        .topic-tree li.branch-node > details > summary {
            cursor: pointer;
            display: inline-block;
            list-style: none;
            margin-left: -15px;
            padding-left: 15px;
            position: relative;
        }
        .topic-tree li.branch-node > details > summary::-webkit-details-marker {
             display: none;
        }
        .topic-tree li.branch-node > details > summary::before {
            content: '▼';
            transform: rotate(-90deg);
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node > details[open] > summary::before {
            content: '▼';
            transform: rotate(0deg);
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node::before {
             content: '';
        }
    </style>
    <ul>

        <li class="branch-node">
            <details>
                <summary style="font-weight: 900;"><b>Topic Tree</b></summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="font-weight: 800;">Topic A</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="font-weight: 500;">Subtopic A1</summary>
                <ul>
                    <li class="leaf-node" style="font-weight: 200;">Subtopic C1</li>
<li class="leaf-node" style="font-weight: 200;">Subtopic C2</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="font-weight: 500;">Subtopic A2</summary>
                <ul>
                    <li class="leaf-node" style="font-weight: 200;">Subtopic C3</li>
<li class="leaf-node" style="font-weight: 200;">Subtopic C4</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="font-weight: 800;">Topic B</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="font-weight: 500;">Subtopic B1</summary>
                <ul>
                    <li class="leaf-node" style="font-weight: 200;">Subtopic C5</li>
<li class="leaf-node" style="font-weight: 200;">Subtopic C6</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="font-weight: 500;">Subtopic B2</summary>
                <ul>
                    <li class="leaf-node" style="font-weight: 200;">Subtopic C7</li>
<li class="leaf-node" style="font-weight: 200;">Subtopic C8</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        </ul></div>"""

    assert result == expected_result

def test_topic_tree_html_with_colors():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_html(tree_dict, topics, topic_sizes, n_objects, variable_color=True)
    expected_result = """<div class="topic-tree">
    <style>
        .topic-tree ul {
            list-style-type: none;
            padding-left: 25px;
            margin-top: 5px;
        }
        .topic-tree li {
            list-style-type: none;
            margin-bottom: 3px;
            position: relative;
            padding-left: 15px;
        }
        .topic-tree li::before {
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
        }
        .topic-tree li.leaf-node::before {
            content: '□';
        }
        .topic-tree li.branch-node > details > summary {
            cursor: pointer;
            display: inline-block;
            list-style: none;
            margin-left: -15px;
            padding-left: 15px;
            position: relative;
        }
        .topic-tree li.branch-node > details > summary::-webkit-details-marker {
             display: none;
        }
        .topic-tree li.branch-node > details > summary::before {
            content: '▼';
            transform: rotate(-90deg);
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node > details[open] > summary::before {
            content: '▼';
            transform: rotate(0deg);
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node::before {
             content: '';
        }
    </style>
    <ul>

        <li class="branch-node">
            <details>
                <summary style="color: #000000; font-weight: 900;;"><b>Topic Tree</b></summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #000000; font-weight: 800;;">Topic A</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d; font-weight: 500;;">Subtopic A1</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C1</li>
<li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C2</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d; font-weight: 500;;">Subtopic A2</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C3</li>
<li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C4</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #000000; font-weight: 800;;">Topic B</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d; font-weight: 500;;">Subtopic B1</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C5</li>
<li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C6</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d; font-weight: 500;;">Subtopic B2</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C7</li>
<li class="leaf-node" style="color: #c8c8c8; font-weight: 200;;">Subtopic C8</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        </ul></div>"""
    assert result == expected_result

def test_topic_tree_html_with_colors_no_weight():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_html(tree_dict, topics, topic_sizes, n_objects, variable_color=True, variable_weight=False)
    expected_result = """<div class="topic-tree">
    <style>
        .topic-tree ul {
            list-style-type: none;
            padding-left: 25px;
            margin-top: 5px;
        }
        .topic-tree li {
            list-style-type: none;
            margin-bottom: 3px;
            position: relative;
            padding-left: 15px;
        }
        .topic-tree li::before {
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
        }
        .topic-tree li.leaf-node::before {
            content: '□';
        }
        .topic-tree li.branch-node > details > summary {
            cursor: pointer;
            display: inline-block;
            list-style: none;
            margin-left: -15px;
            padding-left: 15px;
            position: relative;
        }
        .topic-tree li.branch-node > details > summary::-webkit-details-marker {
             display: none;
        }
        .topic-tree li.branch-node > details > summary::before {
            content: '▼';
            transform: rotate(-90deg);
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node > details[open] > summary::before {
            content: '▼';
            transform: rotate(0deg);
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node::before {
             content: '';
        }
    </style>
    <ul>

        <li class="branch-node">
            <details>
                <summary style="color: #000000;"><b>Topic Tree</b></summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #000000;">Topic A</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d;">Subtopic A1</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8;">Subtopic C1</li>
<li class="leaf-node" style="color: #c8c8c8;">Subtopic C2</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d;">Subtopic A2</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8;">Subtopic C3</li>
<li class="leaf-node" style="color: #c8c8c8;">Subtopic C4</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #000000;">Topic B</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d;">Subtopic B1</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8;">Subtopic C5</li>
<li class="leaf-node" style="color: #c8c8c8;">Subtopic C6</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="color: #8d8d8d;">Subtopic B2</summary>
                <ul>
                    <li class="leaf-node" style="color: #c8c8c8;">Subtopic C7</li>
<li class="leaf-node" style="color: #c8c8c8;">Subtopic C8</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        </ul></div>"""
    assert result == expected_result


def test_topic_tree_html_no_color_no_weight():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    result = topic_tree_html(tree_dict, topics, topic_sizes, n_objects, variable_color=False, variable_weight=False)
    expected_result = """<div class="topic-tree">
    <style>
        .topic-tree ul {
            list-style-type: none;
            padding-left: 25px;
            margin-top: 5px;
        }
        .topic-tree li {
            list-style-type: none;
            margin-bottom: 3px;
            position: relative;
            padding-left: 15px;
        }
        .topic-tree li::before {
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
        }
        .topic-tree li.leaf-node::before {
            content: '□';
        }
        .topic-tree li.branch-node > details > summary {
            cursor: pointer;
            display: inline-block;
            list-style: none;
            margin-left: -15px;
            padding-left: 15px;
            position: relative;
        }
        .topic-tree li.branch-node > details > summary::-webkit-details-marker {
             display: none;
        }
        .topic-tree li.branch-node > details > summary::before {
            content: '▼';
            transform: rotate(-90deg);
            position: absolute;
            left: 0;
            top: 0.15em;
            font-size: 0.9em;
            width: 1em;
            text-align: center;
            color: #555;
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node > details[open] > summary::before {
            content: '▼';
            transform: rotate(0deg);
            transition: transform 0.25s ease-in-out;
        }
        .topic-tree li.branch-node::before {
             content: '';
        }
    </style>
    <ul>

        <li class="branch-node">
            <details>
                <summary style=""><b>Topic Tree</b></summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="">Topic A</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="">Subtopic A1</summary>
                <ul>
                    <li class="leaf-node" style="">Subtopic C1</li>
<li class="leaf-node" style="">Subtopic C2</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="">Subtopic A2</summary>
                <ul>
                    <li class="leaf-node" style="">Subtopic C3</li>
<li class="leaf-node" style="">Subtopic C4</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="">Topic B</summary>
                <ul>
                    
        <li class="branch-node">
            <details>
                <summary style="">Subtopic B1</summary>
                <ul>
                    <li class="leaf-node" style="">Subtopic C5</li>
<li class="leaf-node" style="">Subtopic C6</li>

                </ul>
            </details>
        </li>
        
        <li class="branch-node">
            <details>
                <summary style="">Subtopic B2</summary>
                <ul>
                    <li class="leaf-node" style="">Subtopic C7</li>
<li class="leaf-node" style="">Subtopic C8</li>

                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        
                </ul>
            </details>
        </li>
        </ul></div>"""
    assert result == expected_result

def test_topic_tree_class():
    tree_dict = {
        (3, 0): [(2, 0), (2,1)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2), (1, 3)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
        (1, 2): [(0, 4), (0, 5)],
        (1, 3): [(0, 6), (0, 7)],
    }

    topics = [
        ["Subtopic C1", "Subtopic C2", "Subtopic C3", "Subtopic C4", "Subtopic C5", "Subtopic C6", "Subtopic C7", "Subtopic C8"],
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    topic_sizes = [
        [2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 6, 6],
        [8, 12],
    ]
    n_objects = 20

    tree_instance = TopicTree(tree_dict, topics, topic_sizes=topic_sizes, n_objects=n_objects)
    assert tree_instance.tree == tree_dict
    assert tree_instance.topics == topics

    expected_string = (
        "Topic tree:\n"
        "   - Topic A\n"
        "     - Subtopic A1\n"
        "       - Subtopic C1\n"
        "       - Subtopic C2\n"
        "     - Subtopic A2\n"
        "       - Subtopic C3\n"
        "       - Subtopic C4\n"
        "   - Topic B\n"
        "     - Subtopic B1\n"
        "       - Subtopic C5\n"
        "       - Subtopic C6\n"
        "     - Subtopic B2\n"
        "       - Subtopic C7\n"
        "       - Subtopic C8\n"
    )
    assert str(tree_instance) == expected_string
    assert tree_instance._repr_html_() == topic_tree_html(tree_dict, topics, topic_sizes, n_objects)

def test_unfitted_toponymy_fails(null_llm, embedder, clusterer):
    model = Toponymy(
        null_llm,
        embedder,
        clusterer,
        keyphrase_builder = KeyphraseBuilder(n_jobs=1),
        object_description = "sentences",
        corpus_description = "collection of sentences",
        lowest_detail_level = 0.8,
        highest_detail_level = 1.0,
        show_progress_bars=True,
    )
    with pytest.raises(NotFittedError, match="This Toponymy instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."):
        model.topic_tree_


# Tests for prune_duplicate_children function

def test_prune_duplicate_children_no_duplicates():
    """Test that tree is unchanged when there are no parent-child duplicates."""
    # Root at layer 2, children at layers 0 and 1
    tree_dict = {
        (2, 0): [(1, 0), (1, 1)],  # root node
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2), (0, 3)],
    }
    topic_names = [
        ["leaf_a", "leaf_b", "leaf_c", "leaf_d"],  # layer 0
        ["parent_a", "parent_b"],                  # layer 1
    ]
    # Note: layer 2 (root) has no topic names
    
    result = prune_duplicate_children(tree_dict, topic_names)
    assert result == tree_dict


def test_prune_duplicate_children_single_duplicate_with_grandchildren():
    """Test removing a single child that duplicates parent, promoting grandchildren."""
    tree_dict = {
        (2, 0): [(1, 0)],           # root -> parent_a
        (1, 0): [(0, 0), (0, 1)],   # parent_a -> [child_a (duplicate), child_b]
        (0, 0): [],                 # child_a has grandchildren (leaves)
    }
    topic_names = [
        ["parent_a", "child_b"],  # layer 0: child at index 0 duplicates parent
        ["parent_a"],             # layer 1
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # child (0,0) should be removed from parent (1,0)'s children
    expected = {
        (2, 0): [(1, 0)],
        (1, 0): [(0, 1)],  # only child_b remains
        (0, 0): [],
    }
    assert result == expected


def test_prune_duplicate_children_leaf_duplicate():
    """Test removing a leaf child that duplicates its parent."""
    tree_dict = {
        (2, 0): [(1, 0), (1, 1)],  # root
        (1, 0): [(0, 0), (0, 1)],  # parent_a has two children
        (1, 1): [(0, 2)],
    }
    topic_names = [
        ["parent_a", "unique_child", "other"],  # layer 0: first child duplicates parent
        ["parent_a", "parent_b"],               # layer 1
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # (0,0) is a leaf that duplicates (1,0), should be removed
    expected = {
        (2, 0): [(1, 0), (1, 1)],
        (1, 0): [(0, 1)],  # (0,0) removed
        (1, 1): [(0, 2)],
    }
    assert result == expected


def test_prune_duplicate_children_allows_sibling_duplicates():
    """Test that siblings with duplicate names are NOT removed (key behavior)."""
    tree_dict = {
        (2, 0): [(1, 0)],          # root
        (1, 0): [(0, 0), (0, 1)],  # parent has two children with same name
    }
    topic_names = [
        ["same_name", "same_name"],  # layer 0: siblings with duplicate names
        ["different_parent"],         # layer 1
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # Both children should remain (siblings can have duplicate names)
    assert result == tree_dict


def test_prune_duplicate_children_multiple_children_duplicate_parent():
    """Test when multiple children duplicate the same parent."""
    tree_dict = {
        (2, 0): [(1, 0)],
        (1, 0): [(0, 0), (0, 1), (0, 2)],
        (0, 0): [],
        (0, 1): [],
    }
    topic_names = [
        ["topic_a", "topic_a", "unique"],  # two children duplicate parent
        ["topic_a"],
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # Both (0,0) and (0,1) should be removed
    expected = {
        (2, 0): [(1, 0)],
        (1, 0): [(0, 2)],  # only unique child remains
        (0, 0): [],
        (0, 1): [],
    }
    assert result == expected


def test_prune_duplicate_children_with_promotion():
    """Test that grandchildren are promoted when intermediate duplicate is removed."""
    tree_dict = {
        (3, 0): [(2, 0)],           # root
        (2, 0): [(1, 0), (1, 1)],   # layer 2
        (1, 0): [(0, 0), (0, 1)],   # layer 1: this duplicates parent
        (1, 1): [(0, 2)],
    }
    topic_names = [
        ["a", "b", "c"],        # layer 0
        ["topic_x", "unique"],  # layer 1: first child duplicates parent
        ["topic_x"],            # layer 2
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # (1,0) duplicates (2,0), so (1,0) removed and its children promoted
    expected = {
        (3, 0): [(2, 0)],
        (2, 0): [(0, 0), (0, 1), (1, 1)],  # grandchildren promoted + remaining child
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2)],
    }
    assert result == expected


def test_prune_duplicate_children_cascading_duplicates():
    """Test handling of duplicates at multiple levels."""
    tree_dict = {
        (3, 0): [(2, 0)],
        (2, 0): [(1, 0)],
        (1, 0): [(0, 0), (0, 1)],
    }
    topic_names = [
        ["topic_a", "different"],  # layer 0
        ["topic_a"],               # layer 1: duplicates layer 2
        ["topic_a"],               # layer 2
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # (1,0) duplicates (2,0) and should be removed, promoting grandchildren
    expected = {
        (3, 0): [(2, 0)],
        (2, 0): [(0, 1)],  # grandchildren promoted
        (1, 0): [(0, 0), (0, 1)]
    }
    assert result == expected


def test_prune_duplicate_children_preserves_other_children():
    """Test that non-duplicate children are preserved alongside removed duplicates."""
    tree_dict = {
        (2, 0): [(1, 0), (1, 1), (1, 2)],
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (1, 2): [],
    }
    topic_names = [
        ["a", "b"],                      # layer 0
        ["dup", "unique", "dup"],        # layer 1: indices 0 and 2 duplicate parent
        ["dup"],                         # layer 2 (parent)
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # (1,0) and (1,2) should be removed, promoting their children
    expected = {
        (2, 0): [(0, 0), (1, 1)],  # grandchild from (1,0) + unique child (1,1)
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (1, 2): [],
    }
    assert result == expected


def test_prune_duplicate_children_empty_tree():
    """Test handling of empty tree."""
    tree_dict = {}
    topic_names = []
    
    result = prune_duplicate_children(tree_dict, topic_names)
    assert result == {}


def test_prune_duplicate_children_root_not_compared():
    """Test that root node children are never removed (root has no topic name)."""
    tree_dict = {
        (1, 0): [(0, 0), (0, 1)],  # root at layer 1
    }
    topic_names = [
        ["any_name", "same_name"],  # layer 0
    ]
    # Root is at layer 1, which is >= len(topic_names), so no comparison happens
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # All children of root should remain regardless of their names
    assert result == tree_dict


def test_prune_duplicate_children_complex_tree():
    """Test a complex realistic tree structure."""
    tree_dict = {
        (4, 0): [(3, 0), (3, 1)],           # root
        (3, 0): [(2, 0), (2, 1)],           # layer 3
        (3, 1): [(2, 2)],
        (2, 0): [(1, 0), (1, 1)],           # layer 2
        (2, 1): [(1, 2)],
        (2, 2): [(1, 3)],
        (1, 0): [(0, 0)],                   # layer 1
        (1, 1): [(0, 1)],
        (1, 2): [(0, 2)],
        (1, 3): [],
    }
    topic_names = [
        ["leaf0", "leaf1", "leaf2"],        # layer 0
        ["sub0", "sub1", "topic_b", "x"],   # layer 1: (1,2) duplicates its parent (3,1)
        ["topic_a", "topic_b", "topic_c"],  # layer 2: (2,0) duplicates its parent (3,0)
        ["topic_a", "different"],           # layer 3: no duplicates
    ]

    result = prune_duplicate_children(tree_dict, topic_names)

    # (2,1) has name "topic_b" but we need to check what its parent (3,0) has
    # Parent (3,0) has name "topic_a", so no duplicate there
    # Check (1,2): parent is (2,1) with "topic_b", child is "topic_b" -> DUPLICATE!
    # (2,0) has name "topic_a" which duplicates its parent (3,0)

    x = TopicTree(result, topic_names, topic_sizes=list(list()), n_objects=5, prune_duplicates=False)
    print(x.tree)
    print(x)

    expected = {
        (4, 0): [(3, 0), (3, 1)], 
        (3, 0): [(1, 0), (1, 1), (2, 1)], 
        (3, 1): [(2, 2)], 
        (2, 0): [(1, 0), (1, 1)], 
        (2, 1): [(0, 2)], 
        (2, 2): [(1, 3)], 
        (1, 0): [(0, 0)], 
        (1, 1): [(0, 1)], 
        (1, 2): [(0, 2)], 
        (1, 3): []
    }


    assert result == expected


def test_prune_duplicate_children_all_children_duplicates():
    """Test when all children of a parent are duplicates."""
    tree_dict = {
        (2, 0): [(1, 0), (1, 1)],
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
    }
    topic_names = [
        ["leaf0", "leaf1"],
        ["same", "same"],    # both children have same name as parent
        ["same"],            # parent
    ]
    
    result = prune_duplicate_children(tree_dict, topic_names)
    
    # Both children should be removed, grandchildren promoted
    expected = {
        (2, 0): [(0, 0), (0, 1)],  # all grandchildren promoted
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
    }
    assert result == expected

    
