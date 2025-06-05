from toponymy import Toponymy
from toponymy.clustering import ToponymyClusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.llm_wrappers import HuggingFace

from sentence_transformers import SentenceTransformer
from toponymy.topic_tree import TopicTree, topic_tree_html, topic_tree_string_recursion
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
