import numpy as np
import html
from typing import Dict, List, Tuple, NewType
from typing_extensions import Literal

ClusterTree = NewType("ClusterTree", Dict[Tuple[int, int], List[Tuple[int, int]]])

def topic_name_string(
    topic_names: List[List[str]],
    layer: int,
    index: int,
    cluster_sizes: List[List[int]],
    n_objects: int,
    cluster_size: bool = False,
    cluster_percentage: bool = False,
    show_topic_id: bool = False,
) -> str:
    if layer < len(topic_names) and index < len(topic_names[layer]):
        topic_string = topic_names[layer][index]
    else:
        topic_string = f"Unnamed topic from layer {layer} cluster {index}\n"

    if show_topic_id:
        topic_string = f"{layer}_{index}: {topic_string}"
    if cluster_size:
        topic_string += f" ({cluster_sizes[layer][index]} objects)"
    if cluster_percentage:
        topic_string += f" [{cluster_sizes[layer][index] / n_objects * 100:.2f}%]"

    return topic_string

def topic_tree_string_recursion(
    tree: ClusterTree,
    root_node: Tuple[int, int],
    topics: List[List[str]],
    topic_sizes: List[List[int]],
    n_objects: int,
    indent_level: int = 0,
    indent_size: int = 2,
    cluster_size: bool = False,
    cluster_percentage: bool = False,
    show_topic_id: bool = False,
) -> str:
    """
    Recursively traverses the topic tree and constructs a string representation of the topics.

    Parameters
    ----------
    tree : ClusterTree
        The topic tree represented as a dictionary.
    root_node : Tuple[int, int]
        The current node in the tree.
    topics : List[List[str]]
        The list of topics to be included in the string representation.

    Returns
    -------
    str
        A string representation of the topics in the tree.
    """
    if root_node[0] < len(topics):
        topic_string = topic_name_string(
            topics,
            root_node[0],
            root_node[1],
            topic_sizes,
            n_objects,
            cluster_size=cluster_size,
            cluster_percentage=cluster_percentage,
            show_topic_id=show_topic_id,
        )
        result = f"{' ' * indent_level} - {topic_string}\n"
    else:
        result = "Topic tree:\n"

    for child_node in tree.get(root_node, []):
        result += topic_tree_string_recursion(
            tree,
            child_node,
            topics,
            topic_sizes,
            n_objects,
            indent_level + indent_size,
            indent_size,
            cluster_size,
            cluster_percentage,
            show_topic_id,
        )

    return result


def topic_tree_html_recursion(
    tree_dict: ClusterTree,
    root_node: Tuple[int, int],
    topic_names: List[List[str]],
    topic_sizes: List[List[int]],
    n_objects: int,
    max_layer: int,
    variable_color: bool = False,
    variable_weight: bool = True,
    cluster_size: bool = False,
    cluster_percentage: bool = False,
    show_topic_id: bool = False,
) -> str:
    """
    Recursively traverses the topic tree and constructs an HTML representation of the topics.

    Parameters
    ----------
    tree_dict : ClusterTree
        The topic tree represented as a dictionary.
    root_node : Tuple[int, int]
        The current node in the tree.
    topic_names : List[List[str]]
        The list of topics to be included in the HTML representation.
    max_layer : int
        The maximum layer of the tree.
    variable_color : bool
        If True, the color of the topic name will vary based on its layer.
    variable_weight : bool
        If True, the font weight of the topic name will vary based on its layer.

    Returns
    -------
    str
        An HTML representation of the topics in the tree.
    """
    layer, index = root_node
    if layer < len(topic_names):
        topic_name = html.escape(
            topic_name_string(
                topic_names,
                layer,
                index,
                topic_sizes,
                n_objects,
                cluster_size=cluster_size,
                cluster_percentage=cluster_percentage,
                show_topic_id=show_topic_id,
            )
        )
    else:
        topic_name = "<b>Topic Tree</b>"

    if max_layer > 0:
        if layer == max_layer:
            gray_val = 0
            weight_val = 900
        else:
            gray_val = np.sqrt(1.0 - (layer / (max_layer - 1))) * 200
            weight_val = 200 + (layer / (max_layer - 1)) * 600
            weight_val = round(weight_val / 100) * 100
    elif layer == 0:
        gray_val = 0
        weight_val = 800

    gray_val = int(max(0, min(255, gray_val)))

    # Format as hex code (e.g., #5a5a5a)
    hex_code = f"{gray_val:02x}"
    color_style = f"color: #{hex_code}{hex_code}{hex_code};"
    weight_style = f"font-weight: {weight_val};"
    if variable_weight and variable_color:
        combined_style = f"{color_style} {weight_style};"
    elif variable_weight:
        combined_style = f"{weight_style}"
    elif variable_color:
        combined_style = f"{color_style}"
    else:
        combined_style = ""

    children = tree_dict.get(root_node, [])  # Get children, default to empty list

    # Leaf Node
    if not children:
        return f'<li class="leaf-node" style="{combined_style}">{topic_name}</li>\n'

    # Node with Children
    else:
        child_html_items = ""
        for child_id in children:
            child_html_items += topic_tree_html_recursion(
                tree_dict,
                child_id,
                topic_names,
                topic_sizes,
                n_objects,
                max_layer=max_layer,
                variable_color=variable_color,
                variable_weight=variable_weight,
            )

        html_content = f"""
        <li class="branch-node">
            <details>
                <summary style="{combined_style}">{topic_name}</summary>
                <ul>
                    {child_html_items}
                </ul>
            </details>
        </li>
        """
        return html_content


def topic_tree_html(
    tree_dict: ClusterTree,
    topic_names: List[List[str]],
    topic_sizes: List[List[int]],
    n_objects: int,
    variable_color: bool = False,
    variable_weight: bool = True,
) -> str:
    """
    Converts a topic tree into an HTML representation.

    Parameters
    ----------

    tree_dict : ClusterTree
        The topic tree represented as a dictionary.
    topic_names : List[List[str]]
        The list of topics to be included in the HTML representation.
    variable_color : bool
        If True, the color of the topic name will vary based on its layer.
    variable_weight : bool
        If True, the font weight of the topic name will vary based on its layer.

    Returns
    -------
    str
        An HTML representation of the topics in the tree.
    """
    root_node = max(
        tree_dict.keys(),
    )
    # Start the main HTML list
    root_html = "<ul>\n"
    root_html += topic_tree_html_recursion(
        tree_dict,
        root_node,
        topic_names,
        topic_sizes,
        n_objects,
        max_layer=root_node[0],
        variable_color=variable_color,
        variable_weight=variable_weight,
    )
    root_html += "</ul>"

    # Add CSS styles for the topic tree
    style = """
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
    """

    result = f'<div class="topic-tree">{style}{root_html}</div>'

    return result


class TopicTree:
    """
    A class to represent a topic tree. Provides string representations and
    an HTML representation in notebooks.

    Parameters
    ----------
    tree : ClusterTree
        The topic tree represented as a dictionary.

    topics : List[List[str]]
        The list of topics to be included in the tree.

    Attributes
    ----------
    tree : ClusterTree
        The topic tree represented as a dictionary.

    topics : List[List[str]]
        The list of topics to be included in the tree.
    """

    def __init__(
        self,
        tree: ClusterTree,
        topics: List[List[str]],
        topic_sizes: List[List[int]],
        n_objects: int,
    ) -> None:
        self.tree = tree
        self.topics = topics
        self.topic_sizes = topic_sizes
        self.n_objects = n_objects

    def __str__(self) -> str:
        root_node = max(
            self.tree.keys(),
        )
        result = topic_tree_string_recursion(
            self.tree,
            root_node,
            self.topics,
            self.topic_sizes,
            self.n_objects,
            indent_level=0,
        )
        return result

    def _repr_html_(self) -> str:
        return topic_tree_html(
            self.tree,
            self.topics,
            self.topic_sizes,
            self.n_objects,
            variable_color=False,
            variable_weight=True,
        )

    def print(
        self,
        indent_size: int = 2,
        cluster_size: bool = True,
        cluster_percentage: bool = False,
        show_topic_id: bool = False,
    ) -> None:
        """
        Prints the topic tree in a human-readable format.
        
        Parameters
        ----------
        indent_size : int
            The number of spaces to use for indentation.
        cluster_size : bool
            If True, include the size of each cluster in the output.
        cluster_percentage : bool
            If True, include the percentage of each cluster in the output.
        show_topic_id : bool
            If True, include the topic ID in the output.
        """
        tree_string = topic_tree_string_recursion(
            self.tree,
            max(self.tree.keys()),
            self.topics,
            self.topic_sizes,
            n_objects=self.n_objects,
            indent_level=0,
            indent_size=indent_size,
            cluster_size=cluster_size,
            cluster_percentage=cluster_percentage,
            show_topic_id=show_topic_id,
        )
        print(tree_string)

    def html(
        self,
        variable_color: bool = False,
        variable_weight: bool = True,
        cluster_size: bool = True,
        cluster_percentage: bool = False,
        show_topic_id: bool = False,
    ) -> str:
        """
        Returns an HTML representation of the topic tree.
        
        Parameters
        ----------
        variable_color : bool
            If True, the color of the topic name will vary based on its layer.
        variable_weight : bool
            If True, the font weight of the topic name will vary based on its layer.
        cluster_size : bool
            If True, include the size of each cluster in the output.
        cluster_percentage : bool
            If True, include the percentage of each cluster in the output.
        show_topic_id : bool
            If True, include the topic ID in the output.
            
        Returns
        -------
        str
            An HTML representation of the topics in the tree.
        """
        return topic_tree_html(
            self.tree,
            self.topics,
            self.topic_sizes,
            self.n_objects,
            variable_color=variable_color,
            variable_weight=variable_weight,
            cluster_size=cluster_size,
            cluster_percentage=cluster_percentage,
            show_topic_id=show_topic_id,
        )
