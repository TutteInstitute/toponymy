from collections import Counter
from copy import deepcopy
import pathlib
import anywidget
import numpy as np
import traitlets
from toponymy import ToponymyClusterer
from toponymy.topic_tree import TopicTree


def construct_topic_hierarchy(
    clusterer: ToponymyClusterer,
    topic_names,
    root_name="Root",
    matched_document_indices=None,
):
    """
    Constructs a hierarchical representation of topics based on clustering results.

    Args:
        clusterer: A ToponymyClusterer with a cluster_tree_ attribute representing the hierarchy.
        topic_names: A nested list or structure of topic names corresponding to cluster labels.
        root_name: Name to use for the root node in the hierarchy (default 'Root').
        matched_document_indices: Optional indices of documents returned by a search.
            When provided, each topic contains a ``match_count`` field.

    Returns:
        A nested dictionary representing the hierarchy, suitable for visualization or further processing.
    """

    cluster_tree = clusterer.cluster_tree_
    counters = [Counter(layer.cluster_labels) for layer in clusterer.cluster_layers_]
    # Root is a synthetic node above all cluster layers
    root = (len(counters), 0)
    match_counters = None
    n_matches = None

    if matched_document_indices is not None:
        n_documents = len(clusterer.cluster_layers_[0].cluster_labels)
        matched_document_indices = np.asarray(
            list(matched_document_indices), dtype=int
        )
        if np.any(matched_document_indices < 0) or np.any(
            matched_document_indices >= n_documents
        ):
            raise ValueError("matched_document_indices must contain valid document indices")

        matched_document_indices = np.unique(matched_document_indices)
        match_counters = [
            Counter(layer.cluster_labels[matched_document_indices])
            for layer in clusterer.cluster_layers_
        ]
        n_matches = len(matched_document_indices)

    hierarchy = recurse_hierarchy(
        root,
        cluster_tree,
        counters,
        topic_names,
        root_name,
        match_counters,
        n_matches,
    )

    return hierarchy


def recurse_hierarchy(
    node,
    cluster_tree: ToponymyClusterer,
    counters,
    topic_names,
    root_name,
    match_counters=None,
    n_matches=None,
):
    """
    Recursively traverses the cluster hierarchy tree to build nested topic structure.

    Args:
        node: Current node in the hierarchy (tuple or root index).
        cluster_tree: Dictionary mapping parent nodes to their children nodes.
        counters: List of Counter objects tracking label counts at each layer.
        topic_names: Nested list of topic names corresponding to cluster labels.
        root_name: Name to assign when at the root (used in the topmost call).

    Returns:
        A dictionary representing the current node with optional children and size.
    """

    try:
        u, v = node
        size = counters[u][v]
        topic = topic_names[u][v]
        match_count = (
            match_counters[u][v] if match_counters is not None else None
        )
    except IndexError:
        assert u == len(counters)
        assert v == 0
        size = counters[0].total()
        topic = root_name
        match_count = n_matches

    try:
        children = cluster_tree[node]
        item = {
            "name": topic,
            "children": [
                recurse_hierarchy(
                    child,
                    cluster_tree,
                    counters,
                    topic_names,
                    root_name,
                    match_counters,
                    n_matches,
                )
                for child in children
            ],
        }
        if size is not None:
            item.update({"size": size})
        if match_count is not None:
            item.update({"match_count": match_count})
        return item

    except KeyError:
        item = {"name": topic, "size": size}
        if match_count is not None:
            item.update({"match_count": match_count})
        return item


class IndentedTree(anywidget.AnyWidget):
    """
    A widget class for visualizing hierarchical data as an indented tree structure.
    Adapted from https://observablehq.com/@d3/indented-tree

    Static Path:
        _esm (str): Path to the widget's JavaScript module for frontend rendering.
    Attributes:
        data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
            representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
        width (traitlets.int): The width of the widget svg.
        palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
            Must be one of ['latte','frappe','macchiato','mocha']

    Usage:
        - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
        - Instantiate the widget in a Jupyter notebook.
    """

    _esm = pathlib.Path(__file__).parent / "widgets/indentedTree/dist/widget.js"
    data = traitlets.Dict(default_value={}).tag(sync=True)
    width = traitlets.Int(default_value=1024).tag(sync=True)
    palette = traitlets.Unicode(default_value="latte").tag(sync=True)


class TopicTreeSearch:
    """Search documents and display the matching distribution across a topic tree.

    Args:
        toponymy: A fitted :class:`~toponymy.Toponymy` instance.
        documents: Text documents in the same order used to fit ``toponymy``.
        root_name: Name to use for the tree root.

    Semantic queries must be embedded by the same model used to create
    ``toponymy.embedding_vectors_``. The interactive explorer requires that
    model to be supplied explicitly through its ``embedder`` argument.
    """

    def __init__(self, toponymy, documents, root_name="Root"):
        if not hasattr(toponymy, "embedding_vectors_"):
            raise ValueError("toponymy must be fitted before searching")
        if len(documents) != len(toponymy.embedding_vectors_):
            raise ValueError("documents must align with toponymy.embedding_vectors_")

        self.toponymy = toponymy
        self.documents = np.asarray(documents, dtype=str)
        self.root_name = root_name
        self._topic_name_vectors = None
        self._topic_locations = None

    def keyword(self, query):
        """Return document indices containing ``query``, case-insensitively.
        
        An empty query returns all documents.
        """
        query = query.strip()
        if not query:
            return np.arange(len(self.documents))

        return np.flatnonzero(np.char.find(np.char.lower(self.documents), query.lower()) >= 0)

    def semantic(self, query, n_document_results=20, min_similarity=None):
        """Return the nearest documents to ``query`` by cosine similarity.

        This uses exact brute-force search, which is appropriate for exploratory
        notebooks and small corpora.
        """
        if not query.strip():
            return np.array([], dtype=int)
        if n_document_results < 1:
            raise ValueError("n_document_results must be at least 1")

        query_vector = self._embed_query(query)
        return self._cosine_matches(
            query_vector,
            self.toponymy.embedding_vectors_,
            n_document_results,
            min_similarity,
        )

    def topic_keyword(self, query):
        """Return documents assigned to topics whose names contain ``query``.
        
        An empty query returns all documents.
        """
        query = query.strip().lower()
        if not query:
            return np.arange(len(self.documents))

        topic_locations = self._get_topic_locations()
        matching_topics = [
            location
            for location in topic_locations
            if query in self.toponymy.topic_names_[location[0]][location[1]].lower()
        ]
        return self._documents_for_topics(matching_topics)

    def topic_semantic(
        self, query, n_results=1, min_similarity=None, embedder=None
    ):
        """Return documents assigned to the topic names nearest to ``query``.

        By default, returns the single nearest topic (n_results=1). Topic-name 
        embeddings are generated once on first use and cached for subsequent 
        searches. All embeddings must come from the same model used to create 
        ``toponymy.embedding_vectors_``.
        """
        if not query.strip():
            return np.array([], dtype=int)

        query_vector = self._embed_query(query, embedder)
        matching_topics = self._cosine_matches(
            query_vector,
            self._get_topic_name_vectors(embedder),
            n_results,
            min_similarity,
        )
        return self._documents_for_topics(
            [self._get_topic_locations()[index] for index in matching_topics]
        )

    def _embed_query(self, query, embedder=None):
        if embedder is None:
            embedder = self.toponymy.embedding_model
        return np.asarray(
            embedder.encode([query], show_progress_bar=False)
        ).squeeze()

    @staticmethod
    def _cosine_matches(query_vector, candidate_vectors, n_results, min_similarity):
        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        candidate_vectors = np.asarray(candidate_vectors)
        if (
            query_vector.ndim != 1
            or candidate_vectors.ndim != 2
            or query_vector.shape[0] != candidate_vectors.shape[1]
        ):
            raise ValueError("query embedding dimension does not match search vectors")

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("query embedding must not be the zero vector")

        similarities = candidate_vectors @ query_vector
        similarities /= np.linalg.norm(candidate_vectors, axis=1) * query_norm
        similarities = np.nan_to_num(similarities, nan=-np.inf)
        matched_indices = np.argsort(-similarities, kind="stable")
        if min_similarity is not None:
            matched_indices = matched_indices[similarities[matched_indices] >= min_similarity]
        return matched_indices[:n_results]

    def _get_topic_locations(self):
        if self._topic_locations is None:
            self._topic_locations = [
                (layer_index, topic_index)
                for layer_index, topic_names in enumerate(self.toponymy.topic_names_)
                for topic_index in range(len(topic_names))
            ]
        return self._topic_locations

    def _get_topic_name_vectors(self, embedder=None):
        if embedder is None:
            embedder = self.toponymy.embedding_model
        if self._topic_name_vectors is None:
            self._topic_name_vectors = {}

        cache_key = id(embedder)
        if cache_key not in self._topic_name_vectors:
            topic_names = [
                self.toponymy.topic_names_[layer_index][topic_index]
                for layer_index, topic_index in self._get_topic_locations()
            ]
            self._topic_name_vectors[cache_key] = embedder.encode(
                topic_names, show_progress_bar=False
            )
        return self._topic_name_vectors[cache_key]

    def _documents_for_topics(self, topic_locations):
        matches = []
        for layer_index, topic_index in topic_locations:
            labels = self.toponymy.cluster_layers_[layer_index].cluster_labels
            matches.extend(np.flatnonzero(labels == topic_index))
        return np.unique(matches)

    def hierarchy(self, matched_document_indices):
        """Build a topic hierarchy annotated with counts for the supplied matches."""
        return construct_topic_hierarchy(
            self.toponymy,
            self.toponymy.topic_names_,
            root_name=self.root_name,
            matched_document_indices=matched_document_indices,
        )

    def html(
        self,
        matched_document_indices,
        show_documents=False,
        max_documents_per_topic=20,
    ):
        """Return the existing expandable HTML tree with search match counts.

        When ``show_documents`` is enabled, matching document snippets appear
        below their most detailed topic, up to ``max_documents_per_topic`` per
        topic.
        """
        if max_documents_per_topic < 1:
            raise ValueError("max_documents_per_topic must be at least 1")

        matched_document_indices = np.unique(
            np.asarray(list(matched_document_indices), dtype=int)
        )
        topic_names = deepcopy(self.toponymy.topic_names_)
        document_children = {}

        for layer_index, layer in enumerate(self.toponymy.cluster_layers_):
            match_counts = Counter(layer.cluster_labels[matched_document_indices])
            topic_names[layer_index] = [
                f"{name} ({match_counts[topic_index]} matches)"
                for topic_index, name in enumerate(topic_names[layer_index])
            ]

        if show_documents:
            base_labels = self.toponymy.cluster_layers_[0].cluster_labels
            for document_index in matched_document_indices:
                topic_index = base_labels[document_index]
                if topic_index < 0:
                    continue
                topic_documents = document_children.setdefault((0, topic_index), [])
                if len(topic_documents) < max_documents_per_topic:
                    snippet = self.documents[document_index].replace("\n", " ")
                    topic_documents.append(
                        f"Document {document_index}: {snippet[:160]}"
                    )

        return TopicTree(
            self.toponymy.cluster_tree_,
            topic_names,
            self.toponymy.topic_sizes_,
            len(self.documents),
            prune_duplicates=not show_documents,
            document_children=document_children,
        ).html(cluster_size=True)

    def explorer(
        self,
        n_document_results=20,
        n_topic_results=1,
        max_documents_per_topic=20,
        embedder=None,
    ):
        """Create an ipywidgets search control around the expandable HTML tree.

        Pass the embedder used to create ``toponymy.embedding_vectors_`` to 
        enable semantic document and topic-name search. When omitted, only 
        keyword search is available.
        
        Args:
            n_document_results: Number of documents to return for semantic document search.
            n_topic_results: Number of topic names to return for semantic topic search.
            max_documents_per_topic: Maximum document snippets per topic when expanded.
            embedder: The text embedding model (must be the same model used for 
                the document vectors). If None, semantic search is hidden.
        """
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "TopicTreeSearch.explorer requires ipywidgets. "
                "Install it with `pip install toponymy[interactive]`."
            ) from exc

        query = widgets.Text(
            placeholder="Search documents", layout=widgets.Layout(width="360px")
        )
        mode_options = ["Keyword"]
        if embedder is not None:
            mode_options.append("Semantic")
        mode = widgets.ToggleButtons(options=mode_options)
        target = widgets.ToggleButtons(options=["Documents", "Topic names"])
        show_documents = widgets.Checkbox(description="Show matching documents")
        search_button = widgets.Button(description="Search", button_style="primary")
        export_button = widgets.Button(description="Export CSV", button_style="success")
        status = widgets.HTML()
        tree = widgets.HTML(value=self.html([]))
        current_matches = {"indices": np.array([], dtype=int)}

        def run_search(_):
            if target.value == "Documents" and mode.value == "Keyword":
                matches = self.keyword(query.value)
            elif target.value == "Documents":
                if embedder is None:
                    status.value = "<b>Error:</b> Semantic search requires an embedder. Create explorer with embedder=model.embedding_model"
                    return
                query_vector = self._embed_query(query.value, embedder)
                matches = self._cosine_matches(
                    query_vector,
                    self.toponymy.embedding_vectors_,
                    n_document_results,
                    None,
                )
            elif mode.value == "Keyword":
                matches = self.topic_keyword(query.value)
            else:
                if embedder is None:
                    status.value = "<b>Error:</b> Semantic search requires an embedder. Create explorer with embedder=model.embedding_model"
                    return
                matches = self.topic_semantic(
                    query.value,
                    n_results=n_topic_results,
                    embedder=embedder,
                )
            status.value = f"<b>{len(matches)}</b> matching documents"
            current_matches["indices"] = matches
            tree.value = self.html(
                matches,
                show_documents=show_documents.value,
                max_documents_per_topic=max_documents_per_topic,
            )

        def export_results(_):
            """Export matched documents to JSONL."""
            import re
            
            if len(current_matches["indices"]) == 0:
                status.value = "<b>Error:</b> No search results to export. Run a search first."
                return

            # Build filename from search term and mode
            search_term = query.value.strip() or "all"
            search_term = re.sub(r'\W+', '_', search_term)[:30]
            search_mode = f"{mode.value.lower()}_{target.value.lower()}"
            
            filename = f"toponymy_export_{search_term}_{search_mode}.jsonl"
            
            # Create DataFrame and save
            df = self.export_to_dataframe(current_matches["indices"])
            df.to_json(filename, orient="records", lines=True)
            
            status.value = f"<b>✓ Exported!</b> {len(df)} documents → <code>{filename}</code>"

        def submit_search(_):
            run_search(None)

        search_button.on_click(run_search)
        export_button.on_click(export_results)
        query.continuous_update = False
        query.observe(submit_search, names="value")
        return widgets.VBox(
            [
                widgets.HBox([query, mode, target, search_button, export_button]),
                show_documents,
                status,
                tree,
            ]
        )

    def export_to_dataframe(self, matched_document_indices):
        """Export matched documents to a pandas DataFrame with topics.

        Returns:
            A DataFrame with columns: id, document_text, and topic names for each layer.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "export_to_dataframe requires pandas. Install it with `pip install pandas`."
            ) from exc

        matched_document_indices = np.asarray(
            list(matched_document_indices), dtype=int
        )
        rows = []

        for doc_idx in matched_document_indices:
            row = {
                "id": doc_idx,
                "document_text": self.documents[doc_idx],
            }
            # Get topic names at each layer for this document
            for layer_index, layer in enumerate(self.toponymy.cluster_layers_):
                topic_index = layer.cluster_labels[doc_idx]
                if topic_index >= 0:
                    topic_name = self.toponymy.topic_names_[layer_index][topic_index]
                    row[f"topic_layer_{layer_index}"] = topic_name
                else:
                    row[f"topic_layer_{layer_index}"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    def widget(self, matched_document_indices, **widget_kwargs):
        """Create an indented-tree widget annotated with search match counts."""
        return IndentedTree(
            data=self.hierarchy(matched_document_indices), **widget_kwargs
        )


class RadialTidyTree(anywidget.AnyWidget):
    """
    A widget class for visualizing hierarchical data as a radial, tidy tree diagram.
    Adapted from https://observablehq.com/@d3/radial-tree/2

    Static Paths:
        _esm (str): Path to the widget's JavaScript module for frontend rendering.
        _css (str): Path to the widget's CSS stylesheet for styling.
    Attributes:
        data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
            representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
        width (traitlets.int): The width of the widget svg.
        maxTextChars (traitlets.int): Number of characters from label to display (useful when label names
            are long).
        palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
            Must be one of ['latte','frappe','macchiato','mocha']
    Usage:
        - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
        - Instantiate the widget in a Jupyter notebook.
    """

    _esm = pathlib.Path(__file__).parent / "widgets/radialTidyTree/dist/widget.js"
    _css = pathlib.Path(__file__).parent / "widgets/radialTidyTree/dist/widget.css"
    data = traitlets.Dict(default_value={}).tag(sync=True)
    width = traitlets.Int(default_value=1024).tag(sync=True)
    maxTextChars = traitlets.Int(default_value=30).tag(sync=True)
    palette = traitlets.Unicode(default_value="latte").tag(sync=True)


class CirclePacking(anywidget.AnyWidget):
    """
    A widget class for visualizing hierarchical data as a circle packing diagram.
    Adapted from https://observablehq.com/@d3/zoomable-circle-packing.


    Static Paths:
        _esm (str): Path to the widget's JavaScript module for frontend rendering.
        _css (str): Path to the widget's CSS stylesheet for styling.
    Attributes:
        data (traitlets.Dict): The hierarchical data to visualize. Must be formatted as a nested dictionary
            representing the hierarchy (see `construct_topic_hierarchy` function for creating this structure).
        width (traitlets.int): The width of the widget svg.
        maxTextChars (traitlets.int): Number of characters from label to display (useful when label names
            are long).
        palette (traitlets.Unicode): The colour palette used for the visualization. Defaults to 'latte'.
            Must be one of ['latte','frappe','macchiato','mocha']
    Usage:
        - Assign a hierarchy dictionary to `data` with the structure produced by `construct_topic_hierarchy`.
        - Instantiate the widget in a Jupyter notebook.
    """

    _esm = pathlib.Path(__file__).parent / "widgets/circlePacking/dist/widget.js"
    _css = pathlib.Path(__file__).parent / "widgets/circlePacking/dist/widget.css"
    data = traitlets.Dict(default_value={}).tag(sync=True)
    width = traitlets.Int(default_value=1024).tag(sync=True)
    maxTextChars = traitlets.Int(default_value=50).tag(sync=True)
    palette = traitlets.Unicode(default_value="latte").tag(sync=True)
