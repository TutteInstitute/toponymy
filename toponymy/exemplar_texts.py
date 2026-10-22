import random
import numpy as np
import numba
from typing import List, Tuple, FrozenSet, Dict, Callable, Any
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsTransformer
from toponymy.utility_functions import diversify_max_alpha as diversify
from toponymy._utils import handle_verbose_params
from toponymy.annotation import NodeId

from tqdm.auto import tqdm
import math

from apricot import SaturatedCoverageSelection

###################################################################################
# This code is a modified version of the apricot library's facility location
# facilityLocation.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from apricot.functions.base import BaseGraphSelection
from apricot.optimizers import LazyGreedy
from apricot.optimizers import ApproximateLazyGreedy
from apricot.optimizers import SieveGreedy

dtypes = [
    "void(float64[:,:], float64[:], float64[:], int64[:])",
    "void(float32[:,:], float64[:], float64[:], int64[:])",
]
sdtypes = [
    "void(float64[:], int32[:], int32[:], float64[:], float64[:], int64[:])",
    "void(float32[:], int32[:], int32[:], float64[:], float64[:], int64[:])",
]
sieve_dtypes = (
    "void(float64[:,:], int64, float64[:,:], int64[:,:],"
    "float64[:,:], float64[:], float64[:], int64[:], int64[:])"
)


@numba.njit(dtypes, fastmath=True, cache=True)
def calculate_gains_(X, gains, current_values, idxs):
    for i in range(idxs.shape[0]):
        idx = idxs[i]
        gains[i] = np.maximum(X[idx], current_values).sum()


@numba.njit(sdtypes, fastmath=True, cache=True)
def calculate_gains_sparse_(X_data, X_indices, X_indptr, gains, current_values, idxs):
    for i in range(idxs.shape[0]):
        idx = idxs[i]

        start = X_indptr[idx]
        end = X_indptr[idx + 1]

        for j in range(start, end):
            k = X_indices[j]
            gains[i] += max(X_data[j], current_values[k]) - current_values[k]


class FacilityLocationSelection(BaseGraphSelection):
    """A selector based off a facility location submodular function.

    Facility location functions are general purpose submodular functions that,
    when maximized, choose examples that represent the space of the data well.
    The facility location function is based on maximizing the pairwise
    similarities between the points in the data set and their nearest chosen
    point. The similarity function can be specified by the user but must be
    non-negative where a higher value indicates more similar.

    .. note::
            All ~pairwise~ values in your data must be non-negative for this
            selection to work.

    In many ways, optimizing a facility location function is simply a greedy
    version of k-medoids, where after the first few examples are selected, the
    subsequent ones are at the center of clusters. The function, like most
    graph-based functions, operates on a pairwise similarity matrix, and
    successively chooses examples that are similar to examples whose current
    most-similar example is still very dissimilar. Phrased another way,
    successively chosen examples are representative of underrepresented
    examples.

    The general form of a facility location function is

    .. math::
            f(X, Y) = \\sum\\limits_{y in Y} \\max_{x in X} \\phi(x, y)

    where :math:`f` indicates the function, :math:`X` is a subset, :math:`Y`
    is the ground set, and :math:`\\phi` is the similarity measure between two
    examples. Like most graph-based functons, the facility location function
    requires access to the full ground set.

    This implementation allows users to pass in either their own symmetric
    square matrix of similarity values, or a data matrix as normal and a function
    that calculates these pairwise values.

    For more details, see https://las.inf.ethz.ch/files/krause12survey.pdf
    page 4.

    Parameters
    ----------
    n_samples : int
            The number of samples to return.

    metric : str, optional
            The method for converting a data matrix into a square symmetric matrix
            of pairwise similarities. If a string, can be any of the metrics
            implemented in sklearn (see
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html),
            including "precomputed" if one has already generated a similarity
            matrix. Note that sklearn calculates distance matrices whereas apricot
            operates on similarity matrices, and so a distances.max() - distances
            transformation is performed on the resulting distances. For
            backcompatibility, 'corr' will be read as 'correlation'. Default is
            'euclidean'.

    initial_subset : list, numpy.ndarray or None, optional
            If provided, this should be a list of indices into the data matrix
            to use as the initial subset, or a group of examples that may not be
            in the provided data should beused as the initial subset. If indices,
            the provided array should be one-dimensional. If a group of examples,
            the data should be 2 dimensional. Default is None.

    optimizer : string or optimizers.BaseOptimizer, optional
            The optimization approach to use for the selection. Default is
            'two-stage', which makes selections using the naive greedy algorithm
            initially and then switches to the lazy greedy algorithm. Must be
            one of

                    'random' : randomly select elements (dummy optimizer)
                    'modular' : approximate the function using its modular upper bound
                    'naive' : the naive greedy algorithm
                    'lazy' : the lazy (or accelerated) greedy algorithm
                    'approximate-lazy' : the approximate lazy greedy algorithm
                    'two-stage' : starts with naive and switches to lazy
                    'stochastic' : the stochastic greedy algorithm
                    'sample' : randomly take a subset and perform selection on that
                    'greedi' : the GreeDi distributed algorithm
                    'bidirectional' : the bidirectional greedy algorithm

            Default is 'two-stage'.

    optimizer_kwds : dict or None
            A dictionary of arguments to pass into the optimizer object. The keys
            of this dictionary should be the names of the parameters in the optimizer
            and the values in the dictionary should be the values that these
            parameters take. Default is None.

    n_neighbors : int or None
            When constructing a similarity matrix, the number of nearest neighbors
            whose similarity values will be kept. The result is a sparse similarity
            matrix which can significantly speed up computation at the cost of
            accuracy. Default is None.

    reservoir : numpy.ndarray or None
            The reservoir to use when calculating gains in the sieve greedy
            streaming optimization algorithm in the `partial_fit` method.
            Currently only used for graph-based functions. If a numpy array
            is passed in, it will be used as the reservoir. If None is passed in,
            will use reservoir sampling to collect a reservoir. Default is None.

    max_reservoir_size : int
            The maximum size that the reservoir can take. If a reservoir is passed
            in, this value is set to the size of that array. Default is 1000.

    n_jobs : int
            The number of threads to use when performing computation in parallel.
            Currently, this parameter is exposed but does not actually do anything.
            This will be fixed soon.

    random_state : int or RandomState or None, optional
            The random seed to use for the random selection process. Only used
            for stochastic greedy.

    verbose : bool
            Whether to print output during the selection process.

    Attributes
    ----------
    n_samples : int
            The number of samples to select.

    ranking : numpy.array int
            The selected samples in the order of their gain.

    gains : numpy.array float
            The gain of each sample in the returned set when it was added to the
            growing subset. The first number corresponds to the gain of the first
            added sample, the second corresponds to the gain of the second added
            sample, and so forth.
    """

    def __init__(
        self,
        n_samples,
        metric="euclidean",
        initial_subset=None,
        optimizer="lazy",
        optimizer_kwds={},
        n_neighbors=None,
        reservoir=None,
        max_reservoir_size=1000,
        n_jobs=1,
        random_state=None,
        verbose=False,
    ):

        super(FacilityLocationSelection, self).__init__(
            n_samples=n_samples,
            metric=metric,
            initial_subset=initial_subset,
            optimizer=optimizer,
            optimizer_kwds=optimizer_kwds,
            n_neighbors=n_neighbors,
            reservoir=reservoir,
            max_reservoir_size=max_reservoir_size,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y=None, sample_weight=None, sample_cost=None):
        """A specialized version of Facility Location selection from
        apricot by Jacob Schreiber. This code is a minor modification
        of the original apricot code to make it more suitable for
        selection of exemplars for clusters in a cluster layer.

        Note that this version is specialized and does no support sparse
        input, nor does it support streaming via sieve greedy.

        Run submodular optimization to select the examples.

        This method is a wrapper for the full submodular optimization process.
        It takes in some data set (and optionally labels that are ignored
        during this process) and selects `n_samples` from it in the greedy
        manner specified by the optimizer.

        This method will return the selector object itself, not the transformed
        data set. The `transform` method will then transform a data set to the
        selected points, or alternatively one can use the ranking stored in
        the `self.ranking` attribute. The `fit_transform` method will perform
        both optimization and selection and return the selected items.

        Parameters
        ----------
        X : list or numpy.ndarray, shape=(n, d)
                The data set to transform. Must be numeric.

        y : list or numpy.ndarray or None, shape=(n,), optional
                The labels to transform. If passed in this function will return
                both the data and th corresponding labels for the rows that have
                been selected.

        sample_weight : list or numpy.ndarray or None, shape=(n,), optional
                The weight of each example. Currently ignored in apricot but
                included to maintain compatibility with sklearn pipelines.

        sample_cost : list or numpy.ndarray or None, shape=(n,), optional
                The cost of each item. If set, indicates that optimization should
                be performed with respect to a knapsack constraint.

        Returns
        -------
        self : FacilityLocationSelection
                The fit step returns this selector object.
        """
        if X.shape[0] > 4096:
            X_pairwise = KNeighborsTransformer(
                n_neighbors=512, metric=self.metric
            ).fit_transform(X)
            original_metric = self.metric
            self.metric = "precomputed"
            result = super(FacilityLocationSelection, self).fit(
                X_pairwise, y=y, sample_weight=sample_weight, sample_cost=sample_cost
            )
            self.metric = original_metric
            return result
        else:
            return super(FacilityLocationSelection, self).fit(
                X, y=y, sample_weight=sample_weight, sample_cost=sample_cost
            )

    def _initialize(self, X_pairwise):
        super(FacilityLocationSelection, self)._initialize(X_pairwise)

        self.current_values_sum = self.current_values.sum()

        if self.sparse:
            self.calculate_gains_ = calculate_gains_sparse_
        else:
            self.calculate_gains_ = calculate_gains_

    def _calculate_gains(self, X_pairwise, idxs=None):
        idxs = idxs if idxs is not None else self.idxs
        gains = np.zeros(idxs.shape[0], dtype="float64")
        if self.sparse:
            self.calculate_gains_(
                X_pairwise.data,
                X_pairwise.indices,
                X_pairwise.indptr,
                gains,
                self.current_values,
                idxs,
            )
        else:
            self.calculate_gains_(X_pairwise, gains, self.current_values, idxs)
        gains -= self.current_values_sum

        return gains

    def _select_next(self, X_pairwise, gain, idx):
        """This function will add the given item to the selected set."""
        if self.sparse:
            self.current_values = np.maximum(
                np.squeeze(X_pairwise.toarray()), self.current_values
            )
        else:
            self.current_values = np.maximum(X_pairwise, self.current_values)
        self.current_values_sum = self.current_values.sum()

        super(FacilityLocationSelection, self)._select_next(X_pairwise, gain, idx)


###################################################################################################


class SubmodularExemplarAnnotator:
    """Select a node's exemplars by submodular optimization over its object vectors.

    Chooses ``n_exemplars`` objects that represent the node's object vectors well,
    using a facility location (the default) or saturated coverage submodular
    function with a cosine metric. The vectors are centred by ``null_topic`` first.
    Nodes with more than 16384 objects are randomly subsampled to 16384 before
    selection, and a node with fewer objects than ``n_exemplars`` returns all of them.

    Parameters
    ----------
    n_exemplars : int, optional
        The number of exemplars to select for each node, by default 4.
    object_to_text_function : Callable[[List[Any]], List[str]], optional
        A function that turns the chosen objects into exemplar strings. If None,
        the objects are returned as they are.
    null_topic : np.ndarray, optional
        A background vector, with the same dimension as the object vectors, that is
        subtracted from them before selection. This is typically the mean of all
        object vectors in the corpus. If None, the vectors are not centred.
    submodular_function : {"facility_location", "saturated_coverage"}, optional
        The submodular function to maximise, by default "facility_location".

    Attributes
    ----------
    inputs : tuple of str
        ``cluster_objects``, ``cluster_object_vectors`` and ``cluster_object_indices``.
    outputs : tuple of str
        ``exemplars`` and ``exemplar_original_indices``.
    algorithm_type : str
        ``"node-node"``: each node is annotated from its own inputs.

    Raises
    ------
    ValueError
        If ``submodular_function`` is not one of the supported values.
    """

    inputs = ("cluster_objects", "cluster_object_vectors", "cluster_object_indices")
    outputs = ("exemplars", "exemplar_original_indices")
    algorithm_type = "node-node"

    def __init__(
        self,
        n_exemplars: int = 4,
        object_to_text_function=None,
        null_topic: np.ndarray = None,
        submodular_function: str = "facility_location",
    ):
        self.n_exemplars = n_exemplars
        self.object_to_text_function = object_to_text_function
        self.null_topic = null_topic  # subtracted from vectors; None means no centering
        self.submodular_function = submodular_function
        if submodular_function == "facility_location":
            self.selector = FacilityLocationSelection(
                n_exemplars, metric="cosine", optimizer="lazy"
            )
        elif submodular_function == "saturated_coverage":
            self.selector = SaturatedCoverageSelection(
                n_exemplars, metric="cosine", optimizer="lazy"
            )
        else:
            raise ValueError(
                f"selection_function={submodular_function} is not a valid selection. Please choose one of (facility_location,saturated_coverage)"
            )

    def annotate(
        self,
        node,
        *,
        cluster_objects,
        cluster_object_vectors,
        cluster_object_indices,
    ):
        """Select exemplars for one node.

        Parameters
        ----------
        node : NodeId
            The node being annotated. It is not used in the selection.
        cluster_objects : Sequence
            The objects in the node's cluster.
        cluster_object_vectors : np.ndarray
            One uncentred vector for each entry of ``cluster_objects``.
        cluster_object_indices : Sequence[int]
            The position in the corpus of each entry of ``cluster_objects``.

        Returns
        -------
        dict
            ``{"exemplars": ..., "exemplar_original_indices": ...}``, the chosen
            exemplars (as strings if ``object_to_text_function`` is given) and their
            positions in the corpus. Both are empty for a node with no objects.
        """
        if len(cluster_objects) == 0:
            return {"exemplars": [], "exemplar_original_indices": []}

        # Subsample if cluster is too large
        subsample_indices = None
        if len(cluster_objects) > 16384:
            subsample_indices = np.sort(
                np.random.choice(len(cluster_objects), size=16384, replace=False)
            )
            subsampled_objects = [cluster_objects[i] for i in subsample_indices]
            subsampled_vectors = cluster_object_vectors[subsample_indices]
        else:
            subsampled_objects = cluster_objects
            subsampled_vectors = cluster_object_vectors

        centered_vectors = subsampled_vectors
        if self.null_topic is not None:
            centered_vectors = subsampled_vectors - self.null_topic
        cluster_indices = np.arange(centered_vectors.shape[0])

        # Select exemplars
        if centered_vectors.shape[0] >= self.n_exemplars:
            _, candidate_indices = self.selector.fit_transform(
                centered_vectors, y=cluster_indices
            )
        else:
            candidate_indices = cluster_indices

        # Extract exemplars from subsampled set
        chosen_exemplars = [subsampled_objects[i] for i in candidate_indices]
        if self.object_to_text_function is not None:
            chosen_exemplars = self.object_to_text_function(chosen_exemplars)

        if subsample_indices is not None:
            candidate_indices = subsample_indices[candidate_indices]
        chosen_original_indices = [cluster_object_indices[i] for i in candidate_indices]

        return {
            "exemplars": chosen_exemplars,
            "exemplar_original_indices": chosen_original_indices,
        }  # keys have to match outputs


def submodular_selection_exemplars(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    object_vectors: np.ndarray,
    n_exemplars: int = 4,
    object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
    submodular_function: str = "facility_location",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Generates a list of exemplar text for each cluster in a cluster layer.
    These exemplars are selected by submodular optimization over the object vectors,
    which are centred on the mean of all object vectors before selection.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    object_vectors : np.ndarray
        An ndarray of topic vectors for each object.
    n_exemplars : int, optional
        The number of exemplars to sample for each cluster, by default 4.
    object_to_text_function: Callable[List[Any], List[str]]
        A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
    submodular_function : str, optional
        The sampling method for selecting exemplars 'facility_location' or 'saturated_coverage', by default it is 'facility_location'.
    verbose : bool, optional
        Whether to show a progress bar; takes precedence over ``show_progress_bar``.
    show_progress_bar : bool, optional
        Deprecated, use ``verbose``. Whether to show a progress bar, by default False.

    Returns
    -------
    Tuple[List[List[str]], List[List[int]]]
        A tuple containing:
        - A list of lists of exemplar texts for each cluster
        - A list of lists of indices indicating the position of each exemplar in the original object list
    """
    results = []
    indices = []

    submodular_annotator = SubmodularExemplarAnnotator(
        n_exemplars=n_exemplars,
        object_to_text_function=object_to_text_function,
        null_topic=np.mean(object_vectors, axis=0),
        submodular_function=submodular_function,
    )

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc=f"Selecting {submodular_function} exemplars",
        disable=not show_progress_bar_val,
        unit="cluster",
        leave=False,
        position=1,
    ):

        cluster_mask = cluster_label_vector == cluster_num
        original_indices = np.where(cluster_mask)[0]
        cluster_objects = [objects[i] for i in original_indices]
        cluster_object_vectors = object_vectors[cluster_mask]

        # TODO: layer=0 is a placeholder. Once integrated with Executor and full tree context,
        # this will the be node object that is passed to the annotator.
        node = NodeId(layer=0, cluster=cluster_num)

        result = submodular_annotator.annotate(
            node=node,
            cluster_objects=cluster_objects,
            cluster_object_vectors=cluster_object_vectors,
            cluster_object_indices=original_indices,
        )
        chosen_exemplars = result["exemplars"]
        chosen_original_indices = result["exemplar_original_indices"]

        results.append(chosen_exemplars)
        indices.append(chosen_original_indices)

    return results, indices


class RandomExemplarAnnotator:
    """Select a node's exemplars uniformly at random.

    Draws ``n_exemplars`` objects without replacement from the node's cluster. The
    draw uses NumPy's global random state, so call ``np.random.seed`` beforehand for
    reproducible results.

    Parameters
    ----------
    n_exemplars : int, optional
        The number of exemplars to select for each node, by default 4. A node with
        fewer objects returns all of them.
    object_to_text_function : Callable[[List[Any]], List[str]], optional
        A function that turns the chosen objects into exemplar strings. If None,
        the objects are returned as they are.

    Attributes
    ----------
    inputs : tuple of str
        ``cluster_objects`` and ``cluster_object_indices``.
    outputs : tuple of str
        ``exemplars`` and ``exemplar_original_indices``.
    algorithm_type : str
        ``"node-node"``: each node is annotated from its own inputs.
    """

    inputs = ("cluster_objects", "cluster_object_indices")
    outputs = ("exemplars", "exemplar_original_indices")
    algorithm_type = "node-node"

    def __init__(
        self,
        n_exemplars: int = 4,
        object_to_text_function=None,
    ):
        self.n_exemplars = n_exemplars
        self.object_to_text_function = object_to_text_function

    def annotate(
        self,
        node,
        *,
        cluster_objects,
        cluster_object_indices,
    ):
        """Select exemplars for one node.

        Parameters
        ----------
        node : NodeId
            The node being annotated. It is not used in the selection.
        cluster_objects : Sequence
            The objects in the node's cluster.
        cluster_object_indices : Sequence[int]
            The position in the corpus of each entry of ``cluster_objects``.

        Returns
        -------
        dict
            ``{"exemplars": ..., "exemplar_original_indices": ...}``, the chosen
            exemplars (as strings if ``object_to_text_function`` is given) and their
            positions in the corpus, in the order drawn. Both are empty for a node
            with no objects.
        """
        if len(cluster_objects) == 0:
            return {"exemplars": [], "exemplar_original_indices": []}

        # Random permutation of indices within this cluster
        exemplar_order = np.random.permutation(len(cluster_objects))[: self.n_exemplars]
        # Extract selected exemplar objects
        chosen_exemplars = [cluster_objects[i] for i in exemplar_order]
        # Apply text transformation if provided
        if self.object_to_text_function is not None:
            chosen_exemplars = self.object_to_text_function(chosen_exemplars)

        chosen_original_indices = [cluster_object_indices[i] for i in exemplar_order]

        return {
            "exemplars": chosen_exemplars,
            "exemplar_original_indices": chosen_original_indices,
        }  # keys have to match outputs


def random_exemplars(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    n_exemplars: int = 4,
    object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Generates a list of exemplar texts for each cluster in a cluster layer.
    These exemplars are randomly sampled from each cluster.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.  Cluster labels below zero are ignored and cluster labels are
        explected to be integers ranging from 0 to the number_of_clusters.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    n_exemplars : int, optional
        The number of exemplars to sample for each cluster, by default 4.
    object_to_text_function: Callable[List[Any], List[str]]
        A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
    verbose : bool, optional
        Whether to show a progress bar; takes precedence over ``show_progress_bar``.
    show_progress_bar : bool, optional
        Deprecated, use ``verbose``. Whether to show a progress bar, by default False.

    Returns
    -------
    Tuple[List[List[str]], List[List[int]]]
        A tuple containing:
        - A list of lists of exemplar texts for each cluster
        - A list of lists of indices indicating the position of each exemplar in the original object list
    """
    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    results = []
    indices = []

    random_annotator = RandomExemplarAnnotator(
        n_exemplars=n_exemplars,
        object_to_text_function=object_to_text_function,
    )

    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting random exemplars",
        disable=not show_progress_bar_val,
        unit="cluster",
        leave=False,
        position=1,
    ):
        # Get mask for current cluster
        cluster_mask = cluster_label_vector == cluster_num

        # Store original indices for this cluster
        original_indices = np.where(cluster_mask)[0]

        # Index objects by integer position — no np.array(objects) needed
        cluster_objects = [objects[i] for i in original_indices]

        # TODO: layer=0 is a placeholder. Once integrated with Executor and full tree context,
        # this will the be node object that is passed to the annotator.
        node = NodeId(layer=0, cluster=cluster_num)

        result = random_annotator.annotate(
            node=node,
            cluster_objects=cluster_objects,
            cluster_object_indices=original_indices,
        )
        chosen_exemplars = result["exemplars"]
        chosen_original_indices = result["exemplar_original_indices"]

        results.append(chosen_exemplars)
        indices.append(chosen_original_indices)

    return results, indices


def diverse_exemplars_by_cluster(
    cluster_objects: List[str],
    cluster_centroid: np.ndarray,
    cluster_object_vectors: np.ndarray,
    null_topic: np.ndarray,
    n_exemplars: int = 4,
    diversify_alpha: float = 1.0,
    object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
    method: str = "centroid",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> Tuple[List[str], List[int], List[int]]:
    """Select exemplars for a single cluster.

    Candidates are ranked by cosine distance to the centroid (``method="centroid"``)
    or randomly (``method="random"``), then thinned for diversity. The centroid and
    the object vectors are both centred by ``null_topic`` before any distance is
    computed, so pass them uncentred.

    Parameters
    ----------
    cluster_objects : List[str]
        The objects in the cluster.
    cluster_centroid : np.ndarray
        The centroid vector of the cluster.
    cluster_object_vectors : np.ndarray
        One vector per entry of ``cluster_objects``.
    null_topic : np.ndarray
        Background vector subtracted from the centroid and the object vectors,
        typically the mean of all object vectors in the corpus.
    n_exemplars : int, optional
        The number of exemplars to select, by default 4.
    diversify_alpha : float, optional
        The largest diversification alpha to try; the largest alpha up to this value
        that still retains ``n_exemplars`` candidates is used. By default 1.0.
    object_to_text_function: Callable[List[Any], List[str]], optional
        A function which takes a list of objects and returns exemplar strings, by default
        for strings it is lambda x: x. If None, the objects are returned as they are.
    method : str, optional
        How to rank candidates, 'centroid' or 'random', by default 'centroid'.
    verbose, show_progress_bar : bool, optional
        Unused; accepted for consistency with the other exemplar functions.

    Returns
    -------
    Tuple[List[str], np.ndarray, List[int]]
        - the chosen exemplar texts
        - ``exemplar_order``: positions within ``cluster_objects``, ranked by ``method``
        - ``chosen_indices``: positions of the chosen exemplars within the top-ranked
          candidates, so chosen exemplar ``i`` is at position
          ``exemplar_order[chosen_indices[i]]`` of ``cluster_objects``

        An empty cluster returns three empty lists.
    """

    # If there is an empty cluster emit empty lists
    if len(cluster_objects) == 0:
        return [], [], []

    centered_centroid = cluster_centroid - null_topic
    centered_vectors = cluster_object_vectors - null_topic

    if method == "centroid":
        # Select the central exemplars as the objects to each centroid
        exemplar_distances = pairwise_distances(
            centered_centroid.reshape(1, -1),
            centered_vectors,
            metric="cosine",
        )
        exemplar_order = np.argsort(exemplar_distances.flatten())
    elif method == "random":
        exemplar_order = np.random.permutation(len(cluster_objects))
    else:
        raise ValueError(
            f"method={method} is not a valid selection. Please choose one of (centroid,random)"
        )

    # We need more exemplars than we want in case we drop some via diversify
    n_exemplars_to_take = max((n_exemplars**2), 16)
    exemplar_candidates = [
        cluster_objects[i] for i in exemplar_order[:n_exemplars_to_take]
    ]
    candidate_vectors = np.asarray(
        [centered_vectors[i] for i in exemplar_order[:n_exemplars_to_take]]
    )

    chosen_indices = diversify(
        centered_centroid,
        candidate_vectors,
        n_exemplars,
        max_alpha=diversify_alpha,
        tolerance=0.01,
    )[:n_exemplars]

    if object_to_text_function is None:
        chosen_exemplars = [exemplar_candidates[i] for i in chosen_indices]
    else:
        chosen_exemplars = object_to_text_function(
            [exemplar_candidates[i] for i in chosen_indices]
        )

    return chosen_exemplars, exemplar_order, chosen_indices


class DiverseExemplarAnnotator:
    """Select a node's exemplars closest to its centroid, thinned for diversity.

    Objects are ranked by cosine distance to the node's centroid (or randomly, if
    ``method="random"``), and a diverse subset of the top candidates is kept. The
    centroid and object vectors are both centred by ``null_topic`` before any
    distance is computed, so the annotation inputs should be uncentred.

    Parameters
    ----------
    n_exemplars : int, optional
        The number of exemplars to select for each node, by default 4.
    diversify_alpha : float, optional
        The largest diversification alpha to try. The largest alpha up to this
        value that still retains ``n_exemplars`` candidates is used. By default 1.0.
    object_to_text_function : Callable[[List[Any]], List[str]], optional
        A function that turns the chosen objects into exemplar strings. If None,
        the objects are returned as they are.
    null_topic : np.ndarray, optional
        A background vector, with the same dimension as the object vectors, that is
        subtracted from the centroid and the object vectors. This is typically the
        mean of all object vectors in the corpus. If None, nothing is subtracted.
    method : {"centroid", "random"}, optional
        How candidates are ranked before diversification, by default "centroid".

    Attributes
    ----------
    inputs : tuple of str
        ``cluster_objects``, ``cluster_centroid``, ``cluster_object_vectors`` and
        ``cluster_object_indices``.
    outputs : tuple of str
        ``exemplars`` and ``exemplar_original_indices``.
    algorithm_type : str
        ``"node-node"``: each node is annotated from its own inputs.

    Raises
    ------
    ValueError
        At annotation time, if ``method`` is not one of the supported values.
    """

    inputs = (
        "cluster_objects",
        "cluster_centroid",
        "cluster_object_vectors",
        "cluster_object_indices",
    )
    outputs = ("exemplars", "exemplar_original_indices")
    algorithm_type = "node-node"

    def __init__(
        self,
        n_exemplars: int = 4,
        diversify_alpha: float = 1.0,
        object_to_text_function=None,
        null_topic: np.ndarray = None,
        method="centroid",
    ):
        self.n_exemplars = n_exemplars
        self.diversify_alpha = diversify_alpha
        self.object_to_text_function = object_to_text_function
        self.null_topic = null_topic  # subtracted from vectors; None means no centering
        self.method = method

    def annotate(
        self,
        node,
        *,
        cluster_objects,
        cluster_centroid,
        cluster_object_vectors,
        cluster_object_indices,  # has to match inputs name
    ):
        """Select exemplars for one node.

        Parameters
        ----------
        node : NodeId
            The node being annotated. It is not used in the selection.
        cluster_objects : Sequence
            The objects in the node's cluster.
        cluster_centroid : np.ndarray
            The node's centroid vector, uncentred.
        cluster_object_vectors : np.ndarray
            One uncentred vector for each entry of ``cluster_objects``.
        cluster_object_indices : Sequence[int]
            The position in the corpus of each entry of ``cluster_objects``.

        Returns
        -------
        dict
            ``{"exemplars": ..., "exemplar_original_indices": ...}``, the chosen
            exemplars (as strings if ``object_to_text_function`` is given) and their
            positions in the corpus, closest to the centroid first when
            ``method="centroid"``. Both are empty for a node with no objects.
        """
        null_topic = self.null_topic
        if null_topic is None:
            null_topic = np.zeros_like(cluster_centroid)

        node_result = diverse_exemplars_by_cluster(
            cluster_objects=cluster_objects,
            cluster_centroid=cluster_centroid,
            cluster_object_vectors=cluster_object_vectors,
            null_topic=null_topic,
            n_exemplars=self.n_exemplars,
            diversify_alpha=self.diversify_alpha,
            object_to_text_function=self.object_to_text_function,
            method=self.method,
        )
        chosen_exemplars, exemplar_order, chosen_indices = node_result
        chosen_original_indices = [
            cluster_object_indices[exemplar_order[i]] for i in chosen_indices
        ]
        return {
            "exemplars": chosen_exemplars,
            "exemplar_original_indices": chosen_original_indices,
        }  # keys have to match outputs


def diverse_exemplars(
    cluster_label_vector: np.ndarray,
    objects: List[str],
    object_vectors: np.ndarray,
    centroid_vectors: np.ndarray,
    n_exemplars: int = 4,
    diversify_alpha: float = 1.0,
    object_to_text_function: Callable[List[Any], List[str]] = lambda x: x,
    method: str = "centroid",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Generates a list of exemplar text for each cluster in a cluster layer.
    These exemplars are selected to be the closest vectors to the cluster centroid while retaining
    sufficient diversity. Object vectors and centroids are centred on the mean of all
    object vectors before distances are computed.

    Parameters
    ----------
    cluster_label_vector : np.ndarray
        A vector of cluster labels for each object.
    objects : List[str]
        A list of objects; these are text objects a sample of which are returned as exemplars for each cluster.
    object_vectors : np.ndarray
        An ndarray of topic vectors for each object.
    centroid_vectors : np.ndarray
        An ndarray of centroid vectors for each cluster.
    n_exemplars : int, optional
        The number of exemplars to sample for each cluster, by default 4.
    diversify_alpha : float, optional
        The largest diversification alpha to try, by default 1.0.
    object_to_text_function: Callable[List[Any], List[str]]
        A function which takes an object and returns an exemplar string, by default for strings it is lambda x: x
    method : str, optional
        The sampling method for selecting exemplars 'centroid' or 'random', by default it is 'centroid'.
    verbose : bool, optional
        Whether to show a progress bar; takes precedence over ``show_progress_bar``.
    show_progress_bar : bool, optional
        Deprecated, use ``verbose``. Whether to show a progress bar, by default False.

    Returns
    -------
    Tuple[List[List[str]], List[List[int]]]
        A tuple containing:
        - A list of lists of exemplar texts for each cluster
        - A list of lists of indices indicating the position of each exemplar in the original object list
    """
    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    results = []
    indices = []

    diverse_annotator = DiverseExemplarAnnotator(
        n_exemplars=n_exemplars,
        diversify_alpha=diversify_alpha,
        object_to_text_function=object_to_text_function,
        null_topic=np.mean(object_vectors, axis=0),
        method=method,
    )

    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting central exemplars",
        disable=not show_progress_bar_val,
        unit="cluster",
        leave=False,
        position=1,
    ):
        cluster_mask = cluster_label_vector == cluster_num
        original_indices = np.where(cluster_mask)[0]
        cluster_objects = [objects[i] for i in original_indices]
        cluster_centroid = centroid_vectors[cluster_num]
        cluster_object_vectors = object_vectors[cluster_mask]

        # TODO: layer=0 is a placeholder. Once integrated with Executor and full tree context,
        # this will the be node object that is passed to the annotator.
        node = NodeId(layer=0, cluster=cluster_num)

        result = diverse_annotator.annotate(
            node=node,
            cluster_objects=cluster_objects,
            cluster_centroid=cluster_centroid,
            cluster_object_vectors=cluster_object_vectors,
            cluster_object_indices=original_indices,
        )
        chosen_exemplars = result["exemplars"]
        chosen_original_indices = result["exemplar_original_indices"]

        results.append(chosen_exemplars)
        indices.append(chosen_original_indices)

    return results, indices


def exemplar_annotator(
    method: str,
    n_exemplars: int,
    object_to_text_function: Callable[[List[Any]], List[str]],
    diversify_alpha: float,
    null_topic: np.ndarray,
):
    """Returns the exemplar annotator for a ``Toponymy.fit`` exemplar method name."""
    if method == "central":
        return DiverseExemplarAnnotator(
            n_exemplars=n_exemplars,
            diversify_alpha=diversify_alpha,
            object_to_text_function=object_to_text_function,
            null_topic=null_topic,
        )
    if method in ("facility_location", "saturated_coverage"):
        return SubmodularExemplarAnnotator(
            n_exemplars=n_exemplars,
            object_to_text_function=object_to_text_function,
            null_topic=null_topic,
            submodular_function=method,
        )
    if method == "random":
        return RandomExemplarAnnotator(
            n_exemplars=n_exemplars, object_to_text_function=object_to_text_function
        )
    raise ValueError(f"Unknown exemplar generation method: {method}. Use 'central'.")
