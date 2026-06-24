# This code is a modified version of the apricot library's facility location
# facilityLocation.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from apricot.functions.base import BaseGraphSelection
from apricot.optimizers import LazyGreedy
from apricot.optimizers import ApproximateLazyGreedy
from apricot.optimizers import SieveGreedy
import numba
import numpy as np

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
