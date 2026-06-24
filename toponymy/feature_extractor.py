from abc import ABC, abstractmethod
from typing import Any, List

import numba
import numpy as np
from sklearn.base import BaseEstimator

from toponymy.new_clustering import Clusterer


@numba.njit()
def centroids_from_labels(
    cluster_labels: np.ndarray, vector_data: np.ndarray
) -> np.ndarray:  # pragma: no cover
    result = np.zeros((cluster_labels.max() + 1, vector_data.shape[1]))
    counts = np.zeros(cluster_labels.max() + 1)
    for i in range(cluster_labels.shape[0]):
        cluster_num = cluster_labels[i]
        if cluster_num >= 0:
            result[cluster_num] += vector_data[i]
            counts[cluster_num] += 1

    for i in range(result.shape[0]):
        if counts[i] > 0:
            result[i] /= counts[i]

    return result


class FeatureExtractorBase(ABC, BaseEstimator):
    """
    Abstract base class for a feature extractor.

    A feature extractor is a class that can build features for objects
    and then extract features to represent clusters.
    """

    def __init__(self, *args, **kwargs):
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def can_fit_from_objects(self) -> bool:
        """
        If True, enables the FeatureExtractor to be fitted on the fly.

        If False, specifies that the FeatureExtractor must be pre-fitted.
        """
        return False

    @abstractmethod
    def fit(self, objects: List[Any], *args, **kwargs):
        """
        An abstract method to fit a collection of features to a set of objects.

        Must be defined in any subclass.
        """
        raise NotImplemented

    @abstractmethod
    def get_cluster_features(
        self,
        clusterer: Clusterer,
        layer_id: int,
        *args,
        **kwargs,
    ) -> List[List[str]]:
        """
        An abstract method to get features as a representation for each cluster.

        Must be defined in any subclass.
        """
        raise NotImplemented

    def predict(
        self,
        clusterer: Clusterer,
        layer_id: int,
        *args,
        **kwargs,
    ) -> List[List[str]]:
        """
        A method to get features as a representation for each cluster.

        Checks that the feature extractor is fitted, and then runs `get_cluster_features`.
        """
        return self.get_cluster_features(cluster_indices, layer_id, *args, **kwargs)


class TreeShapFeatureSelector(FeatureExtractorBase):
    """
    One-vs-all Random Forest + TreeSHAP feature selector for cluster analysis.
 
    Trains a binary classifier (cluster_id vs. all others) on a tabular
    dataset and ranks features by their mean absolute TreeSHAP value.
 
    Parameters
    ----------
    dataset      : file path, DataFrame, or None (supply later at call time)
    rf_params    : dict of RandomForestClassifier kwargs (None = use defaults)
    test_size    : held-out fraction for evaluation  (default 0.20)
    random_state : master random seed  (default 42)
    verbose      : print progress/metrics by default  (default True)
    """
 
    def __init__(
        self,
        dataset: Union[str, Path, pd.DataFrame, None] = None,
        rf_params: dict | None = None,
        test_size: float = _DEFAULT_TEST_SIZE,
        random_state: int = _DEFAULT_RANDOM_STATE,
        verbose: bool = True,
    ) -> None:
        self._dataset = dataset
        self._rf_params = {**_DEFAULT_RF_PARAMS, **(rf_params or {})}
        self._test_size = test_size
        self._random_state = random_state
        self._verbose = verbose
 
        # Public post-fit attributes
        self.clf_: RandomForestClassifier | None = None
        self.feature_names_: list[str] = []
        self.shap_ranking_: pd.DataFrame = pd.DataFrame()
 
    # ── Public API ────────────────────────────────────────────────────────
 
    def get_top_features(
        self,
        cluster_id: int = 0,
        top_k: int = 3,
        dataset: Union[str, Path, pd.DataFrame, None] = None,
        verbose: bool | None = None,
    ) -> list[str]:
        """
        Train a one-vs-all Random Forest for *cluster_id* and return the
        top *top_k* most discriminating feature names ranked by TreeSHAP.
 
        Parameters
        ----------
        cluster_id : which cluster to explain  (default: 0)
        top_k      : number of top features to return  (default: 3)
        dataset    : override the dataset set at construction time
        verbose    : override instance-level verbosity for this call
 
        Returns
        -------
        list[str]  — top feature names, most discriminating first
 
        Examples
        --------
        selector = TreeShapFeatureSelector("data.csv")
        top3 = selector.get_top_features(cluster_id=2)
 
        # Reuse the selector on a different cluster without reloading:
        top5 = selector.get_top_features(cluster_id=4, top_k=5)
        """
        verbose = self._verbose if verbose is None else verbose
        data_source = dataset if dataset is not None else self._dataset
 
        if data_source is None:
            raise ValueError(
                "No dataset provided. Pass a file path or DataFrame to the "
                "constructor or to get_top_features()."
            )
 
        sep = "=" * 65
        self._log(f"\n{sep}", verbose)
        self._log(
            f"  TreeShapFeatureSelector | cluster_id={cluster_id} | top_k={top_k}",
            verbose,
        )
        self._log(sep, verbose)
 
        # 1. Load
        df = self._load_dataframe(data_source)
        self._log(f"  Dataset shape  : {df.shape}", verbose)
 
        cluster_col = self._detect_cluster_col(df)
        self._log(f"  Cluster column : '{cluster_col}'", verbose)
 
        dist = df[cluster_col].value_counts().sort_index()
        self._log(f"\n  Cluster distribution:\n{dist.to_string()}\n", verbose)
 
        # 2. Preprocess
        X_train, X_test, y_train, y_test, feature_names = self._preprocess(
            df, cluster_col, cluster_id, verbose
        )
        self.feature_names_ = feature_names
        self._log(
            f"  Features : {len(feature_names)}  |  "
            f"Train : {len(X_train):,}  |  Test : {len(X_test):,}",
            verbose,
        )
 
        # 3. Train
        self._log(
            f"\n  Training Random Forest (Cluster {cluster_id} vs. Rest) ...",
            verbose,
        )
        self.clf_ = self._train(X_train, y_train)
        self._log(f"  Done — {self.clf_.n_estimators} trees built.", verbose)
 
        # 4. Evaluate
        self._log(f"\n  Evaluation:", verbose)
        self._evaluate(self.clf_, X_test, y_test, cluster_id, verbose)
 
        # 5. TreeSHAP
        self._log(f"\n  TreeSHAP feature ranking:", verbose)
        top_features = self._treeshap_top_k(
            self.clf_, X_test, feature_names, top_k, verbose
        )
 
        # 6. Result
        self._log(f"\n{sep}", verbose)
        self._log(f"  TOP {top_k} FEATURES — Cluster {cluster_id}", verbose)
        self._log(sep, verbose)
        for i, feat in enumerate(top_features, 1):
            self._log(f"  #{i}  {feat}", verbose)
        self._log(sep, verbose)
 
        return top_features
 
    def get_shap_ranking(self) -> pd.DataFrame:
        """
        Return the full SHAP ranking DataFrame from the most recent fit.
 
        Raises RuntimeError if get_top_features() has not been called yet.
 
        Returns
        -------
        pd.DataFrame with columns ["feature", "mean_abs_shap"], sorted
        descending by mean_abs_shap.
        """
        if self.shap_ranking_.empty:
            raise RuntimeError(
                "No SHAP ranking available. Call get_top_features() first."
            )
        return self.shap_ranking_.copy()
 
    # ── Private helpers ───────────────────────────────────────────────────
 
    @staticmethod
    def _log(msg: str, verbose: bool) -> None:
        if verbose:
            print(msg)
 
    @staticmethod
    def _load_dataframe(dataset: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Accept a file path or an already-loaded DataFrame."""
        if isinstance(dataset, pd.DataFrame):
            return dataset.copy()
        path = Path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return pd.read_csv(path, low_memory=False)
 
    @staticmethod
    def _detect_cluster_col(df: pd.DataFrame) -> str:
        """Return the name of the cluster column, or fall back to the last column."""
        match = next(
            (c for c in _CLUSTER_COL_CANDIDATES if c in df.columns), None
        )
        return match if match else df.columns[-1]
 
    def _preprocess(
        self,
        df: pd.DataFrame,
        cluster_col: str,
        cluster_id: int,
        verbose: bool,
    ) -> tuple:
        """
        Build binary labels and a numeric feature matrix.
 
        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        available = sorted(df[cluster_col].unique())
        if cluster_id not in available:
            raise ValueError(
                f"Cluster {cluster_id} not found in column '{cluster_col}'. "
                f"Available IDs: {available}"
            )
 
        y = (df[cluster_col] == cluster_id).astype(int)
        X = df.drop(columns=[cluster_col]).copy()
 
        # Drop constant columns
        constant = [c for c in X.columns if X[c].nunique() <= 1]
        if constant:
            self._log(
                f"  [preprocess] Dropping {len(constant)} constant column(s).",
                verbose,
            )
            X.drop(columns=constant, inplace=True)
 
        # Encode categoricals
        le = LabelEncoder()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = le.fit_transform(X[col].astype(str))
 
        X.fillna(0, inplace=True)
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test, list(X.columns)
 
    def _train(self, X_train, y_train) -> RandomForestClassifier:
        clf = RandomForestClassifier(**self._rf_params)
        clf.fit(X_train, y_train)
        return clf
 
    @staticmethod
    def _evaluate(
        clf: RandomForestClassifier,
        X_test,
        y_test,
        cluster_id: int,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"  ROC-AUC : {auc:.4f}")
        except Exception:
            print("  ROC-AUC : N/A")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Other Clusters", f"Cluster {cluster_id}"],
                digits=4,
                zero_division=0,
            )
        )
 
    @staticmethod
    def _extract_class1_shap(shap_values) -> np.ndarray:
        """
        Normalise shap_values to (n_samples, n_features) for the positive class.
 
        Handles all known SHAP output formats:
            list of 2 arrays  ->  [class0(n,f), class1(n,f)]   older SHAP
            3-D ndarray       ->  (n, f, 2)                    SHAP >= 0.42
            2-D ndarray       ->  (n, f)                       single-output
        """
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1])
        else:
            sv = np.array(shap_values)
            if sv.ndim == 3:          # (n_samples, n_features, n_classes)
                sv = sv[:, :, 1]
 
        sv = sv.squeeze()
        if sv.ndim == 1:              # single test sample edge-case
            sv = sv[np.newaxis, :]
        return sv                     # (n_samples, n_features)
 
    def _treeshap_top_k(
        self,
        clf: RandomForestClassifier,
        X_test: pd.DataFrame,
        feature_names: list[str],
        top_k: int,
        verbose: bool,
    ) -> list[str]:
        """Run TreeSHAP, store full ranking in self.shap_ranking_, return top_k names."""
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
 
        sv = self._extract_class1_shap(shap_values)          # (n, f)
        mean_abs_shap = np.abs(sv).mean(axis=0).flatten()    # (f,)
 
        ranking = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        ranking.index += 1   # rank starts at 1
 
        # Persist full ranking as a public attribute
        self.shap_ranking_ = ranking.reset_index().rename(columns={"index": "rank"})
 
        if verbose:
            preview = min(10, len(ranking))
            print(f"\n  {'Rank':<6} {'Feature':<40} {'Mean |SHAP|':>14}")
            print(f"  {'-' * 62}")
            for rank, row in ranking.head(preview).iterrows():
                marker = "  <" if rank <= top_k else ""
                print(
                    f"  {rank:<6} {row['feature']:<40} "
                    f"{row['mean_abs_shap']:>14.6f}{marker}"
                )
 
        return list(ranking.head(top_k)["feature"])
