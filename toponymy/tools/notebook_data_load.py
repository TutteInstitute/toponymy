import atexit
import tempfile
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from toponymy.tools.notebook_test_helpers import notebook_test_replacement, examples_dir

logger = logging.getLogger(__name__)

HF_URL_NEWSGROUPS = (
    "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
)
HF_URL_ARXIV_CT = (
    "hf://datasets/lmcinnes/arxiv_category_theory/data/train-00000-of-00001.parquet"
)
HF_URL_ARXIV_ML = (
    "hf://datasets/lmcinnes/arxiv_ml/data/train-00000-of-00008-f3c9b137f969d545.parquet"
)

_output_dir_cache = None


def _test_output_dir() -> Path:
    """
    Retrieve or create a temporary directory for notebook test outputs.

    Checks for the NB_TEST_OUTPUT_DIR environment variable (set by pytest fixtures);
    if not found, creates and caches a temporary directory with cleanup on exit.

    Returns
    -------
    Path
        Path object pointing to the designated test output directory.
    """
    # pytest temp dir fixture set using NB_TEST_OUTPUT_DIR environment variable
    if "NB_TEST_OUTPUT_DIR" in os.environ:
        return Path(os.environ["NB_TEST_OUTPUT_DIR"])
    # manual fallback for non-pytest runs
    global _output_dir_cache
    if _output_dir_cache is None:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        _output_dir_cache = Path(tmp.name)
    logger.info(f"Using manual fallback to {_output_dir_cache}")
    return _output_dir_cache


@notebook_test_replacement(_test_output_dir)
def notebook_output_dir() -> Path:
    """
    Get the output directory for notebook operations.

    During example notebook tests, this returns a temporary directory via the decorator replacement.
    In normal execution, returns the current working directory.

    Returns
    -------
    Path
        The working directory path. Under NOTEBOOK_TESTING=true, replaced with _test_output_dir().
    """
    return Path().resolve()


def _load_newsgroups(use_small: bool = False) -> pd.DataFrame:
    """
    Load the 20 newsgroups dataset with optional size reduction.

    Parameters
    ----------
    use_small : bool, optional
        If True, load a small pre-extracted subset (150 documents) from a bundled parquet file.
        If False, load the full dataset from Hugging Face Hub. Default is False.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the 20 newsgroups documents and embeddings.
    """

    if use_small:
        # Equivalent to df.sample(n=150, random_state=33).reset_index(drop=True)
        # Use bundled version for PR testing to avoid network dependency
        return pd.read_parquet(examples_dir() / "20newsgroups_embedded_150.parquet")
    df = pd.read_parquet(HF_URL_NEWSGROUPS)
    return df


def load_small_newsgroups(use_small: bool = True) -> pd.DataFrame:
    """
    Load a smaller subset of the 20 newsgroups dataset for faster notebook tests.

    This function is typically used as a replacement for load_newsgroups() in test environments
    via the @notebook_test_replacement decorator. The use_small parameter can override the
    default test-mode behavior to return the full dataset if needed.

    Parameters
    ----------
    use_small : bool, optional
        If True, load the small bundled subset. If False, load the full dataset. Default is True.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the 20 newsgroups documents and embeddings.
    """
    return _load_newsgroups(use_small=use_small)


@notebook_test_replacement(load_small_newsgroups)
def load_newsgroups(use_small: bool = False) -> pd.DataFrame:
    """
    Load the 20 newsgroups dataset from Hugging Face Hub.

    During example notebook tests, this is replaced with load_small_newsgroups() via decorator.

    Parameters
    ----------
    use_small : bool, optional
        If True, load a reduced subset. If False, load the full dataset. Default is False.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the 20 newsgroups documents and embeddings.
    """
    return _load_newsgroups(use_small=use_small)


def _load_bundled_arxiv(
    use_small: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load bundled arXiv computer science papers with embeddings and UMAP coordinates.

    Parameters
    ----------
    use_small : bool, optional
        If True, load only the first 350 documents (respecting min_cluster_size=4).
        If False, load all bundled documents. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of (documents, document_vectors, clusterable_vectors) where:
        - documents: 1D array of strings (title + abstract)
        - document_vectors: 2D array of shape (n_docs, embedding_dim)
        - clusterable_vectors: 2D array of shape (n_docs, 2) [UMAP coordinates]
    """
    base_dir = examples_dir()
    docs_df = pd.read_csv(base_dir / "ai_arxiv_papers.zip")
    document_vectors = np.load(base_dir / "ai_arxiv_vectors.npy")
    clusterable_vectors = np.load(base_dir / "ai_arxiv_coordinates.npz.npy")

    if use_small:
        # limited by min_cluster_size=4
        N = 350

        docs_df = docs_df[:N]
        document_vectors = document_vectors[:N]
        clusterable_vectors = clusterable_vectors[:N]

    documents = (
        docs_df["title"].str.strip() + "\n\n" + docs_df["abstract"].str.strip()
    ).to_numpy()

    return documents, document_vectors, clusterable_vectors


def load_small_bundled_arxiv(
    use_small: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a smaller subset of arXiv papers for faster notebook tests.

    This function is typically used as a replacement for load_bundled_arxiv() in test environments
    via the @notebook_test_replacement decorator. The use_small parameter can override the
    default test-mode behavior to return the full dataset if needed.

    Parameters
    ----------
    use_small : bool, optional
        If True, load only the first 350 documents. If False, load all bundled documents. Default is True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of (documents, document_vectors, clusterable_vectors).
    """
    return _load_bundled_arxiv(use_small=use_small)


@notebook_test_replacement(load_small_bundled_arxiv)
def load_bundled_arxiv(
    use_small: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load bundled arXiv computer science papers from disk.

    During example notebook tests, this is replaced with load_small_bundled_arxiv() via decorator.

    Parameters
    ----------
    use_small : bool, optional
        If True, load a reduced subset (first 350 docs). If False, load all documents. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of (documents, document_vectors, clusterable_vectors).
    """
    return _load_bundled_arxiv(use_small=use_small)


def _load_arxiv_ct(use_small=False):
    df = pd.read_parquet(HF_URL_ARXIV_CT)

    if use_small:
        ## Needs 4 cluster layers in how_toponymy_works notebook
        df = df.sample(n=3000, random_state=99).reset_index(drop=True)

    return df


def load_small_arxiv_ct(use_small=True):
    """
    Helper to load a smaller subset of the arxiv category theory dataset in lieu of load_arxiv_ct when
    running example notebook tests. Can be overrideen via use_small to give the full
    arxiv ct dataset even after replacement.
    """
    return _load_arxiv_ct(use_small=use_small)


@notebook_test_replacement(load_small_arxiv_ct)
def load_arxiv_ct(use_small=False):
    return _load_arxiv_ct(use_small=use_small)


def _load_arxiv_ml(use_small=False):
    df = pd.read_parquet(HF_URL_ARXIV_ML)

    if use_small:
        # needs at least 129 keyphrases -> keyphrases[128] in keyphrases.ipynb
        return df.sample(n=5000, random_state=2).reset_index(drop=True)
    else:
        return df


def load_small_arxiv_ml(use_small=True):
    """
    Helper to load a smaller subset of the arxiv machine learning dataset in lieu of load_arxiv_ct when
    running example notebook tests. Can be overrideen via use_small to give the full
    arxiv ml dataset even after replacement.
    """
    return _load_arxiv_ml(use_small=use_small)


@notebook_test_replacement(load_small_arxiv_ml)
def load_arxiv_ml(use_small=False):
    return _load_arxiv_ml(use_small=use_small)
