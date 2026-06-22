import pandas as pd
import numpy as np
from .notebook_test_helpers import notebook_test_replacement, examples_dir


def _load_newsgroups(use_small=False):
    df = pd.read_parquet(
        "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
    )

    if use_small:
        return df.sample(n=250, random_state=0).reset_index(drop=True)

    return df


def load_small_newsgroups(use_small=True):
    """
    Helper to load a smaller subset of the newsgroups datasets in lieu of load_newsgroups when
    running example notebook tests. Can be overrideen via use_small to give the full
    newsgroups dataset even after replacement.
    """
    return _load_newsgroups(use_small=use_small)


@notebook_test_replacement(load_small_newsgroups)
def load_newsgroups(use_small=False):
    return _load_newsgroups(use_small=use_small)


def _load_bundled_arxiv(use_small=False):
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


def load_small_arxiv(use_small=True):
    """
    Helper to load a smaller subset of the arxiv dataset in lieu of load_bundled_arxiv when
    running example notebook tests. Can be overrideen via use_small to give the full
    arxiv dataset even after replacement.
    """
    return _load_bundled_arxiv(use_small=use_small)


@notebook_test_replacement(load_small_arxiv)
def load_bundled_arxiv(use_small=False):
    return _load_bundled_arxiv(use_small=use_small)


def _load_arxiv_ct(use_small=False):
    df = pd.read_parquet(
        "hf://datasets/lmcinnes/arxiv_category_theory/data/train-00000-of-00001.parquet"
    )

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
    df = pd.read_parquet(
        "hf://datasets/lmcinnes/arxiv_ml/data/train-00000-of-00008-f3c9b137f969d545.parquet"
    )

    if use_small:
        # needs at least 129 keyphrases -> keyphrases[128] in keyphrases.ipynb
        return df.sample(n=5500, random_state=42).reset_index(drop=True)
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
