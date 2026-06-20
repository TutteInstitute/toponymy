import logging
import pandas as pd
import numpy as np
from toponymy.tools.notebook_runner import notebook_test_replacement, examples_dir


def _load_newsgroups(use_small=False):
    df = pd.read_parquet(
        "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
    )

    if use_small:
        ## 602 rows
        df = df[df.newsgroup.str.contains("talk.religion.misc")]

    return df


def load_small_newsgroups(use_small=True):
    """
    Helper to load a smaller subset of the newsgroups datasets in lieu of load_newsgroups when
    running example notebook tests. Can be overrideen via use_small to give the full
    newsgroups dataset even after replacement.
    """
    logging.warning("Using load_small_newsgroups instead of load_newsgroups")
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
        N = 600

        docs_df = docs_df[:N]
        document_vectors = document_vectors[:N]
        clusterable_vectors = clusterable_vectors[:N]

    documents = (
        docs_df["title"].str.strip() + "\n\n" + docs_df["abstract"].str.strip()
    ).to_numpy()

    return documents, document_vectors, clusterable_vectors


def load_small_arxiv(use_small=False):
    """
    Helper to load a smaller subset of the arxiv dataset in lieu of load_bundled_arxiv when
    running example notebook tests. Can be overrideen via use_small to give the full
    newsgroups dataset even after replacement.
    """
    logging.warning("Using small arxiv dataset")
    return _load_bundled_arxiv(use_small=True)


@notebook_test_replacement(load_small_arxiv)
def load_bundled_arxiv(use_small=False):
    return _load_bundled_arxiv(use_small=use_small)
