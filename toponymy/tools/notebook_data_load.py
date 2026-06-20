import logging
import pandas as pd
from toponymy.tools.notebook_runner import notebook_test_replacement


def load_small_newsgroups(use_small=True):
    """
    Helper to load a smaller subset of the newsgroups datasets in lieu of load_newsgroups when
    running example notebook tests. Can be overrideen via testing_fallback to give the full
    newsgroups dataset even after replacement.
    """
    logging.warning("Using load_small_newsgroups instead of load_newsgroups")
    newsgroups_df = pd.read_parquet(
        "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
    )
    if use_small:
        return newsgroups_df[newsgroups_df.newsgroup.str.contains("talk.religion.misc")]
    else:
        return newsgroups_df


@notebook_test_replacement(load_small_newsgroups)
def load_newsgroups(use_small=False):
    return pd.read_parquet(
        "hf://datasets/lmcinnes/20newsgroups_embedded/data/train-00000-of-00001.parquet"
    )
