from .toponymy import Toponymy
from .clustering import ToponymyClusterer, PLSCANClusterer
from .clustering import PrecomputedClusterer, EVoCClusterer, KMeansClusterer
from .feature_extraction import TextExemplarExtractor, TextKeyphraseExtractor
from .serialization import Topic, TopicModel

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("toponymy")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Toponymy",
    "ToponymyClusterer",
    "PLSCANClusterer",
    "KeyphraseBuilder",
    "PrecomputedClusterer",
    "EVoCClusterer",
    "KMeansClusterer",
    "TextExemplarExtractor",
    "TextKeyphraseExtractor",
    "Topic",
    "TopicModel",
]


def __getattr__(name):
    if name == "KeyphraseBuilder":
        from .keyphrases import KeyphraseBuilder

        return KeyphraseBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
