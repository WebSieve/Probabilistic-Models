from .data import dataClass, DATA_PATH, STOPWORDS
from .model import DCM
from .eval import Evaluator

__all__ = [
    "dataClass",
    "DATA_PATH",
    "STOPWORDS",
    "DCM",
    "Evaluator",
    "main",
]


def main() -> None:
    print("dirichlet-compound-multinomial: DCM on Bag of Words / Document Classification")
    print(f"  exports: {', '.join(__all__)}")