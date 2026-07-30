"""
evaluation/__init__.py — Public surface of the evaluation package.

    from evaluation import EvaluationPipeline, evaluate
    from evaluation import ArtifactWriter
    from evaluation import find_best_checkpoint, load_eval_model
"""

from evaluation.artifacts import ArtifactWriter
from evaluation.evaluator import EvaluationPipeline, evaluate
from evaluation.loader import find_best_checkpoint, load_eval_model

__all__ = [
    "EvaluationPipeline",
    "evaluate",
    "ArtifactWriter",
    "find_best_checkpoint",
    "load_eval_model",
]
