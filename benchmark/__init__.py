from .evaluation_result import EvaluationResult


def evaluate_question(*args, **kwargs):
    from .evaluation import evaluate_question as _evaluate_question

    return _evaluate_question(*args, **kwargs)


def save_evaluation_result(*args, **kwargs):
    from .evaluation import save_evaluation_result as _save_evaluation_result

    return _save_evaluation_result(*args, **kwargs)


__all__ = ["EvaluationResult", "evaluate_question", "save_evaluation_result"]
