import math
from typing import Literal, TypedDict

AnswerValue = Literal["yes", "no", "not_sure"]


class QuestionKnowledge(TypedDict):
    question_id: str
    text: str
    support: dict[str, dict[AnswerValue, float]]


def select_next_question(questions: list[QuestionKnowledge], distribution: dict[str, float], answered_ids: set[str]) -> QuestionKnowledge | None:
    unanswered = [question for question in questions if question["question_id"] not in answered_ids]
    if not unanswered:
        return None
    return max(unanswered, key=lambda question: information_gain(question, distribution))


def information_gain(question: QuestionKnowledge, distribution: dict[str, float]) -> float:
    prior_entropy = _entropy(distribution)
    expected_entropy = 0.0
    for answer in ("yes", "no", "not_sure"):
        probability, posterior = _posterior(question, distribution, answer)
        expected_entropy += probability * _entropy(posterior)
    return prior_entropy - expected_entropy


def _posterior(question: QuestionKnowledge, distribution: dict[str, float], answer: AnswerValue) -> tuple[float, dict[str, float]]:
    weighted = {
        disease: prior * question["support"].get(disease, {}).get(answer, 0.5)
        for disease, prior in distribution.items()
    }
    total = sum(weighted.values())
    if total <= 0:
        return 0.0, {}
    return total, {disease: value / total for disease, value in weighted.items()}


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(probability * math.log2(probability) for probability in distribution.values() if probability > 0)
