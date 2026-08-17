import math
from typing import Literal, TypedDict

AnswerValue = Literal["yes", "no", "not_sure"]
CandidateDistribution = dict[str, float]


class QuestionKnowledge(TypedDict):
    question_id: str
    text: str
    support: dict[str, dict[AnswerValue, float]]


def select_next_question(
    questions: list[QuestionKnowledge],
    candidate_distribution: CandidateDistribution,
    answered_question_ids: set[str],
) -> QuestionKnowledge | None:
    """Select the unanswered question with the highest expected information gain."""
    unanswered_questions = [
        question for question in questions if question["question_id"] not in answered_question_ids
    ]
    if not unanswered_questions:
        return None
    return max(unanswered_questions, key=lambda question: information_gain(question, candidate_distribution))


def information_gain(question: QuestionKnowledge, candidate_distribution: CandidateDistribution) -> float:
    prior_entropy = _entropy(candidate_distribution)
    expected_entropy = 0.0

    for answer in ("yes", "no", "not_sure"):
        answer_probability, posterior = _posterior_for_answer(question, candidate_distribution, answer)
        expected_entropy += answer_probability * _entropy(posterior)

    return prior_entropy - expected_entropy


def _posterior_for_answer(
    question: QuestionKnowledge,
    candidate_distribution: CandidateDistribution,
    answer: AnswerValue,
) -> tuple[float, CandidateDistribution]:
    weighted: CandidateDistribution = {}
    for disease, prior_probability in candidate_distribution.items():
        likelihood = question["support"].get(disease, {}).get(answer, 0.5)
        weighted[disease] = prior_probability * likelihood

    answer_probability = sum(weighted.values())
    if answer_probability <= 0:
        return 0.0, {}
    return answer_probability, {disease: score / answer_probability for disease, score in weighted.items()}


def _entropy(distribution: CandidateDistribution) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in distribution.values()
        if probability > 0
    )
