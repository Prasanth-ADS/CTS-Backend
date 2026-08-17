import unittest

from app.reasoning.info_gain import information_gain, select_next_question


class InfoGainTests(unittest.TestCase):
    def test_selects_unanswered_question_with_highest_information_gain(self):
        distribution = {"A": 0.5, "B": 0.5}
        questions = [
            {
                "question_id": "low_gain",
                "text": "Low gain?",
                "support": {"A": {"yes": 0.5, "no": 0.5, "not_sure": 0.5}, "B": {"yes": 0.5, "no": 0.5, "not_sure": 0.5}},
            },
            {
                "question_id": "high_gain",
                "text": "High gain?",
                "support": {"A": {"yes": 0.9, "no": 0.1, "not_sure": 0.5}, "B": {"yes": 0.1, "no": 0.9, "not_sure": 0.5}},
            },
        ]

        selected = select_next_question(questions, distribution, set())

        self.assertEqual(selected["question_id"], "high_gain")
        self.assertGreater(information_gain(questions[1], distribution), information_gain(questions[0], distribution))

    def test_ignores_answered_questions(self):
        distribution = {"A": 0.5, "B": 0.5}
        questions = [
            {"question_id": "answered", "text": "Answered?", "support": {}},
            {"question_id": "next", "text": "Next?", "support": {}},
        ]

        selected = select_next_question(questions, distribution, {"answered"})

        self.assertEqual(selected["question_id"], "next")
