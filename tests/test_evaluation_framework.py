import json
import unittest

from benchmark.evaluation_result import EvaluationResult


class EvaluationResultTests(unittest.TestCase):
    def test_to_dict_and_round_trip(self):
        result = EvaluationResult(
            paper_id="paper_0001",
            question_id=1,
            question="What is the seed lexicon?",
            gold_answer="a vocabulary of positive and negative predicates",
            model_answer="a vocabulary of positive and negative predicates",
            evidence=["The seed lexicon consists of positive and negative predicates."],
            paper_length=1200,
            execution_time=12.34,
            ram_usage=18.5,
            mode="direct",
            model="granite",
        )

        payload = result.to_dict()
        self.assertIn("paper_id", payload)
        self.assertIn("question_id", payload)
        self.assertIn("question", payload)
        self.assertIn("gold_answer", payload)
        self.assertIn("model_answer", payload)
        self.assertIn("evidence", payload)
        self.assertIn("paper_length", payload)
        self.assertIn("execution_time", payload)
        self.assertIn("ram_usage", payload)
        self.assertIn("mode", payload)
        self.assertIn("model", payload)

        restored = EvaluationResult.from_dict(payload)
        self.assertEqual(restored.paper_id, "paper_0001")
        self.assertEqual(restored.model, "granite")
        self.assertEqual(restored.mode, "direct")


if __name__ == "__main__":
    unittest.main()
