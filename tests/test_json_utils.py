import unittest

from heritage_perception_analysis.json_utils import coerce_score, parse_json_response


class JsonUtilsTest(unittest.TestCase):
    def test_parse_fenced_json(self):
        data = parse_json_response('```json\n{"score": 8, "reason": "clear"}\n```')
        self.assertEqual(data["score"], 8)

    def test_parse_embedded_json(self):
        data = parse_json_response('Result:\n{"score": 7}')
        self.assertEqual(data["score"], 7)

    def test_coerce_score(self):
        self.assertEqual(coerce_score("8.4"), 8)
        self.assertIsNone(coerce_score("N/A"))
        self.assertIsNone(coerce_score(11))


if __name__ == "__main__":
    unittest.main()
