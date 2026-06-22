import unittest

from heritage_perception_analysis.evidence import enforce_verbatim_evidence


class EvidenceTest(unittest.TestCase):
    def test_filters_nonverbatim_evidence(self):
        review = "The arches are elegant, but the ticket price feels high."
        kept, removed = enforce_verbatim_evidence(
            review,
            [
                "The arches are elegant",
                "the ticket price feels high",
                "the architecture is beautiful",
            ],
        )

        self.assertEqual(kept, ["The arches are elegant", "the ticket price feels high"])
        self.assertEqual(removed, ["the architecture is beautiful"])

    def test_deduplicates_kept_evidence(self):
        review = "The courtyard is calm."
        kept, removed = enforce_verbatim_evidence(
            review,
            ["The courtyard is calm", "The courtyard is calm"],
        )

        self.assertEqual(kept, ["The courtyard is calm"])
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
