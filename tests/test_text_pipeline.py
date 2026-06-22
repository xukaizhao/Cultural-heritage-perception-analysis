import unittest

from heritage_perception_analysis.dimensions import DIMENSION_KEYS
from heritage_perception_analysis.prompts import (
    PREDEFINED_TEXT_EXAMPLES,
    VERIFIED_OUTPUT_SCHEMA,
    build_preliminary_extraction_prompt,
)
from heritage_perception_analysis.text_pipeline import (
    TextAnalysisResult,
    TextCulturalValuePipeline,
    normalize_verified_payload,
)


class TextPipelineTest(unittest.TestCase):
    def test_prompt_examples_use_expected_schema(self):
        allowed_verified_keys = {
            "evidence",
            "score",
            "status",
            "removed_nonverbatim",
            "removed_irrelevant",
        }
        self.assertTrue(PREDEFINED_TEXT_EXAMPLES)
        for example in PREDEFINED_TEXT_EXAMPLES:
            for payload in example.verified_output.values():
                self.assertTrue(set(payload).issubset(allowed_verified_keys))

        for payload in VERIFIED_OUTPUT_SCHEMA.values():
            self.assertTrue(set(payload).issubset(allowed_verified_keys))

    def test_verified_payload_and_rows_ignore_extra_metadata(self):
        review = "The arches are elegant."
        verified = normalize_verified_payload(
            review,
            {
                "aesthetical": {
                    "evidence": ["The arches are elegant"],
                    "score": 9,
                    "status": "valid",
                    "legacy_metadata": "This extra field should not be retained.",
                }
            },
        )

        self.assertNotIn("legacy_metadata", verified["aesthetical"])
        result = TextAnalysisResult(
            review=review,
            preliminary={key: {"rationale": "", "evidence": []} for key in DIMENSION_KEYS},
            verified=verified,
            token_usage={},
            processing_time_seconds=0,
        )
        row = result.to_row()
        self.assertFalse(any(key.endswith("_metadata") for key in row))

    def test_predefined_examples_seed_rolling_memory(self):
        pipeline = TextCulturalValuePipeline(llm_client=object())

        self.assertEqual(
            [example.label for example in pipeline.memory.examples()],
            [example.label for example in PREDEFINED_TEXT_EXAMPLES],
        )

    def test_prompt_has_only_rolling_contextual_memory_section(self):
        prompt = build_preliminary_extraction_prompt(
            "A short review.",
            list(PREDEFINED_TEXT_EXAMPLES),
        )

        self.assertNotIn("Three predefined examples", prompt)
        self.assertIn("Rolling contextual memory:", prompt)
        self.assertIn(str(PREDEFINED_TEXT_EXAMPLES[0].input_payload), prompt)


if __name__ == "__main__":
    unittest.main()
