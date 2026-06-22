import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from heritage_perception_analysis.image_pipeline import (
    ImageVisualQualityPipeline,
    attraction_name_for_image,
    discover_images,
    image_to_data_uri,
    load_default_image_memory_examples,
)
from heritage_perception_analysis.memory import MemoryExample


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
MEMORY_DIR = EXAMPLES_DIR / "image_memory"
TEST_IMAGE = EXAMPLES_DIR / "test.jpg"


class FakeVisionClient:
    def __init__(self):
        self.config = SimpleNamespace(vision_model="fake-vision-model")
        self.calls: list[list[dict[str, Any]]] = []
        self.responses = [
            ('{"visual_analysis":"first target","score":7,"reason":"first reason"}', {}),
            ('{"visual_analysis":"second target","score":6,"reason":"second reason"}', {}),
        ]

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)


class ImagePipelineTest(unittest.TestCase):
    def test_example_images_are_discovered(self):
        images = discover_images(EXAMPLES_DIR)
        names = [image.name for image in images]

        self.assertIn("test.jpg", names)
        self.assertNotIn("1.jpg", names)
        self.assertNotIn("2.jpg", names)
        self.assertNotIn("3.jpg", names)

    def test_example_image_data_uri(self):
        data_uri = image_to_data_uri(TEST_IMAGE)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(data_uri), 100)

    def test_attraction_name_for_flat_test_folder(self):
        attraction = attraction_name_for_image(EXAMPLES_DIR, TEST_IMAGE)

        self.assertEqual(attraction, "examples")

    def test_default_image_examples_seed_memory(self):
        examples = load_default_image_memory_examples()

        self.assertEqual([example.label for example in examples], ["1.jpg", "2.jpg", "3.jpg"])
        self.assertTrue(all(str(example.input_payload).startswith("data:image/jpeg;base64,") for example in examples))
        self.assertTrue(all("score" in example.verified_output for example in examples))

    def test_image_prompt_includes_contextual_memory_examples(self):
        pipeline = ImageVisualQualityPipeline(llm_client=object())
        messages = pipeline._build_messages(image_to_data_uri(TEST_IMAGE))

        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 3)
        self.assertEqual(messages[1]["content"][0]["text"], "Contextual memory example.")
        self.assertEqual(messages[-1]["content"][0]["text"], "Target image.")

    def test_successful_image_results_roll_into_contextual_memory(self):
        seed_examples = [
            MemoryExample(f"seed-image-{index}", {"score": index, "reason": f"seed {index}"})
            for index in range(1, 4)
        ]
        fake_client = FakeVisionClient()
        pipeline = ImageVisualQualityPipeline(
            llm_client=fake_client,
            seed_examples=seed_examples,
        )

        async def run_two_images():
            await pipeline.analyze_image(TEST_IMAGE, attraction="test", relative_path="first.jpg")
            await pipeline.analyze_image(TEST_IMAGE, attraction="test", relative_path="second.jpg")

        import asyncio

        asyncio.run(run_two_images())

        memory_inputs = [
            message["content"][1]["image_url"]["url"]
            for message in fake_client.calls[1]
            if message["role"] == "user" and message["content"][0]["text"] == "Contextual memory example."
        ]
        memory_outputs = [
            message["content"]
            for message in fake_client.calls[1]
            if message["role"] == "assistant"
        ]

        self.assertEqual(memory_inputs[0], "seed-image-2")
        self.assertEqual(memory_inputs[1], "seed-image-3")
        self.assertTrue(str(memory_inputs[2]).startswith("data:image/jpeg;base64,"))
        self.assertIn('"score":7', memory_outputs[2])


if __name__ == "__main__":
    unittest.main()
