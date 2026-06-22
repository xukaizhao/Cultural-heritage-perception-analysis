import unittest

from heritage_perception_analysis.memory import RollingContextMemory


class RollingContextMemoryTest(unittest.TestCase):
    def test_keeps_most_recent_examples(self):
        memory = RollingContextMemory(max_examples=3)
        for index in range(5):
            memory.add(f"input-{index}", {"score": index})

        examples = memory.examples()
        self.assertEqual([example.input_payload for example in examples], ["input-2", "input-3", "input-4"])
        self.assertEqual(len(memory), 3)


if __name__ == "__main__":
    unittest.main()
