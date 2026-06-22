import unittest

from heritage_perception_analysis.llm_client import normalize_base_url


class LLMClientTest(unittest.TestCase):
    def test_normalize_base_url_removes_chat_completions_suffix(self):
        self.assertEqual(
            normalize_base_url("https://open.bigmodel.cn/api/paas/v4/chat/completions"),
            "https://open.bigmodel.cn/api/paas/v4",
        )

    def test_normalize_base_url_keeps_api_root(self):
        self.assertEqual(
            normalize_base_url("https://open.bigmodel.cn/api/paas/v4/"),
            "https://open.bigmodel.cn/api/paas/v4",
        )

    def test_normalize_base_url_allows_default_openai_endpoint(self):
        self.assertIsNone(normalize_base_url(None))


if __name__ == "__main__":
    unittest.main()
