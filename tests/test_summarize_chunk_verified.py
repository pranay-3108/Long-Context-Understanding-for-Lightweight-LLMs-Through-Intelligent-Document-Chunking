"""
Mocked verification of the Sprint 2 retry/parsing logic.
No ollama.chat calls -- _call_model is fully mocked in every test.

Fixture files are written under tests/_fixtures_verified_chunks (inside the
repo, not the OS temp dir) specifically to avoid Windows sandbox permission
issues on paths like C:\\Users\\...\\AppData\\Local\\Temp.
"""
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from models.granite import summarize_chunk_verified as scv

FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures_verified_chunks"

CHUNK_TEXT = "The model achieved 3.57% error using 152 layers on ImageNet."

GOOD_RESPONSE = (
    "SUMMARY:\n"
    "The model achieved 3.57% error using 152 layers on ImageNet.\n\n"
    "COVERAGE:\n"
    "Problem: Present\n"
    "Method: Present\n"
    "Evidence: Present\n"
    "Numbers: Present\n"
    "Conclusion: Present\n"
)

BAD_RESPONSE = (
    "SUMMARY:\n"
    "The model performed well on the dataset.\n\n"
    "COVERAGE:\n"
    "Problem: Present\n"
    "Method: Present\n"
    "Evidence: Missing\n"
    "Numbers: Missing\n"
    "Conclusion: Present\n"
)

MALFORMED_RESPONSE = "Sure, here's a summary: the model got 3.57% error."  # no SUMMARY:/COVERAGE: at all


class TestSummarizeChunkVerified(unittest.TestCase):

    def setUp(self):
        if FIXTURE_DIR.exists():
            shutil.rmtree(FIXTURE_DIR)
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        (FIXTURE_DIR / "testpaper_chunk_1.txt").write_text(CHUNK_TEXT, encoding="utf-8")

    def tearDown(self):
        if FIXTURE_DIR.exists():
            shutil.rmtree(FIXTURE_DIR)

    def test_good_summary_needs_no_retry(self):
        with patch.object(scv, "_call_model", return_value=GOOD_RESPONSE) as mock_call:
            files, total_time, log = scv.summarize_chunk_verified("testpaper", chunk_dir=FIXTURE_DIR)

        self.assertEqual(mock_call.call_count, 1, "a clean summary must not trigger a retry")
        self.assertEqual(len(files), 1)
        self.assertEqual(log[-1]["retries_used"], 0)
        self.assertEqual(log[-1]["flag"], "OK")

        verification_file = FIXTURE_DIR / "testpaper__adaptive_verified__verification_1.json"
        self.assertTrue(verification_file.exists())
        data = json.loads(verification_file.read_text(encoding="utf-8"))
        self.assertEqual(data["ollama_calls_used"], 1)

    def test_bad_summary_triggers_exactly_one_retry_then_succeeds(self):
        with patch.object(
            scv, "_call_model", side_effect=[BAD_RESPONSE, GOOD_RESPONSE]
        ) as mock_call:
            files, total_time, log = scv.summarize_chunk_verified("testpaper", chunk_dir=FIXTURE_DIR)

        self.assertEqual(mock_call.call_count, 2, "must retry exactly once, not zero, not more")
        self.assertEqual(log[-1]["retries_used"], 1)
        self.assertEqual(log[-1]["flag"], "OK")

        verification_file = FIXTURE_DIR / "testpaper__adaptive_verified__verification_1.json"
        data = json.loads(verification_file.read_text(encoding="utf-8"))
        self.assertEqual(data["retries_used"], 1)
        self.assertEqual(data["ollama_calls_used"], 2)

    def test_still_bad_after_retry_does_not_retry_again(self):
        with patch.object(
            scv, "_call_model", side_effect=[BAD_RESPONSE, BAD_RESPONSE]
        ) as mock_call:
            files, total_time, log = scv.summarize_chunk_verified("testpaper", chunk_dir=FIXTURE_DIR)

        self.assertEqual(mock_call.call_count, 2, "must NEVER exceed 1 retry, even if still bad")
        self.assertEqual(log[-1]["retries_used"], 1)
        self.assertIn(log[-1]["flag"], ("FACTS_DROPPED", "FAILED_AFTER_RETRY"))

    def test_malformed_response_does_not_crash_the_run(self):
        """
        Regression test for the bug found on 2026-08-13: a model response
        missing the SUMMARY:/COVERAGE: markers entirely used to raise an
        uncaught ValueError and crash the whole paper's run. It must now
        be treated like any other bad response: one bounded retry, then
        move on.
        """
        with patch.object(
            scv, "_call_model", side_effect=[MALFORMED_RESPONSE, MALFORMED_RESPONSE]
        ) as mock_call:
            files, total_time, log = scv.summarize_chunk_verified("testpaper", chunk_dir=FIXTURE_DIR)

        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(log[-1]["retries_used"], 1)
        # Must still produce output files even in the worst case -- no crash, no silent drop.
        self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
