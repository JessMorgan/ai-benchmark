"""Coverage tests for the newer challenge evaluators."""
import unittest

from plugins.challenges.debug_traversal import DebugTraversalPlugin
from plugins.challenges.error_recovery import ErrorRecoveryPlugin
from plugins.challenges.multi_turn_conversation import MultiTurnConversationPlugin


class TestDebugTraversalPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = DebugTraversalPlugin()

    def test_metadata_prompt_and_temperature(self):
        self.assertIn("BUG REPORT", self.plugin.get_prompt())
        self.assertEqual(self.plugin.get_temperature({"debug_traversal_temperature": 0.4}), 0.4)
        self.assertIsNone(self.plugin.get_temperature({}))

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_complete_root_cause_analysis_scores_high(self):
        response = """
        ## Root Cause
        Walk through the execution step by step, line by line, following the code path and
        control flow. The bug is an incorrect condition involving the user_id key: a null or
        empty value is an edge case, while a string or integer ID is hashable.
        ## Analysis
        This traces the logic and explains the error, issue, problem, and defect.
        ## Fix
        Here is the corrected solution and patch/change:
        ```python
        def find_duplicate_users(entries):
            return [user_id for user_id in entries]
        ```
        ## Test
        ```python
        def test_duplicates():
            assert find_duplicate_users([]) == []
        ```
        Use pytest and assert the result.
        ## Side Effects
        Discuss side effects, impact, consequences, and regression risk for the caller,
        calling code, upstream and downstream integration, including performance, memory,
        speed, efficiency, and overhead.
        """
        self.assertGreaterEqual(self.plugin.score(response), 90)


class TestErrorRecoveryPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = ErrorRecoveryPlugin()

    def test_metadata_prompt_and_temperature(self):
        self.assertIn("WeatherAPI", self.plugin.get_prompt())
        self.assertEqual(self.plugin.get_temperature({"error_recovery_temperature": 0.6}), 0.6)
        self.assertIsNone(self.plugin.get_temperature({}))

    def test_empty_response_scores_base_structure(self):
        # The evaluator awards one point for the structure criterion even
        # before any content-specific signals are present.
        self.assertEqual(self.plugin.score(""), 5)

    def test_complete_resilient_design_scores_high(self):
        response = '''
        ```python
        import asyncio
        import logging
        class AllProvidersFailedError(Exception):
            pass
        class WeatherClient:
            """Client with a fallback strategy."""
            async def fetch(self, city: str) -> dict:
                return {}
        async def get_weather_resilient(city: str) -> dict:
            try:
                results = await asyncio.gather(asyncio.create_task(fetch(city)))
                return results[0]
            except Exception as exc:
                logging.getLogger(__name__).error("provider failure: %s", exc)
                raise AllProvidersFailedError("all providers failed") from exc
        async def demo():
            # Demonstrate all succeed, one fail, partial success, and all fail.
            await asyncio.wait_for(get_weather_resilient("Tokyo"), timeout=2)
            print("scenario: all providers succeed; one provider fails; all providers fail")
        ```
        Use retry with exponential backoff and timeout attempts: fall through to the next
        provider, alternate and secondary fallback. Handle rate limit and error payloads,
        bad data and invalid formats with schema parsing and unexpected/mismatched data.
        Detect conflicts when providers disagree, differ, contradict, or have a discrepancy.
        Return the result and output, with type hints, docstrings, a demo, example, usage,
        and an if __name__ entry point.
        '''
        self.assertGreaterEqual(self.plugin.score(response), 90)

    def test_partial_design_exercises_false_branches(self):
        response = "async def get_weather_resilient(city: str) -> dict:\n    return {}"
        self.assertGreater(self.plugin.score(response), 0.0)
        self.assertLess(self.plugin.score(response), 100)


class TestMultiTurnConversationPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = MultiTurnConversationPlugin()

    def test_metadata_prompt_and_temperature(self):
        self.assertIn("Version 3", self.plugin.get_prompt())
        self.assertEqual(self.plugin.get_temperature({"multi_turn_conversation_temperature": 0.5}), 0.5)
        self.assertIsNone(self.plugin.get_temperature({}))

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_three_revisions_and_summary_score_high(self):
        response = """
        ## Version 1 (Initial)
        ```
        Dear team, I decline the offer but appreciate the opportunity.
        ```
        ## Version 2 (After Feedback 1)
        ```
        I enjoyed meeting the specific team and the interview left a positive impression.
        ```
        ## Version 3 (After Feedback 2)
        ```
        Warm, genuine best wishes and kind regards; I hope to follow the company's future work.
        ```
        ## Summary of Changes
        Changed and added personalization; removed formality; replaced wording, modified and
        softened the tone, tightened the close, and made it more personal. V1 changed into V2;
        V2 changed into V3. I decline the offer for this role while leaving an opportunity to
        stay in touch and preserve the connection. I sincerely wish you well.
        """
        self.assertGreaterEqual(self.plugin.score(response), 90)

    def test_missing_summary_and_blocks_is_penalized(self):
        response = "## Version 1\nA polite email to decline the offer and future opportunity."
        score = self.plugin.score(response)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 100)


if __name__ == "__main__":
    unittest.main()
