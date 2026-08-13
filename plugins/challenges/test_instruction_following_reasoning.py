"""Tests for the instruction-following and reasoning challenge plugins."""

from plugins.challenges.instruction_following import InstructionFollowingPlugin
from plugins.challenges.reasoning import ReasoningPlugin


class TestInstructionFollowingPlugin:
    def setup_method(self):
        self.plugin = InstructionFollowingPlugin()

    def test_metadata_prompt_and_temperature(self):
        assert self.plugin.version == "0.1.0"
        assert "T-09" in self.plugin.get_prompt()
        assert self.plugin.get_temperature({"instruction_following_temperature": 0.2}) == 0.2
        assert self.plugin.get_temperature({}) is None

    def test_empty_response_scores_zero(self):
        assert self.plugin.score("") == 0.0

    def test_exact_response_scores_full(self):
        response = """ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-09 | CUSTOMER RAVI | TOTAL 65.00
[SUMMARY] count=4; total=404.90; top_order=T-02"""
        assert self.plugin.score(response) == self.plugin.max_score

    def test_wrong_tie_break_and_filtered_record_are_penalized(self):
        response = """ORDER T-05 | CUSTOMER NOOR | TOTAL 120.00
ORDER T-02 | CUSTOMER JULES | TOTAL 120.00
ORDER T-08 | CUSTOMER ZARA | TOTAL 99.90
ORDER T-06 | CUSTOMER ASHA | TOTAL 50.00
[SUMMARY] count=4; total=404.90; top_order=T-05"""
        score = self.plugin.score(response)
        assert score > 0.0
        assert score < self.plugin.max_score


class TestReasoningPlugin:
    def setup_method(self):
        self.plugin = ReasoningPlugin()

    def test_metadata_prompt_and_temperature(self):
        assert self.plugin.version == "0.1.0"
        assert "Profile happened before Auth" in self.plugin.get_prompt()
        assert self.plugin.get_temperature({"reasoning_temperature": 0.1}) == 0.1
        assert self.plugin.get_temperature({}) is None

    def test_empty_response_scores_zero(self):
        assert self.plugin.score("") == 0.0

    def test_complete_deduction_scores_full(self):
        response = """
1. Auth is immediately before Search, Profile is before Auth, Upload is after Search, and Billing is after Upload but before Notifications.
2. Therefore Profile is at 09:00, Auth is at 09:15, Search is at 09:30, Upload is at 09:45, Billing is at 10:00, and Notifications is at 10:15.
3. Ben owned Search, Eli owned Upload, and Ana owned Notifications at 10:15.
4. Auth is P1, Notifications is P2, and Upload has higher priority than Search, which has higher priority than Billing.
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P4
TIME: 09:30
"""
        assert self.plugin.score(response) == self.plugin.max_score

    def test_final_answer_without_deductions_is_incomplete(self):
        response = """
FAILED_SERVICE: Search
OWNER: Ben
PRIORITY: P4
TIME: 09:30
"""
        score = self.plugin.score(response)
        assert score > 0.0
        assert score < self.plugin.max_score
