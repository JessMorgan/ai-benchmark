"""Tests for the PRD creation challenge plugin."""
import unittest

from plugins import discover_plugins


class TestPRDCreationScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "prd-creation")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_partial_response_scores(self):
        text = (
            "## Executive Summary\n\n"
            "FlowState is a productivity app.\n\n"
            "## Problem Statement\n\n"
            "People struggle to focus.\n\n"
            "## Goals & Objectives\n\n"
            "Increase focus time by 20%.\n\n"
            "## Target Users & Personas\n\n"
            "Persona 1: Alice, a developer.\n"
            "Persona 2: Bob, a designer.\n\n"
            "## User Stories\n\n"
            "As a developer, I want to block focus time, so that I can ship code.\n"
            "As a designer, I want focus music, so that I can stay in flow.\n"
            "As a manager, I want calendar integration, so that meetings don't interrupt deep work.\n\n"
            "## Functional Requirements\n\n"
            "FR-1: Calendar integration.\n"
            "FR-2: Focus music.\n"
            "FR-3: AI planning.\n"
            "FR-4: Time blocking.\n"
            "FR-5: Daily playlist.\n\n"
            "## Non-Functional Requirements\n\n"
            "Performance: < 2s load.\n"
            "Security: OAuth2.\n\n"
            "## Success Metrics\n\n"
            "Daily active users, focus time.\n\n"
            "## Competitive Analysis\n\n"
            "Todoist and Notion are competitors.\n\n"
            "## Timeline\n\n"
            "Q1: MVP, Q2: Beta, Q3: Launch.\n\n"
            "## Open Questions\n\n"
            "Which music provider to integrate?"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_full_response_scores_high(self):
        text = (
            "## Executive Summary\n\n"
            "FlowState is a cross-platform productivity application that combines time-blocking, "
            "focus music, and AI-powered daily planning.\n\n"
            "## Problem Statement\n\n"
            "Knowledge workers face constant distractions and lack a unified tool for scheduling, "
            "focus, and adaptive music.\n\n"
            "## Goals & Objectives\n\n"
            "1. Increase average daily deep-work time by 25% within 3 months.\n"
            "2. Achieve 100,000 monthly active users within the first year.\n"
            "3. Maintain a 4.5-star app store rating.\n\n"
            "## Target Users & Personas\n\n"
            "Persona 1: Alice, a software engineer who needs uninterrupted focus blocks.\n"
            "Persona 2: Bob, a product manager juggling meetings and deep work.\n\n"
            "## User Stories\n\n"
            "As a software engineer, I want AI-suggested focus blocks, so that I can protect deep-work time.\n"
            "As a product manager, I want calendar integration, so that focus time is respected.\n"
            "As a remote worker, I want adaptive focus music, so that I can maintain energy.\n\n"
            "## Functional Requirements\n\n"
            "FR-1: Integrate with Google Calendar and Microsoft Outlook.\n"
            "FR-2: Generate AI-powered daily schedules based on historical focus patterns.\n"
            "FR-3: Provide adaptive focus music playlists.\n"
            "FR-4: Send smart notifications before focus blocks.\n"
            "FR-5: Track focus sessions and productivity analytics.\n\n"
            "## Non-Functional Requirements\n\n"
            "Performance: App cold start under 2 seconds.\n"
            "Security: OAuth2 and encrypted tokens.\n"
            "Reliability: 99.9% uptime.\n"
            "Scalability: Support 1M concurrent users.\n\n"
            "## Success Metrics / KPIs\n\n"
            "1. Daily active users (DAU).\n"
            "2. Average focus session length.\n"
            "3. User retention at 30 days.\n\n"
            "## Competitive Analysis\n\n"
            "Todoist offers task management but lacks focus music.\n"
            "Notion offers flexibility but lacks adaptive scheduling.\n\n"
            "## Timeline / Milestones\n\n"
            "Q1: MVP with calendar integration.\n"
            "Q2: Beta with AI planning.\n"
            "Q3: Public launch with focus music.\n\n"
            "## Open Questions / Risks\n\n"
            "Risk: Music licensing. Open question: Which calendar provider to prioritize?"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)

    def test_quantified_kpis_with_days_and_comma_counts_earn_full_credit(self):
        text = (
            "## Success Metrics / KPIs\n\n"
            "1. User retention at 30 days.\n"
            "2. Grow weekly active users to 2,500 users.\n"
            "3. Cut onboarding time to 5 minutes.\n"
        )
        result = self.plugin.evaluate(text)
        kpi = next(item for item in result.rubric if item["name"] == "Success Metrics / KPIs")
        self.assertEqual(kpi["earned"], 2.0)

    def test_numbered_user_stories_earn_story_credit(self):
        text = (
            "## User Stories\n\n"
            "1. As a developer, I want to block focus time, so that I can ship code.\n"
            "2. As a designer, I want focus music, so that I can stay in flow.\n"
            "3. As a manager, I want calendar integration, so that meetings don't interrupt deep work.\n"
        )
        result = self.plugin.evaluate(text)
        stories = next(item for item in result.rubric if item["name"] == "User Stories")
        self.assertEqual(stories["earned"], 2.0)
        two_stories = text.replace("3. As a manager, I want calendar integration, so that meetings don't interrupt deep work.\n", "")
        result = self.plugin.evaluate(two_stories)
        stories = next(item for item in result.rubric if item["name"] == "User Stories")
        self.assertEqual(stories["earned"], 1.0)

    def test_prose_personas_earn_persona_credit(self):
        text = (
            "## Target Users & Personas\n\n"
            "The primary persona, Alice, is a software engineer who needs uninterrupted focus blocks.\n"
            "The secondary persona, Bob, is a product manager juggling meetings and deep work.\n"
        )
        result = self.plugin.evaluate(text)
        personas = next(item for item in result.rubric if item["name"] == "Target Users & Personas")
        self.assertEqual(personas["earned"], 2.0)

    def test_phase_timeline_earn_milestone_credit(self):
        text = (
            "## Timeline / Milestones\n\n"
            "Phase 1: MVP with calendar integration.\n"
            "Phase 2: Beta with AI planning.\n"
            "Phase 3: Public launch with focus music.\n"
        )
        result = self.plugin.evaluate(text)
        timeline = next(item for item in result.rubric if item["name"] == "Timeline / Milestones")
        self.assertEqual(timeline["earned"], 2.0)

    def test_repo_fixture_kpi_days_unit_earns_partial_credit(self):
        # The repo's own strong-PRD fixture listed "User retention at 30 days."
        # as its only quantified KPI and scored 0/2 because the KPI quantifier
        # omitted the days/weeks/months units the Goals criterion accepts.
        text = (
            "## Success Metrics / KPIs\n\n"
            "1. Daily active users (DAU).\n"
            "2. Average focus session length.\n"
            "3. User retention at 30 days.\n"
        )
        result = self.plugin.evaluate(text)
        kpi = next(item for item in result.rubric if item["name"] == "Success Metrics / KPIs")
        self.assertEqual(kpi["earned"], 1.0)


if __name__ == "__main__":
    unittest.main()
