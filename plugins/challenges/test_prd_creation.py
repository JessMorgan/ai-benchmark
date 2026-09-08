"""Tests for the PRD creation challenge plugin."""
import unittest

from plugins import discover_plugins
from plugins.challenges import prd_creation


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

    def test_generic_competitor_names_earn_competitive_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Pomofocus is the closest competitor but lacks adaptive scheduling.\n"
            "Rize.io's strength is automatic time tracking, which we do not offer.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 2.0)

    def test_case_variant_competitor_names_dedupe(self):
        generic = (
            "## Competitive Analysis\n\n"
            "POMOFOCUS and Pomofocus both lack adaptive scheduling.\n"
        )
        whitelisted = (
            "## Competitive Analysis\n\n"
            "Todoist and TODOIST both lack focus music.\n"
        )
        for text in (generic, whitelisted):
            result = self.plugin.evaluate(text)
            competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
            self.assertEqual(competitive["earned"], 0.5)

    def test_duplicate_stories_do_not_inflate_count(self):
        case_variant = (
            "## User Stories\n\n"
            "1. As a developer, I want to block focus time, so that I can ship code.\n"
            "2. As a designer, I want focus music, so that I can stay in flow.\n"
            "3. As a DEVELOPER, I want to block focus time, so that I can ship code.\n"
        )
        exact_repeat = (
            "## User Stories\n\n"
            "As a developer, I want to block focus time, so that I can ship code.\n"
            "As a developer, I want to block focus time, so that I can ship code.\n"
            "As a developer, I want to block focus time, so that I can ship code.\n"
        )
        for text in (case_variant, exact_repeat):
            result = self.plugin.evaluate(text)
            stories = next(item for item in result.rubric if item["name"] == "User Stories")
            self.assertEqual(stories["earned"], 1.0)

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


class TestCompetitorNamesMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "prd-creation")

    def test_article_prefixed_names_are_not_swallowed(self):
        text = ("The Pomofocus app lacks adaptive scheduling. "
                "The Rize.io app tracks time automatically.")
        self.assertEqual(prd_creation._competitor_names(text), {"pomofocus", "rize.io"})

    def test_pronoun_only_prose_names_nothing(self):
        text = "It lacks adaptive scheduling. They offer focus music."
        self.assertEqual(prd_creation._competitor_names(text), set())

    def test_schedule_prose_names_nothing(self):
        text = "We plan to launch in Q3. Phase 2 adds adaptive scheduling."
        self.assertEqual(prd_creation._competitor_names(text), set())

    def test_adjacent_names_count_separately(self):
        text = "Pomofocus Rize.io both lack adaptive scheduling."
        self.assertEqual(prd_creation._competitor_names(text), {"pomofocus", "rize.io"})

    def test_whitelist_names_still_match(self):
        text = ("Todoist offers task management but lacks focus music. "
                "Notion offers flexibility but lacks adaptive scheduling.")
        self.assertEqual(prd_creation._competitor_names(text), {"todoist", "notion"})

    def test_generic_names_with_possessive_suffix(self):
        text = ("Pomofocus is the closest competitor but lacks adaptive scheduling. "
                "Rize.io's strength is automatic time tracking.")
        self.assertEqual(prd_creation._competitor_names(text), {"pomofocus", "rize.io"})

    def test_case_variants_dedupe(self):
        text = "POMOFOCUS and Pomofocus both lack adaptive scheduling."
        self.assertEqual(prd_creation._competitor_names(text), {"pomofocus"})

    def test_whitelist_multi_word_name_not_double_counted(self):
        text = "Rescue Time lacks adaptive scheduling."
        self.assertEqual(prd_creation._competitor_names(text), {"rescue time"})

    def test_bare_common_words_do_not_become_names(self):
        self.assertEqual(prd_creation._competitor_names("Time is key. Phase matters."), set())

    def test_article_prefixed_competitors_earn_full_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "The Pomofocus app lacks adaptive scheduling.\n"
            "The Rize.io app tracks time automatically.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 2.0)

    def test_pronoun_only_competitors_earn_no_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "It lacks adaptive scheduling.\n"
            "They offer focus music.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 0.0)

    def test_adjacent_competitor_names_earn_full_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Pomofocus Rize.io both lack adaptive scheduling; their strength is focus music.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 2.0)

    def test_discourse_connectives_do_not_become_names(self):
        text = ("However, our approach is different. "
                "First, we plan to launch in Q3.")
        self.assertEqual(prd_creation._competitor_names(text), set())

    def test_comparative_connectives_do_not_become_names(self):
        text = ("Compared to Todoist, our approach is different. "
                "Unlike Todoist, it lacks focus music.")
        self.assertEqual(prd_creation._competitor_names(text), {"todoist"})

    def test_subheading_and_label_words_do_not_become_names(self):
        text = "### Key Competitors\n### Overview\nTop competitor: Todoist. It lacks focus music."
        self.assertEqual(prd_creation._competitor_names(text), {"todoist"})

    def test_generic_names_require_comparative_context(self):
        text = "Pomofocus is solid. Rize.io is solid too."
        self.assertEqual(prd_creation._competitor_names(text), set())

    def test_discourse_connective_prose_earns_no_competitive_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "However, our approach is different.\n"
            "First, we plan to launch in Q3.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 0.0)

    def test_single_competitor_after_connective_earns_partial_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Compared to Todoist, our approach is different.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 0.5)

    def test_comparative_object_after_comma_counts(self):
        text = ("Compared to Pomofocus, our app adds adaptive scheduling. "
                "Unlike Rize.io, it also offers focus music. "
                "Versus Todoist, we add adaptive scheduling.")
        self.assertEqual(prd_creation._competitor_names(text), {"pomofocus", "rize.io", "todoist"})

    def test_comparative_object_patterns_count(self):
        self.assertEqual(
            prd_creation._competitor_names("Versus Pomofocus, our app is different."),
            {"pomofocus"},
        )
        self.assertEqual(
            prd_creation._competitor_names("Its main competitor, Rize.io, lacks focus music."),
            {"rize.io"},
        )

    def test_comma_object_comparatives_earn_full_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Compared to Pomofocus, our app adds adaptive scheduling.\n"
            "Unlike Rize.io, it also offers focus music.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 2.0)

    def test_unlike_single_competitor_earns_partial_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Unlike Rize.io, our app is better at adaptive scheduling.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 0.5)

    def test_corporate_suffixes_do_not_become_names(self):
        text = "Todoist, Inc. lacks focus music."
        self.assertEqual(prd_creation._competitor_names(text), {"todoist"})

    def test_comma_enumeration_without_marker_earns_partial_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Pomofocus, Rize.io, and Todoist all lack adaptive scheduling.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 0.5)

    def test_better_than_comparative_earns_full_credit(self):
        text = (
            "## Competitive Analysis\n\n"
            "Pomofocus is better than Rize.io for time tracking.\n"
        )
        result = self.plugin.evaluate(text)
        competitive = next(item for item in result.rubric if item["name"] == "Competitive Analysis")
        self.assertEqual(competitive["earned"], 2.0)


if __name__ == "__main__":
    unittest.main()
