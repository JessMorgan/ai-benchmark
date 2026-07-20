"""Tests for the wireframes challenge plugin."""
import unittest

from plugins import discover_plugins


class TestWireframesScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "wireframes")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_whitespace_response_scores_zero(self):
        self.assertEqual(self.plugin.score("   \n\t  "), 0.0)

    def test_get_temperature_from_config(self):
        self.assertEqual(self.plugin.get_temperature({"wireframes_temperature": 0.5}), 0.5)

    def test_get_temperature_default(self):
        self.assertIsNone(self.plugin.get_temperature({}))

    def test_single_screen_header(self):
        text = "Screen 1: Dashboard\n\n```\n[Header] Dashboard\n```"
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_two_screen_headers(self):
        text = (
            "Screen 1: Dashboard\n\n"
            "Screen 2: Focus\n\n"
            "```\n[Header] Dashboard\n[Button] Start\n```"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_partial_response_scores(self):
        text = (
            "## Screen 1: Dashboard\n\n"
            "Purpose: Show today's schedule.\n\n"
            "```\n"
            "[Header] FlowState\n"
            "[Card] Today's focus blocks\n"
            "[List] Upcoming events\n"
            "[Button] Start focus\n"
            "```\n\n"
            "## Screen 2: Focus Session\n\n"
            "Purpose: Run a focus timer.\n\n"
            "```\n"
            "[Header] Focus\n"
            "[Timer] 25:00\n"
            "[Button] Pause\n"
            "[Slider] Music volume\n"
            "```\n\n"
            "## Screen 3: Calendar\n\n"
            "Purpose: Manage calendar events.\n\n"
            "```\n"
            "[Header] Calendar\n"
            "[List] Events\n"
            "[Button] Sync\n"
            "```\n\n"
            "## Screen 4: Settings\n\n"
            "Purpose: Configure app.\n\n"
            "```\n"
            "[Header] Settings\n"
            "[Toggle] Notifications\n"
            "[Button] Connect Calendar\n"
            "```\n\n"
            "Navigation: Dashboard -> Focus, Dashboard -> Calendar, Dashboard -> Settings.\n\n"
            "Note: Tapping Start focus navigates to Focus Session."
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_full_response_scores_high(self):
        text = (
            "## Wireframe 1: Dashboard\n\n"
            "**Purpose:** Provide an overview of today's schedule, focus blocks, and energy level.\n\n"
            "```\n"
            "┌─────────────────────────┐\n"
            "│  FlowState              │\n"
            "│  Good morning, Alex     │\n"
            "├─────────────────────────┤\n"
            "│  [Card] Focus Score: 85 │\n"
            "│  [Chart] Energy level   │\n"
            "├─────────────────────────┤\n"
            "│  [List] Today's blocks  │\n"
            "│  - Deep work 09:00-11:00│\n"
            "│  - Meeting 14:00-15:00  │\n"
            "├─────────────────────────┤\n"
            "│  [Button] Start Focus   │\n"
            "│  [Nav] Home | Focus | Cal│\n"
            "└─────────────────────────┘\n"
            "```\n\n"
            "**Annotations:** Tapping 'Start Focus' navigates to the Focus Session screen.\n\n"
            "## Wireframe 2: Focus Session\n\n"
            "**Purpose:** Run a focused work session with music and timer.\n\n"
            "```\n"
            "┌─────────────────────────┐\n"
            "│  Focus Session          │\n"
            "│  [Timer] 25:00          │\n"
            "│  Current task: Write PRD │\n"
            "├─────────────────────────┤\n"
            "│  [Button] Pause         │\n"
            "│  [Slider] Music volume  │\n"
            "│  [Card] Now playing...  │\n"
            "├─────────────────────────┤\n"
            "│  [Nav] Home | Focus | Cal│\n"
            "└─────────────────────────┘\n"
            "```\n\n"
            "**Annotations:** Swipe down to return to Dashboard.\n\n"
            "## Wireframe 3: Calendar Integration\n\n"
            "**Purpose:** Import and manage calendar events.\n\n"
            "```\n"
            "┌─────────────────────────┐\n"
            "│  Calendar               │\n"
            "│  [Toggle] Google Cal    │\n"
            "│  [Toggle] Outlook       │\n"
            "├─────────────────────────┤\n"
            "│  [List] Imported events │\n"
            "│  - Standup 09:00        │\n"
            "│  - Review 15:00         │\n"
            "├─────────────────────────┤\n"
            "│  [Button] Sync Now      │\n"
            "│  [Nav] Home | Focus | Cal│\n"
            "└─────────────────────────┘\n"
            "```\n\n"
            "**Annotations:** Tapping 'Sync Now' refreshes events from connected calendars.\n\n"
            "## Wireframe 4: AI Planning\n\n"
            "**Purpose:** Generate tomorrow's optimal schedule.\n\n"
            "```\n"
            "┌─────────────────────────┐\n"
            "│  AI Planning            │\n"
            "│  [Card] Energy forecast │\n"
            "├─────────────────────────┤\n"
            "│  [List] Suggested blocks│\n"
            "│  - Deep work 09:00-11:00│\n"
            "│  - Break 11:00-11:30    │\n"
            "├─────────────────────────┤\n"
            "│  [Button] Apply Schedule│\n"
            "│  [Nav] Home | Focus | Cal│\n"
            "└─────────────────────────┘\n"
            "```\n\n"
            "**Annotations:** Tapping 'Apply Schedule' writes events to the calendar.\n\n"
            "## Wireframe 5: Settings\n\n"
            "**Purpose:** Configure notifications, music, and accounts.\n\n"
            "```\n"
            "┌─────────────────────────┐\n"
            "│  Settings               │\n"
            "│  [Toggle] Notifications │\n"
            "│  [Slider] Music volume  │\n"
            "│  [List] Music sources   │\n"
            "├─────────────────────────┤\n"
            "│  [Button] Connect Cal   │\n"
            "│  [Button] Log Out       │\n"
            "│  [Nav] Home | Focus | Cal│\n"
            "└─────────────────────────┘\n"
            "```\n\n"
            "**Annotations:** Tapping 'Connect Calendar' opens the Calendar Integration screen.\n\n"
            "## Navigation Flow\n\n"
            "Dashboard → Focus Session (tap Start Focus)\n"
            "Dashboard → Calendar Integration (tap Calendar nav)\n"
            "Dashboard → AI Planning (tap Plan nav)\n"
            "Dashboard → Settings (tap Settings nav)\n"
            "Settings → Calendar Integration (tap Connect Calendar)\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 10.0)


if __name__ == "__main__":
    unittest.main()
