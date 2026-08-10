"""Wireframe creation benchmark task for frontend/UX design capability."""
import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import validate_sections

# Canonical set of header texts that count as a real screen for the
# "Multiple screens present" criterion.  A header matches if, after
# stripping markdown decoration (bold / italics / leading numbering /
# trailing parentheticals like "(placeholder)"), its normalized text is
# in this set exactly.
#
# Keep in sync with the screen list in the Wireframes challenge prompt.
_PRD_SCREEN_HEADERS = frozenset({
    # Single-word screen names
    "dashboard", "focus", "calendar", "planning", "settings",
    "schedule", "timer", "music", "session", "planner",
    "home", "onboarding", "profile", "analytics", "notifications",
    # Canonical two-word / suffix variants the prompt suggests
    "dashboard tab", "dashboard screen", "dashboard view",
    "focus session", "focus screen", "focus mode", "focus view",
    "focus timer",
    "calendar screen", "calendar view", "calendar integration",
    "planning screen", "planning view", "ai planning", "ai planner",
    "settings screen", "settings page", "settings view",
    "schedule screen", "schedule view",
    "timer screen", "timer view",
    "music screen", "music player", "music view",
    "session screen", "session view",
    "planner screen", "planner view",
})


def _normalize_header(text: str) -> str:
    """Strip markdown decoration from a header for screen-name matching.

    Handles (in order):
    - Leading numbering: ``1.``, ``1)``, ``1``, ``(a)``, ``-``, ``*``.
    - Emoji-keycap numbering: ``1️⃣`` / ``2️⃣`` / ... where the digit is
      followed by VS16 + COMBINING ENCLOSING KEYCAP (no `.` / `)` / space).
      ``\\d+\\W*?(?=\\s)`` keeps non-greedy with a ``\\s`` lookahead so
      we never eat into the next word.
    - Trailing parenthetical: ``(placeholder)``, ``(optional)``, ``(Tomorrow's Schedule)``.
    - Leading labels: ``Screen:``, ``Wireframe:``, ``Page:``.
    """
    s = text.strip().strip("_*`")
    # Leading numbering — covers 1. / 1) / 1 + emoji / 1 / (a) / - / *
    s = re.sub(r'^\s*(?:\d+\W*?(?=\s)|\(\w+\)|[-*])\s*', '', s)
    # Trailing parenthetical like "(placeholder)", "(optional)"
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s)
    # Leading "Screen:" / "Wireframe:" / "Page:" labels.
    s = re.sub(r'^(?:screen|wireframe|page)\s*[:\-]\s*', '', s, flags=re.IGNORECASE)
    # Collapse internal punctuation to single spaces, lowercase
    s = re.sub(r'[\s\u2013\u2014\-_/]+', ' ', s).strip().lower()
    return s


def _looks_like_screen_header(header_text: str) -> bool:
    """True iff the markdown header text is unambiguously a screen name."""
    return _normalize_header(header_text) in _PRD_SCREEN_HEADERS


class WireframesPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "wireframes"

    @property
    def version(self):
        return "0.4.1"

    @property
    def name(self):
        return "Wireframes"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a UX designer. Based on the following Product Requirements Document (PRD), "
            "create a set of low-fidelity wireframes for the mobile application.\n\n"
            "PRD — 'FlowState' Mobile App:\n"
            "FlowState is a cross-platform productivity app that combines time-blocking, "
            "focus music, and AI-powered daily planning. It integrates with Google Calendar "
            "and Microsoft Outlook, suggests optimal work blocks based on historical focus "
            "patterns, and generates a daily focus playlist that adapts to the user's energy level.\n\n"
            "Key features:\n"
            "1. Dashboard showing today's schedule, focus blocks, and energy level.\n"
            "2. Focus session screen with timer, current task, and focus music controls.\n"
            "3. Calendar integration screen to import and manage calendar events.\n"
            "4. AI planning screen to generate tomorrow's optimal schedule.\n"
            "5. Settings screen for notifications, music preferences, and calendar accounts.\n\n"
            "Produce a set of wireframes that includes:\n"
            "- At least 4 distinct screens.\n"
            "- A clear screen name and purpose for each screen.\n"
            "- A text-based wireframe representation using ASCII art, box-drawing characters, "
            "  or a structured component list with positions (e.g., top, middle, bottom).\n"
            "- Key UI components on each screen (buttons, lists, cards, navigation, etc.).\n"
            "- Navigation flows between screens (e.g., arrows, labels, or explicit transitions).\n"
            "- Annotations or notes explaining important interactions or design decisions.\n\n"
            "Use clear headings and organize by screen. Do not write prose paragraphs without "
            "structural wireframe content."
        )

    def get_temperature(self, global_config):
        if "wireframes_temperature" in global_config:
            return global_config["wireframes_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        if not t or not t.strip():
            return EvaluationResult(0.0, [])

        rubric = Rubric(self.max_score)
        rubric.record_validation(validate_sections(t, [
            "Dashboard", "Focus", "Calendar", "Planning", "Settings",
        ], min_chars=10))

        # Multiple screens present (0-3)
        # Two signals are combined: explicit "Screen:" labels AND
        # well-formed markdown headers whose text *is* a PRD-known screen
        # name (or a close canonical variant like "Focus Session" /
        # "Dashboard Tab").  We deliberately score only exact-match headers
        # so that prose-style headings like "## Pre-Dashboard Loading State"
        # or "## Focus Group Notes" do NOT over-count as screens.  An
        # earlier substring-match heuristic miscounted those and gave
        # filler wireframes 3/3 here.
        # `[^a-zA-Z]{0,8}?` lets us tolerate up to eight non-letter chars
        # (digits / emojis / `,` / `:` etc.) between the markdown `#` and
        # the keyword. This handles emoji-keycap headers like
        # `## 1️⃣ Screen: Dashboard` without breaking the line anchor.
        explicit = re.findall(
            r'(?:^|\n)\s*(?:#{1,3}\s+[^a-zA-Z]{0,8}?)?(?:screen|wireframe|page)\s*[:\-]?\s*\w+',
            t,
            re.IGNORECASE,
        )
        section_headers = re.findall(r'(?:^|\n)\s*#{1,4}\s+([^\n]+)', t)
        named_in_headers = sum(
            1 for h in section_headers if _looks_like_screen_header(h.lower())
        )
        screen_count = max(len(explicit), named_in_headers)
        # A1: defensive default — if neither detector finds any screens the
        # if/elif chain below would skip every branch and `earned` would be
        # unbound when we hand it to `rubric.add_criterion`. Captured in
        # the 2026-07-26 run as
        # `UnboundLocalError: cannot access local variable 'earned'`.
        earned = 0.0
        if screen_count >= 4:
            earned = 3.0
        elif screen_count >= 3:
            earned = 2.0
        elif screen_count >= 2:
            earned = 1.5
        elif screen_count >= 1:
            earned = 0.5
        rubric.add_criterion("Multiple screens present", 3.0, earned)

        # Screen names and purposes (0-3)
        earned = 0.0
        if re.search(r'(?:screen|page)\s*(?:name|title)|purpose', t, re.IGNORECASE):
            earned += 1.0
        screen_names = ["dashboard", "focus", "calendar", "planning", "settings", "schedule", "timer", "music"]
        matched_names = sum(1 for name in screen_names if re.search(rf'\b{name}\b', t, re.IGNORECASE))
        if matched_names >= 4:
            earned += 2.0
        elif matched_names >= 2:
            earned += 1.0
        rubric.add_criterion("Screen names and purposes", 3.0, earned)

        # Visual/structural wireframe representation (0-4)
        earned = 0.0
        if re.search(r'[┌┐└┘├┤┬┴│─┌]', t):
            earned += 2.0
        elif re.search(r'```|(\[.*\].*\n.*){2,}', t):
            earned += 1.5
        if re.search(r'\b(top|bottom|left|right|center|header|footer|sidebar)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(button|card|list|nav|menu|tab|modal|input|icon)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Visual/structural wireframe", 4.0, earned)

        # Key UI components (0-4)
        earned = 0.0
        components = ["button", "card", "list", "nav", "menu", "tab", "modal", "input", "icon", "timer", "slider", "toggle"]
        matched_components = sum(1 for comp in components if re.search(rf'\b{comp}\b', t, re.IGNORECASE))
        if matched_components >= 6:
            earned = 4.0
        elif matched_components >= 4:
            earned = 3.0
        elif matched_components >= 2:
            earned = 1.5
        elif matched_components >= 1:
            earned = 0.5
        rubric.add_criterion("Key UI components", 4.0, earned)

        rubric.eval_regex(
            "Navigation flows",
            3.0,
            t,
            [
                (r'\bnavigat|flow|transition|arrow|\->|→|=>|\bto\s+(?:the\s+)?(?:dashboard|focus|calendar|planning|settings)\b', 1.5),
                (r'\bfrom\s+\w+\s+to\s+\w+|tapping\s+.*?(?:opens?|navigates?)|clicking\s+.*?(?:opens?|navigates?)', 1.5),
            ],
        )

        rubric.eval_regex(
            "Annotations and interaction notes",
            2.0,
            t,
            [
                (r'annotation|note:|interaction|behavior|on tap|on click|when user|feedback', 1.0),
                (r'\*\*Note\*\*|\*Note\*|_Note_|\(Note|\[Note\]', 1.0),
            ],
        )

        # Coverage of PRD features (0-1)
        earned = 0.0
        prd_features = ["focus", "calendar", "music", "schedule", "timer", "AI", "planning", "settings"]
        matched_features = sum(1 for feat in prd_features if re.search(rf'\b{feat}\b', t, re.IGNORECASE))
        if matched_features >= 5:
            earned = 1.0
        elif matched_features >= 3:
            earned = 0.5
        rubric.add_criterion("Coverage of PRD features", 1.0, earned)

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
