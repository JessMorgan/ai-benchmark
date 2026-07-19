"""Wireframe creation benchmark task for frontend/UX design capability."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


class WireframesPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "wireframes"

    @property
    def version(self):
        return "0.1.0"

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
            return 0.0, []

        rubric = []
        s = 0.0

        # 1. Multiple screens present (0-3)
        earned = 0.0
        screen_headers = re.findall(r'(?:^|\n)\s*(?:#{1,3}\s+)?(?:screen|wireframe|page)\s*[:\-]?\s*\w+', t, re.IGNORECASE)
        if len(screen_headers) >= 4:
            earned = 3.0
        elif len(screen_headers) >= 2:
            earned = 1.5
        elif len(screen_headers) >= 1:
            earned = 0.5
        s += earned
        rubric.append({"name": "Multiple screens present", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 2. Screen names and purposes (0-3)
        earned = 0.0
        if re.search(r'(?:screen|page)\s*(?:name|title)|purpose', t, re.IGNORECASE):
            earned += 1.0
        # Look for named screens related to the PRD
        screen_names = ["dashboard", "focus", "calendar", "planning", "settings", "schedule", "timer", "music"]
        matched_names = sum(1 for name in screen_names if re.search(rf'\b{name}\b', t, re.IGNORECASE))
        if matched_names >= 4:
            earned += 2.0
        elif matched_names >= 2:
            earned += 1.0
        earned = round(min(earned, 3.0), 1)
        s += earned
        rubric.append({"name": "Screen names and purposes", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 3. Visual/structural wireframe representation (0-4)
        earned = 0.0
        # ASCII/box drawing
        if re.search(r'[┌┐└┘├┤┬┴│─┌]', t):
            earned += 2.0
        # Markdown code blocks or structured layout
        elif re.search(r'```|(\[.*\].*\n.*){2,}', t):
            earned += 1.5
        # Component list with positions
        if re.search(r'\b(top|bottom|left|right|center|header|footer|sidebar)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(button|card|list|nav|menu|tab|modal|input|icon)\b', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 4.0), 1)
        s += earned
        rubric.append({"name": "Visual/structural wireframe", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        # 4. Key UI components (0-4)
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
        s += earned
        rubric.append({"name": "Key UI components", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        # 5. Navigation flows between screens (0-3)
        earned = 0.0
        if re.search(r'\bnavigat|flow|transition|arrow|\->|→|=>|\bto\s+(?:the\s+)?(?:dashboard|focus|calendar|planning|settings)\b', t, re.IGNORECASE):
            earned += 1.5
        if re.search(r'\bfrom\s+\w+\s+to\s+\w+|tapping\s+.*?(?:opens?|navigates?)|clicking\s+.*?(?:opens?|navigates?)', t, re.IGNORECASE):
            earned += 1.5
        earned = round(min(earned, 3.0), 1)
        s += earned
        rubric.append({"name": "Navigation flows", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 6. Annotations and interaction notes (0-2)
        earned = 0.0
        if re.search(r'annotation|note:|interaction|behavior|on tap|on click|when user|feedback', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\*\*Note\*\*|\*Note\*|_Note_|\(Note|\[Note\]', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Annotations and interaction notes", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 7. Coverage of PRD features (0-1)
        earned = 0.0
        prd_features = ["focus", "calendar", "music", "schedule", "timer", "AI", "planning", "settings"]
        matched_features = sum(1 for feat in prd_features if re.search(rf'\b{feat}\b', t, re.IGNORECASE))
        if matched_features >= 5:
            earned = 1.0
        elif matched_features >= 3:
            earned = 0.5
        s += earned
        rubric.append({"name": "Coverage of PRD features", "max": 1.0, "earned": earned, "missed": round(1.0 - earned, 1)})

        return round(min(s, self.max_score), 1), rubric

    def score(self, response_text):
        return self.evaluate(response_text)[0]
