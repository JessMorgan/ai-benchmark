"""Product Requirements Document (PRD) creation benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric


class PRDCreationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "prd-creation"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "PRD Creation"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a senior product manager. Create a comprehensive Product Requirements Document (PRD) "
            "for the following product idea.\n\n"
            "Product Idea: 'FlowState' — a cross-platform productivity application that combines "
            "time-blocking, focus music, and AI-powered daily planning. It integrates with "
            "Google Calendar and Microsoft Outlook, suggests optimal work blocks based on "
            "historical focus patterns, and generates a daily focus playlist that adapts to "
            "the user's energy level.\n\n"
            "Your PRD must be structured and professional. Include the following sections:\n"
            "1. Executive Summary — one-paragraph overview of the product.\n"
            "2. Problem Statement — the pain points this product solves.\n"
            "3. Goals & Objectives — at least 3 specific, measurable goals.\n"
            "4. Target Users & Personas — describe 2 distinct user personas.\n"
            "5. User Stories — at least 3 user stories in 'As a [persona], I want [goal], so that [benefit]' format.\n"
            "6. Functional Requirements — at least 5 distinct features or capabilities.\n"
            "7. Non-Functional Requirements — cover performance, security, reliability, and scalability.\n"
            "8. Success Metrics / KPIs — at least 3 quantitative metrics.\n"
            "9. Competitive Analysis — compare against at least 2 existing competitors.\n"
            "10. Timeline / Milestones — high-level phases or release milestones.\n"
            "11. Open Questions / Risks — list unresolved questions and major risks.\n\n"
            "Use clear headings and be specific. The PRD should be detailed enough for a "
            "development team to begin scoping work."
        )

    def get_temperature(self, global_config):
        if "prd_creation_temperature" in global_config:
            return global_config["prd_creation_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        if not t or not t.strip():
            return 0.0, []

        rubric = Rubric(self.max_score)

        rubric.eval_regex(
            "Executive Summary",
            2.0,
            t,
            [
                (r'executive summary|overview|product overview', 1.0),
                (r'flowstate|productivity|focus|time.block|calendar|playlist|ai', 1.0),
            ],
        )

        rubric.eval_regex(
            "Problem Statement",
            2.0,
            t,
            [
                (r'problem statement|pain point|challenge|issue', 1.0),
                (r'focus|distraction|scheduling|planning|productivity|overwhelm', 1.0),
            ],
        )

        rubric.eval_regex(
            "Goals & Objectives",
            2.0,
            t,
            [
                (r'goals?\s*(and|&)?\s*objectives?|objectives?', 1.0),
                (r'\d+%|\d+\s*(min|hours?|days?|weeks?|months?)\b|increase|reduce|improve', 1.0),
            ],
        )

        # Target Users & Personas needs custom counting logic
        earned = 0.0
        if re.search(r'target users?|personas?|user personas?', t, re.IGNORECASE):
            earned += 1.0
        if len(re.findall(r'\b(?:persona|user)\s*[:\-]?\s*\w+', t, re.IGNORECASE)) >= 2:
            earned += 1.0
        rubric.add_criterion("Target Users & Personas", 2.0, earned)

        # User Stories needs custom counting logic
        stories = re.findall(r'as a\s+\w+.*?i want\s+.*?so that\s+.*', t, re.IGNORECASE | re.DOTALL)
        earned = 0.0
        if stories:
            earned += 1.0
        if len(stories) >= 3:
            earned += 1.0
        rubric.add_criterion("User Stories", 2.0, earned)

        # Functional Requirements needs custom counting logic
        earned = 0.0
        if re.search(r'functional requirements?', t, re.IGNORECASE):
            earned += 1.0
        req_matches = len(re.findall(r'(?:FR[-\s]?\d+|\b(?:must|should|shall)\s+\w+|\-\s+\w+.*?:)', t, re.IGNORECASE))
        if req_matches >= 5:
            earned += 2.0
        elif req_matches >= 3:
            earned += 1.0
        rubric.add_criterion("Functional Requirements", 3.0, earned)

        # Non-Functional Requirements needs custom counting logic
        earned = 0.0
        if re.search(r'non.functional requirements?|NFR', t, re.IGNORECASE):
            earned += 0.5
        nfr_topics = ["performance", "security", "reliability", "scalability", "availability", "usability"]
        nfr_hits = sum(1 for topic in nfr_topics if re.search(rf'\b{topic}\b', t, re.IGNORECASE))
        earned += min(nfr_hits, 4) * 0.375
        rubric.add_criterion("Non-Functional Requirements", 2.0, earned)

        # Success Metrics / KPIs needs custom counting logic
        earned = 0.0
        if re.search(r'success metrics?|KPIs?|key performance indicators?', t, re.IGNORECASE):
            earned += 0.5
        metric_matches = len(re.findall(r'(?:^|\n)\s*(?:\d+\.\s+|\-\s+|\*\s+).*(?:\d+%|\d+\s*(?:users?|customers?|hours?|minutes?)|retention|conversion|churn|engagement|satisfaction)', t, re.IGNORECASE | re.MULTILINE))
        earned += min(metric_matches, 3) * 0.5
        rubric.add_criterion("Success Metrics / KPIs", 2.0, earned)

        # Competitive Analysis needs custom counting logic
        earned = 0.0
        if re.search(r'competitive analysis|competitors?|comparison|vs\.?', t, re.IGNORECASE):
            earned += 0.5
        competitor_names = re.findall(r'\b(todoist|notion|calendar|outlook|google calendar|trello|asana|clockify|forest|rescue time|focusmate)\b', t, re.IGNORECASE)
        earned += min(len(competitor_names), 2) * 0.5
        if re.search(r'(?:lacks|strength|weakness|advantage|differentiator|comparison)', t, re.IGNORECASE):
            earned += 0.5
        rubric.add_criterion("Competitive Analysis", 2.0, earned)

        rubric.eval_regex(
            "Timeline / Milestones",
            2.0,
            t,
            [
                (r'timeline|milestones?|roadmap|phases?|release plan', 1.0),
                (r'\b(Q\d|week\s*\d+|month\s*\d+|phase\s*\d+|MVP|beta|launch)\b', 1.0),
            ],
        )

        # Open Questions / Risks needs custom logic
        earned = 0.0
        if re.search(r'open questions?|risks?|assumptions?|dependencies?', t, re.IGNORECASE):
            earned += 0.5
        if re.search(r'\?', t):
            earned += 0.25
        if re.search(r'\b(risk|mitigation|concern|dependency|assumption)\b', t, re.IGNORECASE):
            earned += 0.25
        rubric.add_criterion("Open Questions / Risks", 1.0, earned)

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text)[0]
