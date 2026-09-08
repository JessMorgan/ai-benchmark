"""Structured Product Requirements Document challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import (
    exact_section,
    markdown_sections,
    numbered_or_bulleted_items,
)
from plugins.challenges._rubric import Rubric

# Matches an "As a ... I want ... so that ..." user story line with an
# optional bullet or list-number prefix; group(1) is the story content
# without that prefix, which is what story dedupe must compare.
_STORY_RE = re.compile(r"(?:[-*]\s*)?(?:\d+[.)]\s*)?(As an?\s+.+?\s*,?\s+I want\s+.+?\s*,?\s+so that\s+.+)", re.IGNORECASE)
_KNOWN_COMPETITORS_RE = re.compile(r"\b(?:Todoist|Notion|Trello|Asana|Forest|Rescue Time|Focusmate)\b", re.IGNORECASE)
# A single capitalized product-name token ("Pomofocus", "Rize.io"); the
# optional dotted segment keeps "Rize.io" whole while a trailing sentence
# period ("Q3.") is not captured.
_COMPETITOR_NAME_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*")
# A section carries a comparative predicate when it contains one of the
# criterion's comparative-context words (lacks/strength/weakness/different/
# advantage/comparison) or a comparative verb/adjective. Used twice: the
# generic fallback only attributes capitalized tokens to competitors inside
# such a section (so neutral prose and subheadings cannot fabricate names),
# and the full-credit comparative-context check reuses the same vocabulary
# so the two stay consistent.
_COMPARATIVE_PREDICATE_RE = re.compile(
    r"\b(?:lacks?|lacking|strength|strengths|weakness|weaknesses|"
    r"different|differentiates|differentiated|differentiating|"
    r"advantage|advantages|disadvantage|disadvantages|"
    r"comparison|comparisons|comparative|compare|compared|comparing|"
    r"versus|better|worse|faster|slower|cheaper|outperforms?|outperformed|"
    r"beats?|beaten|behind|ahead|similar|similarly)\b",
    re.IGNORECASE,
)
# Discourse connectives that open a sentence rather than name a product
# ("However,", "First,", "Compared to ..."). Consulted only for the first
# word of a sentence; most function words are already in the stoplist.
_SENTENCE_START_CONNECTIVES = frozenset({
    "however", "first", "second", "third", "fourth", "fifth", "sixth",
    "compared", "compare", "comparing", "unlike", "versus", "moreover",
    "additionally", "furthermore", "finally", "overall", "similar",
    "similarly", "note", "notes", "notably", "also", "besides", "thus",
    "hence", "therefore", "consequently", "accordingly", "meanwhile",
    "instead", "otherwise", "nonetheless", "nevertheless", "whereas",
    "generally", "typically", "basically", "essentially", "ultimately",
    "primarily", "specifically", "particularly", "currently", "recently",
})
# Splits a section body into sentences for the connective checks: a
# sentence ends at sentence punctuation followed by whitespace, or at a
# newline, so subheadings and bullet lines are their own sentences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A comma-followed name still counts when it is the object of a comparative
# construction in the same sentence: a comparative connective/preposition
# sits within the last couple of words before the token (optionally with a
# comma, as in "its competitor, Todoist,"): "Compared to X,", "Compared
# with X,", "Unlike X,", "Versus X,", "vs X,", "Against X,", "competitor,
# X,", "rival, X,", "alternative, X,". Comma-separated enumerations without
# such a per-token marker ("Pomofocus, Rize.io, and Todoist all lack X")
# keep the documented partial-credit tradeoff.
_COMPARATIVE_OBJECT_RE = re.compile(
    r"\b(?:compared\s+(?:to|with)|unlike|versus|vs\.?|against|competitors?|rivals?|alternatives?)\s*,?\s*$",
    re.IGNORECASE,
)
# Words that are not competitor product names: articles, pronouns, and
# function words, section headings, PRD vocabulary, and common capitalized
# sentence-starters / acronyms. Applied per token by the generic fallback
# so degenerate prose cannot fabricate names.
_COMPETITOR_STOPWORDS = frozenset({
    # articles / determiners / pronouns
    "a", "an", "the", "this", "that", "these", "those", "each", "every",
    "both", "all", "some", "any", "no", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "hers", "its", "our", "their",
    # conjunctions / interrogatives
    "and", "or", "but", "nor", "so", "yet", "because", "while", "when",
    "if", "then", "than", "as", "although", "though", "where", "which",
    "who", "whom", "whose", "what", "how", "why",
    # prepositions
    "in", "on", "at", "to", "for", "by", "per", "via", "with", "from",
    "of", "into", "onto", "upon", "about", "over", "under", "after",
    "before", "during", "since", "until", "against", "between", "among",
    "through", "across", "around", "near", "off", "out", "up", "down",
    "without",
    # auxiliaries / common verbs
    "be", "am", "is", "are", "was", "were", "been", "being", "have",
    "has", "had", "do", "does", "did", "doing", "will", "would", "can",
    "could", "shall", "should", "may", "might", "must", "need", "let",
    "lets", "make", "makes", "made", "get", "gets", "got", "take",
    "takes", "took", "offer", "offers", "offered", "plan", "plans",
    "planned", "planning", "launch", "launches", "launched", "add",
    "adds", "added", "track", "tracks", "tracked", "provide", "provides",
    "lacks", "lacking", "lack", "use", "uses", "used", "build", "builds",
    "include", "includes", "included", "believe", "believes", "think",
    "thinks", "know", "knows", "want", "wants",
    # section headings / PRD vocabulary
    "competitive", "analysis", "competitor", "competitors", "flowstate",
    "executive", "summary", "problem", "statement", "goal", "goals",
    "objective", "objectives", "target", "targets", "persona", "personas",
    "story", "stories", "requirement", "requirements", "success", "metric",
    "metrics", "question", "questions", "risk", "risks", "mitigation",
    "assumption", "dependency", "roadmap", "roadmaps", "milestone",
    "milestones", "timeline", "phase", "phases", "beta", "product",
    "products", "market", "markets", "user", "users", "team", "teams",
    "demo", "demos", "feature", "features", "strength", "strengths",
    "weakness", "weaknesses", "different", "differentiator",
    "differentiators", "advantage", "advantages", "comparison",
    "comparisons", "leader", "leaders",
    # product-domain common words
    "time", "tracking", "tracker", "focus", "focused", "focusing",
    "music", "scheduling", "schedule", "scheduled", "adaptive",
    "integration", "integrations", "notification", "notifications",
    "analytics", "performance", "security", "reliability", "scalability",
    # acronyms / quarters / abbreviations
    "ai", "mvp", "prd", "kpi", "kpis", "dau", "q1", "q2", "q3", "q4",
    "vs", "e.g", "i.e",
    # corporate suffixes (never product names)
    "inc", "llc", "corp", "ltd", "co",
    # comparative vocabulary (predicate words, never product names)
    "better", "best", "worse", "worst", "faster", "fastest", "slower",
    "slowest", "cheaper", "cheapest", "crowded", "differentiates",
    "differentiated", "differentiating", "differentiation", "competition",
    "compete", "competes", "competed", "competing", "outperform",
    "outperforms", "outperformed", "outperforming", "approach",
    "approaches", "approached", "approaching",
    # subheadings / label words
    "key", "top", "bottom", "overview", "summaries",
    "conclusion", "conclusions", "introduction", "intro", "landscape",
    "segment", "segments", "niche", "player", "players", "option",
    "options", "alternative", "alternatives", "rival", "rivals",
    "marketplace", "tool", "tools", "app", "apps", "service",
    "services", "platform", "platforms", "solution", "solutions",
    "company", "companies", "vendor", "vendors", "brand", "brands",
    "startup", "startups", "industry", "industries", "trend", "trends",
    "growth", "adoption", "usage", "customer", "customers",
    "audience", "audiences",
})


def _competitor_names(text: str) -> set[str]:
    """Collect distinct competitor names, casefolded, from a section body.

    The known-competitor whitelist is one source of names (it keeps
    multi-word names such as "Rescue Time" intact). The generic fallback
    then counts capitalized product-name tokens (e.g. "Pomofocus",
    "Rize.io") so real competitors outside the whitelist earn credit —
    but only when the section carries a comparative predicate, so neutral
    prose and subheadings cannot fabricate names. Within such a section,
    discourse connectives are dropped per token: the first word of a
    sentence that is a connective ("However,", "First,", "Compared to
    ..."), and a token immediately followed by a comma — unless the token
    is the object of a comparative construction in the same sentence
    ("Compared to X,", "Unlike X,", "Versus X,", "its competitor, X,"),
    which still counts; comma-separated enumerations without such a
    per-token marker keep the documented partial-credit tradeoff.
    Stopword tokens are dropped per token — never the whole run — so "The
    Pomofocus" still yields "pomofocus", and adjacent names ("Pomofocus
    Rize.io") count as two distinct competitors. Tokens that are part of a
    whitelist match are skipped so "Rescue Time" is not also counted as
    "rescue". Names are casefolded so "Pomofocus", "pomofocus", and
    "POMOFOCUS" dedupe.
    """
    names = {name.casefold() for name in _KNOWN_COMPETITORS_RE.findall(text)}
    if _COMPARATIVE_PREDICATE_RE.search(text):
        generic_text = _KNOWN_COMPETITORS_RE.sub(lambda m: " " * len(m.group(0)), text)
        for sentence in _SENTENCE_SPLIT_RE.split(generic_text):
            first_word = re.search(r"[A-Za-z][A-Za-z0-9.]*", sentence)
            for match in _COMPETITOR_NAME_TOKEN_RE.finditer(sentence):
                token = match.group(0)
                if len(token) < 2:
                    continue
                token_cf = token.casefold()
                if token_cf in _COMPETITOR_STOPWORDS:
                    continue
                if (
                    re.match(r"\s*,", sentence[match.end():])
                    and not _COMPARATIVE_OBJECT_RE.search(sentence[: match.start()])
                ):
                    continue
                if (
                    first_word is not None
                    and match.start() == first_word.start()
                    and token_cf in _SENTENCE_START_CONNECTIVES
                ):
                    continue
                names.add(token_cf)
    return names


class PRDCreationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "prd-creation"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "PRD Creation"

    @property
    def max_score(self) -> int:
        return int(22.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Create a detailed FlowState PRD with exactly these sections: Executive Summary, "
            "Problem Statement, Goals & Objectives, Target Users & Personas, User Stories, "
            "Functional Requirements, Non-Functional Requirements, Success Metrics / KPIs, "
            "Competitive Analysis, Timeline / Milestones, Open Questions / Risks. Include at "
            "least 3 measurable goals, 2 distinct personas, 3 lines in `As a ... I want ... so "
            "that ...` format, 5 distinct functional requirements, performance/security/"
            "reliability/scalability NFRs, 3 quantitative KPIs with targets, 2 distinct competitors, "
            "milestones, and risks/questions. Use section-local content."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("prd_creation_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) >= 11,
            "evidence": [{"kind": "section", "heading": section.heading, "chars": len(section.body)} for section in sections],
            "errors": [],
        })())
        def body(names: list[str]) -> str:
            section = exact_section(text, names[0], names[1:])
            return section.body if section else ""
        executive = body(["Executive Summary"])
        problem = body(["Problem Statement"])
        goals = body(["Goals & Objectives", "Goals", "Objectives"])
        personas = body(["Target Users & Personas", "Target Users", "Personas"])
        stories = body(["User Stories"])
        functional = body(["Functional Requirements"])
        nfr = body(["Non-Functional Requirements", "NFR"])
        metrics = body(["Success Metrics / KPIs", "Success Metrics", "KPIs", "Key Performance Indicators"])
        competitors = body(["Competitive Analysis", "Competitors"])
        timeline = body(["Timeline / Milestones", "Timeline", "Milestones", "Roadmap"])
        risks = body(["Open Questions / Risks", "Open Questions", "Risks"])
        rubric.add_criterion("Executive Summary", 2.0, 2.0 if len(executive) >= 80 and re.search(r"flowstate|productivity|focus", executive, re.IGNORECASE) else 0.0)
        rubric.add_criterion("Problem Statement", 2.0, 2.0 if len(problem) >= 60 and re.search(r"problem|pain|distraction|scheduling|focus", problem, re.IGNORECASE) else 0.0)
        goal_items = numbered_or_bulleted_items(goals)
        measurable = [item for item in goal_items if re.search(r"\d+\s*%|\d+\s*(?:users?|days?|weeks?|months?|seconds?)|increase|reduce|improve", item, re.IGNORECASE)]
        rubric.add_criterion("Goals & Objectives", 2.0, 2.0 if len(measurable) >= 3 else (1.0 if len(measurable) >= 1 else 0.0))
        persona_items = numbered_or_bulleted_items(personas)
        prose_persona_names = {
            match.group(1).lower()
            for match in re.finditer(r"\bpersona[s]?\b[^.\n]*?\b([A-Z][a-z]+)\b", personas)
        }
        persona_ok = (
            len(persona_items) >= 2
            and len({item.split(":", 1)[0].strip().lower() for item in persona_items}) >= 2
        ) or len(prose_persona_names) >= 2
        rubric.add_criterion("Target Users & Personas", 2.0, 2.0 if persona_ok else 0.0)
        story_lines = []
        for line in stories.splitlines():
            match = _STORY_RE.match(line)
            if match:
                story_lines.append(match.group(1).strip())
        distinct_stories = {line.lower() for line in story_lines}
        rubric.add_criterion("User Stories", 2.0, 2.0 if len(distinct_stories) >= 3 else (1.0 if story_lines else 0.0))
        req_items = numbered_or_bulleted_items(functional)
        req_items += re.findall(r"(?im)^\s*FR[- ]?\d+\s*:\s*(.+)$", functional)
        rubric.add_criterion("Functional Requirements", 3.0, 3.0 if len({item.lower() for item in req_items}) >= 5 else (1.5 if len(req_items) >= 3 else 0.0))
        nfr_hits = sum(bool(re.search(rf"\b{topic}\b", nfr, re.IGNORECASE)) for topic in ("performance", "security", "reliability", "scalability"))
        rubric.add_criterion("Non-Functional Requirements", 2.0, 2.0 if nfr_hits == 4 else nfr_hits / 2.0)
        metric_items = numbered_or_bulleted_items(metrics)
        quantified = [item for item in metric_items if re.search(r"\d+(?:,\d{3})*\s*%|\d+(?:,\d{3})*\s*(?:users?|minutes?|seconds?|hours?|days?|weeks?|months?)", item, re.IGNORECASE)]
        rubric.add_criterion("Success Metrics / KPIs", 2.0, 2.0 if len(quantified) >= 3 else (1.0 if quantified else 0.0))
        names = _competitor_names(competitors)
        # Full credit reuses the predicate gate's vocabulary so a section
        # that earns 2 generic names via "better/faster/..." also passes
        # the comparative-context check (gate and credit stay consistent).
        rubric.add_criterion("Competitive Analysis", 2.0, 2.0 if len(names) >= 2 and _COMPARATIVE_PREDICATE_RE.search(competitors) else float(min(len(names), 2) / 2.0))
        timeline_items = numbered_or_bulleted_items(timeline)
        timeline_phases = re.findall(r"\bphase\s*\d+", timeline, re.IGNORECASE)
        milestone_ok = re.search(r"MVP|beta|launch|Q[1-4]|phase", timeline, re.IGNORECASE)
        rubric.add_criterion("Timeline / Milestones", 2.0, 2.0 if milestone_ok and (len(timeline_items) >= 2 or len(timeline_phases) >= 2) else 0.0)
        rubric.add_criterion("Open Questions / Risks", 1.0, 1.0 if re.search(r"\?|risk|mitigation|assumption|dependency", risks, re.IGNORECASE) else 0.0)
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
