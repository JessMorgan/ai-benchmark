"""Software architecture document creation benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


class SoftwareArchitecturePlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "software-architecture"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Software Architecture"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are a senior backend/coding architect. Based on the following Product Requirements Document (PRD), "
            "produce a comprehensive software architecture document.\n\n"
            "PRD — 'FlowState' Productivity Platform:\n"
            "FlowState is a cross-platform productivity application that combines time-blocking, "
            "focus music, and AI-powered daily planning. It integrates with Google Calendar and Microsoft Outlook, "
            "suggests optimal work blocks based on historical focus patterns, and generates a daily focus playlist "
            "that adapts to the user's energy level.\n\n"
            "Key requirements:\n"
            "- Support web, iOS, and Android clients.\n"
            "- Real-time sync of focus sessions and calendar events across devices.\n"
            "- AI service that analyzes historical focus data and generates schedules.\n"
            "- Music service integration (Spotify, Apple Music).\n"
            "- OAuth2-based authentication with Google and Microsoft.\n"
            "- Push notifications for focus blocks and calendar events.\n"
            "- Analytics dashboard for productivity trends.\n"
            "- Must scale to 1 million daily active users.\n\n"
            "Your architecture document must include:\n"
            "1. Executive Summary — brief overview of the architecture approach.\n"
            "2. Requirements Summary — functional and non-functional requirements derived from the PRD.\n"
            "3. Architecture Style — e.g., microservices, modular monolith, event-driven, serverless.\n"
            "4. Component Diagram / Description — major components and their responsibilities.\n"
            "5. Data Model — key entities, relationships, and storage choices.\n"
            "6. API Design — key endpoints or GraphQL schema for core operations.\n"
            "7. Technology Stack — languages, frameworks, databases, message queues, etc.\n"
            "8. Deployment Architecture — cloud provider, containers, CI/CD, observability.\n"
            "9. Security Considerations — authentication, authorization, data protection.\n"
            "10. Scalability & Performance — caching, load balancing, database scaling.\n"
            "11. Trade-offs & Decisions — rationale for major architectural choices.\n\n"
            "Use clear headings and be specific. The document should be detailed enough for a "
            "development team to begin technical design and implementation."
        )

    def get_temperature(self, global_config):
        if "software_architecture_temperature" in global_config:
            return global_config["software_architecture_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        if not t or not t.strip():
            return 0.0, []

        rubric = []
        s = 0.0

        # 1. Executive Summary / Overview (0-1)
        earned = 0.0
        if re.search(r'executive summary|overview|architecture overview', t, re.IGNORECASE):
            earned = 1.0
        s += earned
        rubric.append({"name": "Executive Summary", "max": 1.0, "earned": earned, "missed": round(1.0 - earned, 1)})

        # 2. Requirements Summary (0-2)
        earned = 0.0
        if re.search(r'requirements summary|functional requirements|non.functional requirements', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:sync|real.time|oauth|push notification|analytics|scale|1 million|calendar|music|AI)\b', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Requirements Summary", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 3. Architecture Style (0-2)
        earned = 0.0
        styles = ["microservices", "modular monolith", "event-driven", "serverless", "layered", "hexagonal", "SOA"]
        if any(re.search(rf'\b{re.escape(style)}\b', t, re.IGNORECASE) for style in styles):
            earned += 1.0
        if re.search(r'architecture style|architectural pattern|chosen approach|rationale', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Architecture Style", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 4. Component Diagram / Description (0-3)
        earned = 0.0
        if re.search(r'component|service|module|subsystem', t, re.IGNORECASE):
            earned += 1.0
        # Look for diagram markers or structured component lists
        if re.search(r'```|┌|├──|(\[.*\].*\n.*){2,}', t):
            earned += 1.0
        # Look for specific component names
        components = ["auth", "calendar", "music", "AI", "analytics", "notification", "api", "gateway", "database"]
        matched = sum(1 for comp in components if re.search(rf'\b{comp}\b', t, re.IGNORECASE))
        if matched >= 4:
            earned += 1.0
        earned = round(min(earned, 3.0), 1)
        s += earned
        rubric.append({"name": "Component Description", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 5. Data Model (0-3)
        earned = 0.0
        if re.search(r'data model|database schema|entity|ERD|entity.relationship', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(user|session|focus|calendar|event|playlist|schedule)\b', t, re.IGNORECASE):
            earned += 0.5
        # Look for relationship indicators
        if re.search(r'relationship|foreign key|one.to.many|many.to.many|primary key|index', t, re.IGNORECASE):
            earned += 1.0
        # Look for table/schema descriptions
        if re.search(r'\b(table|schema|collection)\s*[:\-]?\s*\w+', t, re.IGNORECASE):
            earned += 0.5
        earned = round(min(earned, 3.0), 1)
        s += earned
        rubric.append({"name": "Data Model", "max": 3.0, "earned": earned, "missed": round(3.0 - earned, 1)})

        # 6. API Design (0-2)
        earned = 0.0
        if re.search(r'API design|API endpoints|REST|GraphQL|gRPC|endpoint', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'GET|POST|PUT|DELETE|PATCH|/api/v1|mutation|query', t):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "API Design", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 7. Technology Stack (0-2)
        earned = 0.0
        if re.search(r'technology stack|tech stack|stack|languages|frameworks', t, re.IGNORECASE):
            earned += 1.0
        # Look for specific technologies
        techs = ["python", "node", "go", "rust", "java", "react", "flutter", "swift", "kotlin",
                 "postgres", "mongodb", "redis", "kafka", "rabbitmq", "docker", "kubernetes",
                 "aws", "gcp", "azure", "terraform", "nginx"]
        matched = sum(1 for tech in techs if re.search(rf'\b{tech}\b', t, re.IGNORECASE))
        if matched >= 3:
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Technology Stack", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 8. Deployment Architecture (0-2)
        earned = 0.0
        if re.search(r'deployment|CI/CD|continuous integration|continuous delivery|pipeline', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(docker|kubernetes|k8s|ecs|eks|gke|aks|lambda|container)\b', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Deployment Architecture", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 9. Security Considerations (0-2)
        earned = 0.0
        if re.search(r'security|authentication|authorization|OAuth|JWT|encryption', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(OAuth2?|JWT|TLS|SSL|HTTPS|mTLS|RBAC|CORS|XSS|CSRF|SQL injection)\b', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Security Considerations", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 10. Scalability & Performance (0-2)
        earned = 0.0
        if re.search(r'scalability|performance|caching|load balancing|database scaling', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(cache|redis|CDN|load balancer|sharding|replica|horizontal|vertical|rate.limit)\b', t, re.IGNORECASE):
            earned += 1.0
        earned = round(min(earned, 2.0), 1)
        s += earned
        rubric.append({"name": "Scalability & Performance", "max": 2.0, "earned": earned, "missed": round(2.0 - earned, 1)})

        # 11. Trade-offs & Decisions (0-1)
        earned = 0.0
        if re.search(r'trade.off|tradeoffs|decisions|rationale|pros and cons|considerations', t, re.IGNORECASE):
            earned = 1.0
        s += earned
        rubric.append({"name": "Trade-offs & Decisions", "max": 1.0, "earned": earned, "missed": round(1.0 - earned, 1)})

        return round(min(s, self.max_score), 1), rubric

    def score(self, response_text):
        return self.evaluate(response_text)[0]
