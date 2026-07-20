"""Software architecture document creation benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric


class SoftwareArchitecturePlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "software-architecture"

    @property
    def version(self):
        return "0.2.0"

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

        rubric = Rubric(self.max_score)

        # Architecture & Patterns (3.0)
        earned = 0.0
        if re.search(r'\b(?:microservices|event.driven|serverless|modular monolith|hexagonal|layered|SOA)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:api gateway|backend.for.frontend|bff|service mesh)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'```(?:mermaid|plantuml)|graph\s+(?:TD|LR)|sequenceDiagram|classDiagram|flowchart', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Architecture & Patterns", 3.0, earned)

        # Data Modeling & Strategy (3.0)
        earned = 0.0
        if re.search(r'\b(?:relational|sql|postgres|mysql|sqlite)\b', t, re.IGNORECASE):
            earned += 0.5
        if re.search(r'\b(?:time.series|nosql|document db|document.database|columnar|graph db|wide.column|cassandra|mongodb|dynamodb)\b', t, re.IGNORECASE):
            earned += 0.5
        if re.search(r'\b(?:sharding|read replica|partitioning|cluster|database scaling|federation)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:cache eviction|ttl|time.to.live|write.through|read.aside|cache invalidation|cache.aside)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Data Modeling & Strategy", 3.0, earned)

        # Real-Time Sync & Communication (3.0)
        earned = 0.0
        if re.search(r'\b(?:websocket|wss|server.sent events|sse|grpc|mqtt|long.polling)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:crdt|conflict.free|conflict resolution|operational transformation|event sourcing|offline.first|vector clock)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:kafka|rabbitmq|event bus|pub.sub|kinesis|sqs|sns|nats|pulsar)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Real-Time Sync & Communication", 3.0, earned)

        # Scalability & Capacity Planning (3.0)
        earned = 0.0
        if re.search(r'\b(?:requests per second|rps|qps|capacity planning|back.of.the.envelope|throughput estimate|bandwidth|1[\s,]*000[\s,]*000|million)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:auto.scaling|kubernetes|k8s|horizontal pod|hpa|vertical scaling)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:cdn|content delivery network|edge computing|multi.region|cloudfront|cloudflare)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Scalability & Capacity Planning", 3.0, earned)

        # Resiliency & Failure Modes (3.0)
        earned = 0.0
        if re.search(r'\b(?:circuit breaker|bulkhead pattern|bulkheads|failover|fault isolation)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:dead.letter|dlq|poison pill|message retry|retry queue)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:exponential backoff|jitter|chaos engineering|fallback|graceful degradation|retry mechanism)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Resiliency & Failure Modes", 3.0, earned)

        # Security & Protections (3.0)
        earned = 0.0
        if re.search(r'\b(?:refresh token|token rotation|oidc|openid connect|mfa|multi.factor)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:rate limit|throttling|waf|ddos protection|api gateway security)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:gdpr|ccpa|pii|encryption at rest|encryption in transit|kms|secrets manager)\b', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Security & Protections", 3.0, earned)

        # Observability & SLOs (2.0)
        earned = 0.0
        if re.search(r'\b(?:opentelemetry|distributed tracing|prometheus|datadog|grafana|elk|metrics and logs|logging|monitoring)\b', t, re.IGNORECASE):
            earned += 1.0
        if re.search(r'\b(?:slo|sli|sla|service level objective|error budget|99\.[0-9]%)', t, re.IGNORECASE):
            earned += 1.0
        rubric.add_criterion("Observability & SLOs", 2.0, earned)

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text)[0]
