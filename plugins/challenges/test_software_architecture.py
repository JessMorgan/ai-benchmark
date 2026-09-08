"""Tests for the software architecture challenge plugin."""
import unittest

from plugins import discover_plugins

# Prompt-conformant correct response: every section the rubric scores, with
# real content, a markdown-table endpoint list, a workload estimate, genuine
# failure handling, and a component diagram with arrows pointing at named
# components.
CORRECT_RESPONSE = """## Executive Summary

FlowState will be built as an event-driven microservices platform. An API Gateway
routes traffic to bounded contexts (Auth, Calendar, Music, AI Planner, Analytics,
Notifications). Real-time sync uses WebSockets; calendar conflicts are resolved
with CRDTs.

## Requirements Summary

Support web, iOS, and Android clients, calendar OAuth, AI planning, focus music,
push notifications, analytics, and 1M DAU with sub-2s latency.

## Architecture Style

Event-driven microservices with independently scalable bounded contexts behind an
API gateway. Each service owns its data and communicates over an event bus.

## Component Diagram / Description

```mermaid
graph TD
  A[Client Apps] --> B[API Gateway]
  B --> C[Auth Service]
  B --> D[Calendar Service]
  B --> E[Music Service]
  B --> F[AI Planner Service]
  B --> G[Analytics Service]
  B --> H[Notification Service]
```

Each service exposes a narrow REST API and publishes domain events to Kafka.

## Data Model

Relational data (users, sessions, schedules) lives in PostgreSQL with read replicas.
Time-series focus metrics and event streams go to a columnar store. Sharding by
user_id partitions data, and Redis provides read-aside caching with TTL.

## API Design

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/v1/sessions | List focus sessions |
| POST | /api/v1/sessions | Start a focus session |
| GET | /api/v1/calendar/events | List calendar events |
| POST | /api/v1/plan | Generate AI schedule |
| GET | /api/v1/analytics | Productivity trends |

## Technology Stack

Python, FastAPI, Celery, PostgreSQL, Redis, ClickHouse, Kafka, Docker, Kubernetes.

## Deployment Architecture

AWS multi-region Kubernetes deployment with a CDN in front of static assets and
managed Kafka workers for async processing.

## Real-Time Sync & Communication

WebSockets push live focus sessions to clients. Calendar updates use CRDTs for
offline-first conflict resolution with last-write-wins fallback. Kafka acts as the
event bus for async processing across services.

## Resiliency & Failure Modes

Circuit breakers protect downstream calendar and music APIs. Failed Kafka messages
go to a dead-letter queue. Exponential backoff with jitter retries transient errors,
and services fail over across regions on outage.

## Security Considerations

OAuth2/OIDC with refresh-token rotation, rate limiting and a WAF at the API gateway,
encryption at rest for PII, and GDPR-compliant data retention.

## Scalability & Performance

At 1M DAU we estimate ~2,300 concurrent users and ~1,150 RPS at peak. Kubernetes
HPA scales pods horizontally. A CDN (CloudFront) serves static assets from edge
locations and read replicas absorb analytics read load.

## Trade-offs & Decisions

Microservices add operational complexity but allow independent scaling and failure
isolation. PostgreSQL chosen over NoSQL for strong consistency of scheduling data.

## Observability & SLOs

OpenTelemetry tracing, Prometheus metrics, and Grafana dashboards. SLO: 99.9%
availability with an error budget, p99 latency under 200 ms.
"""

# Keyword-stuffing / degenerate response: every heading is present, but each
# section is a dump of rubric keywords. It uses table endpoints, only
# "multi-region" in the resiliency section (no real failure handling), bare
# arrows in the architecture section, and no workload estimate.
STUFFED_RESPONSE = """## Executive Summary

FlowState architecture microservices event-driven serverless api gateway component.

## Requirements Summary

web iOS Android OAuth realtime sync AI planning music notifications analytics 1M DAU.

## Architecture Style

microservices modular monolith event-driven serverless api gateway service component ->

## Component Diagram / Description

service component -> -> -> responsibilities of each service component.

## Data Model

postgres sql relational nosql document columnar mongodb dynamodb time-series entity user
session schedule shard replica partition cache redis ttl.

## API Design

| GET | /api/v1/sessions | x |
| POST | /api/v1/sessions | x |
| GET | /api/v1/calendar/events | x |

## Technology Stack

Python FastAPI PostgreSQL Redis Kafka Docker Kubernetes.

## Deployment Architecture

Docker containers on AWS multi-region kubernetes cdn edge load balancer.

## Real-Time Sync & Communication

websocket sse grpc polling sync crdt conflict offline eventual last-write kafka queue
broker pub-sub event bus.

## Resiliency & Failure Modes

multi-region deployment for availability.

## Security Considerations

oauth oidc mfa refresh token authorization rate limit waf ddos throttl pii gdpr
encryption kms secret.

## Scalability & Performance

1M DAU million users. cdn edge multi-region load balancer.

## Trade-offs & Decisions

trade-offs of microservices versus modular monolith.

## Observability & SLOs

opentelemetry tracing prometheus metrics logging monitoring slo sli sla error budget
99.9% availability.
"""


class TestSoftwareArchitectureScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "software-architecture")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

    def test_prompt_includes_scored_headings(self):
        # The rubric awards points for Real-Time Sync, Resiliency, and
        # Observability sections, so the prompt must request them; otherwise a
        # prompt-conformant response is penalized for omitting scored sections.
        prompt = self.plugin.get_prompt()
        self.assertIn("Real-Time Sync", prompt)
        self.assertIn("Resiliency", prompt)
        self.assertIn("Observability", prompt)

    def test_prompt_conformant_correct_beats_stuffed(self):
        correct = self.plugin.score(CORRECT_RESPONSE)
        stuffed = self.plugin.score(STUFFED_RESPONSE)
        self.assertGreaterEqual(correct, 15.0)
        self.assertLess(stuffed, correct)

    def test_table_endpoints_are_counted(self):
        # Markdown-table endpoint rows ("| GET | /api/... |") must count toward
        # the API sub-score; the Data Model section is deliberately empty of
        # data keywords so only the API sub-part can contribute.
        text = (
            "## Data Model\n\n"
            "A blank section with no storage keywords.\n\n"
            "## API Design\n\n"
            "| Method | Endpoint |\n"
            "|---|---|\n"
            "| GET | /api/v1/sessions |\n"
            "| POST | /api/v1/sessions |\n"
            "| GET | /api/v1/calendar/events |\n"
        )
        rubric = {c["name"]: c for c in self.plugin.evaluate(text).rubric}
        self.assertGreaterEqual(rubric["Data Modeling & API Design"]["earned"], 1.0)

    def test_partial_response_scores(self):
        text = (
            "## Executive Summary\n\n"
            "We will build a modular monolith.\n\n"
            "## Requirements Summary\n\n"
            "Support web, iOS, Android. Real-time sync. OAuth2.\n\n"
            "## Architecture Style\n\n"
            "Modular monolith.\n\n"
            "## Component Description\n\n"
            "- Auth service\n"
            "- Calendar service\n"
            "- Music service\n"
            "- AI service\n"
            "- Analytics service\n\n"
            "## Data Model\n\n"
            "User, Session, FocusSession, CalendarEvent.\n\n"
            "## API Design\n\n"
            "REST endpoints:\n"
            "GET /api/v1/focus\n"
            "POST /api/v1/focus\n\n"
            "## Technology Stack\n\n"
            "Python, FastAPI, PostgreSQL, Redis, Docker.\n\n"
            "## Deployment Architecture\n\n"
            "Docker containers on AWS.\n\n"
            "## Security Considerations\n\n"
            "OAuth2, JWT, TLS.\n\n"
            "## Scalability & Performance\n\n"
            "Redis caching, load balancer.\n\n"
            "## Trade-offs & Decisions\n\n"
            "Modular monolith chosen for simpler deployment.\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_mid_tier_response_scores_in_middle_range(self):
        text = (
            "## Architecture\n\n"
            "We will use a microservices architecture with an API Gateway.\n\n"
            "## Data Model\n\n"
            "PostgreSQL for relational data. Redis for caching with TTL.\n\n"
            "## Real-Time Sync\n\n"
            "WebSockets for live updates and Kafka for events.\n\n"
            "## Scalability\n\n"
            "Kubernetes with horizontal pod autoscaling and a CDN.\n\n"
            "## Security\n\n"
            "OAuth2, JWT, and rate limiting.\n\n"
            "## Observability\n\n"
            "Prometheus and Grafana with 99.9% uptime target.\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 5.0)
        self.assertLess(score, 15.0)

    def test_full_response_scores_high(self):
        text = (
            "## Executive Summary\n\n"
            "FlowState will use an event-driven microservices architecture deployed on AWS, "
            "with a React/Flutter frontend, Python/FastAPI backend services, PostgreSQL for relational data, "
            "Redis for caching, and Kafka for event streaming.\n\n"
            "## Requirements Summary\n\n"
            "Functional: time-blocking, focus music, AI planning, calendar integration, push notifications, analytics.\n"
            "Non-functional: support 1M DAU, real-time sync, <2s latency, 99.9% uptime.\n\n"
            "## Architecture Style\n\n"
            "Event-driven microservices. This enables independent scaling of the AI scheduler, music service, and analytics.\n\n"
            "## Component Description\n\n"
            "```\n"
            "[Client Apps] -> [API Gateway] -> [Auth Service]\n"
            "                 -> [Calendar Service] -> [Google/Outlook APIs]\n"
            "                 -> [Music Service] -> [Spotify/Apple Music APIs]\n"
            "                 -> [AI Planning Service] -> [ML Pipeline]\n"
            "                 -> [Analytics Service] -> [Data Warehouse]\n"
            "                 -> [Notification Service] -> [Push Gateway]\n"
            "```\n\n"
            "## Data Model\n\n"
            "- User(id, email, oauth_provider)\n"
            "- Session(id, user_id, start_time, end_time, focus_score)\n"
            "- CalendarEvent(id, user_id, external_id, start_time, end_time)\n"
            "- Playlist(id, user_id, tracks, energy_level)\n"
            "- Schedule(id, user_id, date, blocks)\n\n"
            "## API Design\n\n"
            "REST API v1:\n"
            "- GET /api/v1/sessions — list focus sessions\n"
            "- POST /api/v1/sessions — start a focus session\n"
            "- GET /api/v1/calendar/events — list calendar events\n"
            "- POST /api/v1/plan — generate AI schedule\n"
            "- GET /api/v1/analytics — productivity trends\n\n"
            "## Technology Stack\n\n"
            "- Backend: Python, FastAPI, Celery\n"
            "- Frontend: React, Flutter\n"
            "- Databases: PostgreSQL, Redis, ClickHouse\n"
            "- Messaging: Kafka\n"
            "- Infrastructure: AWS, Docker, Kubernetes, Terraform\n\n"
            "## Deployment Architecture\n\n"
            "- CI/CD: GitHub Actions -> Docker build -> EKS deployment\n"
            "- Observability: Prometheus, Grafana, ELK\n"
            "- CDN: CloudFront for static assets\n\n"
            "## Security Considerations\n\n"
            "- OAuth2 / OIDC via Google and Microsoft\n"
            "- JWT access tokens with short expiry\n"
            "- TLS 1.3 for all traffic\n"
            "- RBAC for admin endpoints\n"
            "- Encryption at rest for PII\n\n"
            "## Scalability & Performance\n\n"
            "- Redis caching for hot data\n"
            "- Read replicas for PostgreSQL\n"
            "- Horizontal pod autoscaling on EKS\n"
            "- Kafka for async event processing\n"
            "- Rate limiting at API gateway\n\n"
            "## Trade-offs & Decisions\n\n"
            "Microservices add operational complexity but allow independent scaling of AI and music services. "
            "PostgreSQL chosen over NoSQL for strong consistency of scheduling data.\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 0.0)
        self.assertLess(score, self.plugin.max_score)

    def test_excellent_response_scores_near_max(self):
        text = (
            "## Executive Summary\n\n"
            "FlowState will be built as an event-driven microservices platform. "
            "An API Gateway routes traffic to bounded contexts (Auth, Calendar, Music, AI Planner, Analytics, Notifications). "
            "Real-time sync uses WebSockets; calendar conflicts are resolved with CRDTs.\n\n"
            "## Requirements Summary\n\n"
            "Support web, iOS, Android, calendar OAuth, AI planning, music, notifications, analytics, and 1M DAU.\n\n"
            "## Architecture Style\n\n"
            "Event-driven microservices with independently scalable bounded contexts.\n\n"
            "## Component Diagram / Description\n\n"
            "Auth, Calendar, Music, AI Planner, Analytics, and Notification services sit behind the gateway.\n\n"
            "## API Design\n\n"
            "GET /api/v1/sessions, POST /api/v1/sessions, GET /api/v1/calendar/events.\n\n"
            "## Technology Stack\n\n"
            "Python, FastAPI, PostgreSQL, Redis, Kafka, Docker, and Kubernetes.\n\n"
            "## Deployment Architecture\n\n"
            "AWS multi-region Kubernetes deployment with a CDN and monitored Kafka workers.\n\n"
            "## Trade-offs & Decisions\n\n"
            "Microservices add operational complexity but allow independent scaling and failure isolation.\n\n"
            "```mermaid\n"
            "graph TD\n"
            "  A[Client Apps] --> B[API Gateway]\n"
            "  B --> C[Auth Service]\n"
            "  B --> D[Calendar Service]\n"
            "  B --> E[Music Service]\n"
            "  B --> F[AI Planner Service]\n"
            "  B --> G[Analytics Service]\n"
            "  B --> H[Notification Service]\n"
            "```\n\n"
            "## Data Model & Strategy\n\n"
            "Relational data (users, sessions, schedules) lives in PostgreSQL with read replicas. "
            "Time-series focus metrics and event streams go to a columnar store. "
            "Sharding by user_id partitions data, and Redis provides read-aside caching with TTL.\n\n"
            "## Real-Time Sync & Communication\n\n"
            "WebSockets push live focus sessions to clients. Calendar updates use CRDTs for offline-first "
            "conflict resolution. Kafka acts as the event bus for async processing.\n\n"
            "## Scalability & Capacity Planning\n\n"
            "At 1M DAU we estimate ~2,300 concurrent users and ~1,150 RPS at peak. Kubernetes HPA scales "
            "pods horizontally. A CDN (CloudFront) serves static assets from edge locations.\n\n"
            "## Resiliency & Failure Modes\n\n"
            "Circuit breakers protect downstream calendar and music APIs. Failed Kafka messages go to a "
            "dead-letter queue. Exponential backoff with jitter retries transient errors.\n\n"
            "## Security & Protections\n\n"
            "OAuth2/OIDC with refresh-token rotation, rate limiting and a WAF at the API gateway, "
            "encryption at rest for PII, and GDPR-compliant data retention.\n\n"
            "## Observability & SLOs\n\n"
            "OpenTelemetry tracing, Prometheus metrics, and Grafana dashboards. SLO: 99.9% availability, "
            "p99 latency < 200 ms.\n"
        )
        score = self.plugin.score(text)
        self.assertGreater(score, 15.0)
        self.assertLessEqual(score, self.plugin.max_score)


if __name__ == "__main__":
    unittest.main()
