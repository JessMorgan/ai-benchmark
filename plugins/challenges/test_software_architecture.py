"""Tests for the software architecture challenge plugin."""
import unittest

from plugins import discover_plugins


class TestSoftwareArchitectureScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = next(p for p in discover_plugins() if p.id == "software-architecture")

    def test_empty_response_scores_zero(self):
        self.assertEqual(self.plugin.score(""), 0.0)

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
        self.assertGreater(score, 10.0)


if __name__ == "__main__":
    unittest.main()
