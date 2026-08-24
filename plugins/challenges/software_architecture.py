"""Structured software architecture challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._analysis import exact_section, markdown_sections
from plugins.challenges._rubric import Rubric


class SoftwareArchitecturePlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "software-architecture"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "Software Architecture"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Produce a FlowState architecture document with headings Executive Summary, Requirements Summary, "
            "Architecture Style, Component Diagram / Description, Data Model, API Design, Technology Stack, "
            "Deployment Architecture, Security Considerations, Scalability & Performance, Trade-offs & Decisions. "
            "The design supports web/iOS/Android, calendar OAuth, realtime sync, AI planning, music, notifications, "
            "analytics, and 1M DAU. Include concrete entities/endpoints, capacity numbers, failure handling, and rationale."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("software_architecture_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def evaluate(self, response_text: str) -> EvaluationResult:
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) >= 8,
            "evidence": [{"kind": "section", "heading": section.heading} for section in sections],
            "errors": [],
        })())
        section_aliases = {
            "Executive Summary": (),
            "Requirements Summary": (),
            "Architecture Style": ("Architecture",),
            "Component Diagram / Description": ("Component Description", "Component Diagram"),
            "Real-Time Sync & Communication": ("Real-Time Sync", "Realtime", "Communication"),
            "Data Model": ("Data Model & Strategy", "Data Design", "Persistence"),
            "API Design": (),
            "Technology Stack": (),
            "Deployment Architecture": (),
            "Resiliency & Failure Modes": ("Resilience", "Failure Modes"),
            "Security Considerations": ("Security & Protections", "Security", "Privacy"),
            "Scalability & Performance": ("Scalability & Capacity Planning", "Scalability", "Capacity", "Performance"),
            "Trade-offs & Decisions": ("Trade-offs",),
            "Observability & SLOs": ("Observability", "SLO", "Monitoring"),
        }
        matched = {
            name: exact_section(text, name, aliases)
            for name, aliases in section_aliases.items()
        }
        rubric.add_criterion(
            "Required architecture sections", 3.0,
            3.0 * sum(section is not None for section in matched.values()) / len(matched),
            evidence=[{"kind": "section", "heading": name} for name, section in matched.items() if section],
            negative_findings=[{"finding": f"missing section: {name}"} for name, section in matched.items() if section is None],
        )
        def body(name: str) -> str:
            section = matched[name]
            return section.body if section else ""
        architecture = " ".join((body("Architecture Style"), body("Component Diagram / Description")))
        data = body("Data Model")
        api = body("API Design")
        realtime = body("Real-Time Sync & Communication")
        scale = body("Scalability & Performance")
        resilience = body("Resiliency & Failure Modes")
        security = body("Security Considerations")
        observability = body("Observability & SLOs")
        arch_hits = sum(bool(re.search(pattern, architecture, re.IGNORECASE)) for pattern in (r"microservices|modular monolith|event.?driven|serverless", r"api gateway|service|component", r"```(?:mermaid|plantuml)|graph TD|->|responsibilit",))
        rubric.add_criterion("Architecture & Patterns", 2.5, min(2.5, float(arch_hits)))
        data_hits = sum(bool(re.search(pattern, data, re.IGNORECASE)) for pattern in (r"postgres|sql|relational", r"nosql|document|columnar|mongodb|dynamodb|time.?series", r"entity|user|session|schedule", r"shard|replica|partition|cache|redis|ttl"))
        endpoint_count = len(re.findall(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/\S+", api, re.IGNORECASE))
        data_score = min(1.5, data_hits * 0.375) + min(1.0, endpoint_count / 3.0)
        rubric.add_criterion("Data Modeling & API Design", 2.5, data_score)
        realtime_hits = sum(bool(re.search(pattern, realtime, re.IGNORECASE)) for pattern in (r"websocket|sse|grpc|polling|sync", r"crdt|conflict|offline|eventual|last.?write", r"kafka|queue|broker|pub.?sub|event bus"))
        rubric.add_criterion("Real-Time Sync & Communication", 2.5, min(2.5, float(realtime_hits)))
        scale_hits = sum(bool(re.search(pattern, scale, re.IGNORECASE)) for pattern in (r"rps|qps|requests per second|million|1[\s,]*000[\s,]*000", r"auto.?scal|kubernetes|horizontal|hpa", r"cdn|edge|multi.?region|load balanc"))
        rubric.add_criterion("Scalability & Capacity Planning", 2.5, min(2.5, float(scale_hits)))
        resilience_hits = sum(bool(re.search(pattern, resilience, re.IGNORECASE)) for pattern in (r"circuit breaker|bulkhead|failover|idempot|disaster recovery", r"dead.?letter|dlq|retry queue|message retry", r"backoff|jitter|fallback|graceful degradation|timeout"))
        rubric.add_criterion("Resiliency & Failure Modes", 2.5, min(2.5, float(resilience_hits)))
        security_hits = sum(bool(re.search(pattern, security, re.IGNORECASE)) for pattern in (r"oauth|oidc|mfa|refresh token|authorization", r"rate limit|waf|ddos|throttl", r"pii|gdpr|encryption|kms|secret"))
        rubric.add_criterion("Security & Protections", 2.5, min(2.5, float(security_hits)))
        obs_hits = sum(bool(re.search(pattern, observability, re.IGNORECASE)) for pattern in (r"opentelemetry|tracing|prometheus|metrics|logging|monitoring", r"slo|sli|sla|error budget|99\.\d%"))
        rubric.add_criterion("Observability & SLOs", 1.5, min(1.5, float(obs_hits)))
        if re.search(r"1\s*(?:million|000\s*000)|1M", scale, re.IGNORECASE) and not re.search(r"(?:rps|qps|capacity|throughput)", scale, re.IGNORECASE):
            rubric.penalize_criterion("Scalability & Capacity Planning", 0.5, "capacity claim lacks a workload estimate")
        if re.search(r"99\.\d%|SLO|SLA", observability, re.IGNORECASE) and not re.search(r"circuit|failover|retry|backoff|multi.?region", resilience, re.IGNORECASE):
            rubric.penalize_criterion("Observability & SLOs", 0.5, "availability target lacks supporting failure handling")
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
