"""Tests for newly added capability-focused challenges."""
from plugins.challenges.error_recovery import ErrorRecoveryPlugin
from plugins.challenges.event_processor import EventProcessorPlugin
from plugins.challenges.long_context import LongContextPlugin
from plugins.challenges.rate_limiter import RateLimiterPlugin


def test_event_processor_requires_behavioral_contract():
    response = "```python\nclass EventProcessor:\n    pass\n```"
    assert EventProcessorPlugin().score(response) < 10.0


def test_event_processor_complete_implementation_scores_high():
    response = '''```python
import threading
from concurrent.futures import ThreadPoolExecutor
class EventProcessor:
    """Idempotent retrying event processor."""
    def __init__(self, handler, max_workers=4, max_retries=2):
        if not callable(handler): raise TypeError("handler must be callable")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool): raise TypeError("max_workers must be an int")
        if not callable(handler): raise TypeError("handler must be callable")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool):
            raise TypeError("max_workers must be an int")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise TypeError("max_retries must be an int")
        if max_workers < 1: raise ValueError("max_workers must be >= 1")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool): raise TypeError("max_retries must be an int")
        if max_retries < 0: raise ValueError("max_retries must be >= 0")
        self.handler, self.max_workers, self.max_retries = handler, max_workers, max_retries
        self._completed, self._state_lock = set(), threading.Lock()
    def process(self, events):
        seen, unique, duplicates = set(), [], []
        for event in events:
            if not isinstance(event, dict): raise ValueError("event must be a dict")
            event_id = event.get("id")
            if not isinstance(event_id, str) or not event_id: raise ValueError("invalid event id")
            with self._state_lock:
                already_done = event_id in seen or event_id in self._completed
            if already_done:
                duplicates.append(event_id)
            else:
                seen.add(event_id); unique.append(event)
        def run(event):
            for attempt in range(self.max_retries + 1):
                try: self.handler(event); return event["id"], None
                except Exception as exc:
                    if attempt == self.max_retries: return event["id"], exc
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            outcomes = list(pool.map(run, unique))
        with self._state_lock:
            for key, error in outcomes:
                if error is None: self._completed.add(key)
        return {"processed": [key for key, error in outcomes if error is None], "duplicates": duplicates, "failed": [key for key, error in outcomes if error is not None]}
```'''
    assert EventProcessorPlugin().score(response) >= 18.0


def test_rate_limiter_complete_three_strategy_implementation_scores_high():
    response = '''```python
import math
import threading
class _Base:
    """Deterministic thread-safe window limiter."""
    def __init__(self, limit: int, window_seconds: float):
        if limit <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0: raise ValueError("invalid")
        self.limit, self.window_seconds, self.counts, self.lock = limit, window_seconds, {}, threading.RLock()
    def allow_request(self, client_id: str, now: float) -> bool:
        with self.lock:
            window, count = self.counts.get(client_id, (now, 0))
            if now - window >= self.window_seconds: window, count = now, 0
            if count >= self.limit: self.counts[client_id] = (window, count); return False
            self.counts[client_id] = (window, count + 1); return True
    def get_usage_stats(self, client_id: str) -> dict:
        with self.lock: return {"count": self.counts.get(client_id, (0, 0))[1], "limit": self.limit}
    def cleanup(self, now: float) -> int:
        with self.lock:
            old = [key for key, (start, _) in self.counts.items() if now - start >= self.window_seconds]
            for key in old: del self.counts[key]
            return len(old)
class TokenBucket(_Base): pass
class SlidingWindowLog(_Base): pass
class FixedWindow(_Base): pass
```'''
    assert RateLimiterPlugin().score(response) >= 18.0


def test_rate_limiter_delegation_exempt_for_any_shared_base_name():
    response = """```python
import math
import threading
class RateLimiterBase:
    def __init__(self, limit: int, window_seconds: float):
        if limit <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0: raise ValueError("invalid")
        self.limit, self.window_seconds, self.counts, self.lock = limit, window_seconds, {}, threading.RLock()
    def allow_request(self, client_id: str, now: float) -> bool:
        with self.lock:
            window, count = self.counts.get(client_id, (now, 0))
            if now - window >= self.window_seconds: window, count = now, 0
            if count >= self.limit: self.counts[client_id] = (window, count); return False
            self.counts[client_id] = (window, count + 1); return True
    def get_usage_stats(self, client_id: str) -> dict:
        with self.lock: return {"count": self.counts.get(client_id, (0, 0))[1], "limit": self.limit}
    def cleanup(self, now: float) -> int:
        with self.lock:
            old = [key for key, (start, _) in self.counts.items() if now - start >= self.window_seconds]
            for key in old: del self.counts[key]
            return len(old)
class TokenBucket(RateLimiterBase): pass
class SlidingWindowLog(RateLimiterBase): pass
class FixedWindow(RateLimiterBase): pass
```"""
    result = RateLimiterPlugin().evaluate(response)
    # The pass-body subclasses delegate to a real (non-stub) base whose name
    # is not the hardcoded "_Base"; they must NOT be penalized as placeholders.
    for criterion in ("TokenBucket", "SlidingWindowLog", "FixedWindow"):
        item = next(item for item in result.rubric if item["name"] == criterion)
        assert not any("placeholder" in finding["finding"] for finding in item["negative_findings"])


def test_rate_limiter_bool_cleanup_return_is_not_full_behavioral_credit():
    response = """```python
import math
import threading
class _Base:
    def __init__(self, limit: int, window_seconds: float):
        if limit <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0: raise ValueError("invalid")
        self.limit, self.window_seconds, self.counts, self.lock = limit, window_seconds, {}, threading.RLock()
    def allow_request(self, client_id: str, now: float) -> bool:
        with self.lock:
            window, count = self.counts.get(client_id, (now, 0))
            if now - window >= self.window_seconds: window, count = now, 0
            if count >= self.limit: self.counts[client_id] = (window, count); return False
            self.counts[client_id] = (window, count + 1); return True
    def get_usage_stats(self, client_id: str) -> dict:
        with self.lock: return {"count": self.counts.get(client_id, (0, 0))[1], "limit": self.limit}
    def cleanup(self, now: float) -> int:
        with self.lock:
            old = [key for key, (start, _) in self.counts.items() if now - start >= self.window_seconds]
            for key in old: del self.counts[key]
            return True
class TokenBucket(_Base): pass
class SlidingWindowLog(_Base): pass
class FixedWindow(_Base): pass
```"""
    result = RateLimiterPlugin().evaluate(response)
    item = next(item for item in result.rubric if item["name"] == "Behavioral strategy tests")
    # bool is a subclass of int, so a cleanup returning a bool must not pass
    # the return-type check; the behavioral criterion must score 0.
    assert item["earned"] == 0.0


def test_error_recovery_missing_injection_is_not_full_credit():
    response = """```python
class AllProvidersFailedError(Exception):
    pass
async def get_weather_resilient(city: str) -> dict:
    raise RuntimeError('bad')
```"""
    assert ErrorRecoveryPlugin().score(response) < 15.0


def test_error_recovery_complete_injectable_implementation_scores_high():
    response = '''```python
import asyncio
import logging
class AllProvidersFailedError(Exception):
    """Raised after every provider fails."""
class WeatherClient:
    async def fetch(self, provider: str, city: str) -> dict:
        raise NotImplementedError
async def get_weather_resilient(city: str, client: WeatherClient) -> dict:
    providers = ["WeatherAPI", "OpenMeteo", "VisualCrossing"]
    async def call(provider):
        try:
            value = await asyncio.wait_for(client.fetch(provider, city), 1)
            if not isinstance(value, dict) or "error" in value: raise ValueError("bad payload")
            return provider, value, None
        except Exception as exc:
            logging.getLogger(__name__).error("%s failed: %s", provider, exc)
            return provider, None, exc
    results = await asyncio.gather(*(call(provider) for provider in providers))
    for provider, value, error in results:
        if error is None: return value
    raise AllProvidersFailedError("; ".join(provider for provider, _, _ in results))
async def demo():
    """Show all succeed, partial failure, and all failure scenarios."""
    return ["all providers succeed", "one provider fails", "all providers fail"]
```'''
    assert ErrorRecoveryPlugin().score(response) >= 18.0


def test_long_context_retrieves_and_cross_references_facts():
    response = """INCIDENT: I-17
OWNER: Omar
ESCALATION CHANNEL: PagerDuty
EVIDENCE: F02, F05, F09
REASONING: F02 identifies I-17 in EU at 14:30 with P1; F05 links its runbook; F09 maps P1 to PagerDuty.
"""
    assert LongContextPlugin().score(response) == 20.0


def test_long_context_does_not_accept_only_the_final_guess():
    response = "INCIDENT: I-17\nOWNER: Omar\nESCALATION CHANNEL: PagerDuty\n"
    assert LongContextPlugin().score(response) < 12.0
