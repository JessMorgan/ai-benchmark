"""Rate limiter code-generation benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric


class RateLimiterPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "rate-limiter"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Rate Limiter"

    @property
    def max_score(self):
        return 20.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "Design and implement a concurrent rate limiter in Python with the following specifications:\n\n"
            "ARCHITECTURE:\n"
            "- Support multiple rate limiting strategies: Token Bucket, Sliding Window Log, and Fixed Window\n"
            "- Use a clean abstract base class or protocol that all strategies implement\n"
            "- Thread-safe throughout with minimal lock contention\n\n"
            "REQUIREMENTS:\n"
            "- Configurable per-client limits (each client_id can have different rate limits)\n"
            "- Method: allow_request(client_id: str) -> bool — returns True if request is allowed\n"
            "- Method: get_usage_stats(client_id: str) -> dict — returns current usage info\n"
            "- Efficient automatic cleanup of stale client entries to prevent memory leaks\n"
            "- Handle edge cases: burst traffic, zero limits, concurrent requests from same client\n\n"
            "IMPLEMENTATION:\n"
            "- Implement at least TokenBucket and SlidingWindowLog strategies fully\n"
            "- Token Bucket: tokens refill at a configurable rate, burst capacity = bucket size\n"
            "- Sliding Window Log: track timestamps of recent requests, O(log n) for inserts\n"
            "- Fixed Window: simple counter per time window, resets at window boundary\n\n"
            "Return the complete implementation with all classes and a usage example."
        )

    def get_temperature(self, global_config):
        if "rate_limiter_temperature" in global_config:
            return global_config["rate_limiter_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        rubric = Rubric(self.max_score)

        earned = 0.0
        if re.search(r'(?:ABC|abstractmethod|Protocol|ABCMeta)', t):
            earned += 2.0
        elif re.search(r'(?:class\s+\w+RateLimiter|class\s+Base)', t):
            earned += 1.0
        if re.search(r'allow_request', t) and re.search(r'get_usage_stats|get_usage|usage_stats', t):
            earned += 1.0
        rubric.add_criterion("Interface design", 3.0, earned)

        earned = 0.0
        if re.search(r'(?:class\s+TokenBucket|TokenBucket)', t):
            earned += 1.0
        if re.search(r'(?:refill|last_refill|_refill|tokens\s*[+=-])', t):
            earned += 1.5
        if re.search(r'(?:consume|allow|acquire|try_acquire)', t) and re.search(r'(?:tokens?\s*[>=-]|if\s+\w+\s*[>=-])', t):
            earned += 1.5
        rubric.add_criterion("Token Bucket", 4.0, earned)

        earned = 0.0
        if re.search(r'(?:class\s+SlidingWindow|SlidingWindowLog)', t):
            earned += 1.0
        if re.search(r'(?:timestamp|time\.time|deque|list|sorted|bisect)', t) and re.search(r'(?:window|log|history)', t.lower()):
            earned += 1.0
        if re.search(r'(?:prune|clean|remove_old|pop.*while|while.*pop|deque.*popleft)', t):
            earned += 1.0
        rubric.add_criterion("Sliding Window", 3.0, earned)

        earned = 0.0
        if re.search(r'(?:threading\.Lock|threading\.RLock|from threading import)', t):
            earned += 1.5
        if re.search(r'(?:with\s+.*lock|with\s+.*mutex|\.acquire|\.release)', t, re.IGNORECASE):
            earned += 1.5
        rubric.add_criterion("Thread safety", 3.0, earned)

        earned = 0.0
        if re.search(r'(?:cleanup|clean_up|remove_stale|expire|ttl|timeout)', t.lower()):
            earned += 1.0
        if re.search(r'(?:background|thread.*clean|scheduler|Timer|loop.*clean)', t.lower()):
            earned += 1.0
        rubric.add_criterion("Cleanup/memory management", 2.0, earned)

        earned = 0.0
        if re.search(r'->\s*(?:bool|dict|int|None|str|float)', t):
            earned += 1.0
        if re.search(r':\s*(?:int|str|bool|float|dict|list|Optional|Callable|Type)', t):
            earned += 1.0
        rubric.add_criterion("Type hints", 2.0, earned)

        earned = 0.0
        if '"""' in t:
            earned += 1.0
        if t.count('"""') >= 4 or "'''" in t:
            earned += 1.0
        rubric.add_criterion("Docstrings", 2.0, earned)

        rubric.eval_regex(
            "Error handling",
            1.0,
            t,
            [(r'(?:raise\s+|try\s*:|except\s+|ValueError|TypeError|Invalid)', 1.0)],
        )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text)[0]
