import unittest

from benchmark.observer import TaskObserver


class TestTaskObserver(unittest.TestCase):
    def test_forwards_content_thinking_and_retry_events(self):
        events = []
        observer = TaskObserver(
            model_name="model",
            pid="plugin",
            on_chunk=lambda value: events.append(("content", value)),
            on_think_chunk=lambda value: events.append(("thinking", value)),
            on_retry=lambda: events.append(("retry", None)),
        )

        observer.chunk("answer")
        observer.think_chunk("reasoning")
        observer.retry()

        self.assertEqual(events, [
            ("content", "answer"),
            ("thinking", "reasoning"),
            ("retry", None),
        ])

    def test_callback_failures_are_isolated(self):
        observer = TaskObserver(
            on_chunk=lambda _value: (_ for _ in ()).throw(RuntimeError("content")),
            on_think_chunk=lambda _value: (_ for _ in ()).throw(RuntimeError("thinking")),
            on_retry=lambda: (_ for _ in ()).throw(RuntimeError("retry")),
        )

        observer.chunk("answer")
        observer.think_chunk("reasoning")
        observer.retry()

    def test_noop_observer_is_safe(self):
        observer = TaskObserver.noop()
        observer.chunk("answer")
        observer.think_chunk("reasoning")
        observer.retry()
