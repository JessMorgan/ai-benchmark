import io
import json
import os
import tempfile
import unittest
from unittest import mock

from benchmark.completions import build_parser
from benchmark.core import resolve_targets
from benchmark.logs import iter_log_members
from benchmark.pi import PiProcessResult, _node_version, run_process
from benchmark.transport import TransportRequest, execute_transport


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def close(self):
        return None


class _FakeProcess:
    def __init__(self, stdout, stderr="", returncode=0):
        self.stdin = _FakeStdin()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.pid = os.getpid() + 100000

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        return None

    def kill(self):
        return None


class TestPiAdapter(unittest.TestCase):
    def test_node_version_parser(self):
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(stdout="v22.19.1\n", stderr="")
            self.assertEqual(_node_version("node"), (22, 19, 1))

    def test_run_process_extracts_ndjson_and_keeps_artifacts(self):
        events = "".join(
            json.dumps({"protocol": "pi-worker-v1", "event": event, "attempt": 1, "data": data})
            + "\n"
            for event, data in (
                ("worker_started", {"worker_version": "1.0.0", "sdk_version": "@earendil-works/pi-coding-agent@0.84.2"}),
                ("reasoning_delta", {"text": "think"}),
                ("text_delta", {"text": "answer"}),
                ("usage", {"usage": {"output": 2, "totalTokens": 3}}),
                ("finish", {"finish_reason": "stop", "usage": {"output": 2}, "tools": ["read"]}),
            )
        )
        fake = _FakeProcess(events, "worker diagnostic\n")
        content = []
        thinking = []
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("benchmark.pi.resolve_pi_worker", return_value=("node", "/tmp/worker.mjs")), \
                mock.patch("benchmark.pi.subprocess.Popen", return_value=fake):
            result = run_process(
                "prompt",
                source_config={"Local": {"api_url": "http://localhost:11434/v1/chat/completions"}},
                source="Local",
                api_model="model",
                max_tokens=32,
                timeout=2,
                pi_config={"tools": ["read"], "permissions": {"read": "allow"}},
                output_dir=tmpdir,
                target_key="model",
                plugin_id="test",
                observer=mock.Mock(chunk=content.append, think_chunk=thinking.append),
                debug_logs=True,
            )

            self.assertEqual(result.text, "answer")
            self.assertEqual(result.think_text, "think")
            self.assertEqual(result.error, None)
            self.assertEqual(result.tools, ("read",))
            self.assertEqual(result.requested_tools, ("read",))
            self.assertEqual(result.permissions, {"read": "allow"})
            self.assertEqual(content, ["answer"])
            self.assertEqual(thinking, ["think"])
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "logs", "model", "test.stdout.ndjson.gz")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "logs", "model", "test.stderr.txt.gz")))
            request = json.loads(fake.stdin.writes[0])
            self.assertEqual(request["prompt"], "prompt")
            self.assertEqual(request["prompt_altered"], "none")
            self.assertEqual(request["max_tokens"], 32)
            self.assertEqual(request["tools"], ["read"])
            self.assertIn(b"worker diagnostic", b"".join(
                iter_log_members(os.path.join(tmpdir, "logs", "model", "test.stderr.txt.gz"))
            ))

    def test_compact_pi_run_does_not_write_transcripts(self):
        fake = _FakeProcess("", "")
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("benchmark.pi.resolve_pi_worker", return_value=("node", "/tmp/worker.mjs")), \
                mock.patch("benchmark.pi.subprocess.Popen", return_value=fake):
            run_process(
                "prompt", source_config={"Local": {}}, source="Local",
                api_model="model", max_tokens=32, timeout=2,
                output_dir=tmpdir, target_key="model", plugin_id="test",
            )
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "logs")))


class TestPiTransport(unittest.TestCase):
    def test_pi_result_is_normalized_with_runner_metadata(self):
        process_result = PiProcessResult(
            text="answer",
            think_text="think",
            stderr="",
            elapsed=1.5,
            error=None,
            returncode=0,
            finish_reason="stop",
            usage={"reasoning_tokens": 4},
            tool_called=True,
            tools=("read",),
            requested_tools=("read",),
            permissions={"read": "allow"},
            provider="local",
        )
        request = TransportRequest(
            prompt="prompt",
            max_tokens=32,
            source_config={"Local": {"api_url": "http://localhost/v1"}},
            api_model="model",
            source="Local",
            timeout=5,
            transport="pi",
            pi_config={"tools": ["read"]},
        )
        with mock.patch("benchmark.transport.run_pi_process", return_value=process_result):
            result = execute_transport(request)
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.response_nature, "completed")
        self.assertEqual(result.thinking_tokens, 4)
        self.assertEqual(result.runner_metadata["runner"], "pi")
        self.assertEqual(result.runner_metadata["requested_tools"], ["read"])
        self.assertTrue(result.runner_metadata["tool_called"])


class TestPiConfigurationAndSelection(unittest.TestCase):
    def test_pi_config_is_validated_and_preserved(self):
        cfg = {
            "max_tokens": 100,
            "sources": {"Local": {"api_url": "http://localhost/v1"}},
            "models": {
                "model": {
                    "source": "Local",
                    "pi": {
                        "tools": ["read", "grep"],
                        "permissions": {"read": "allow", "grep": "deny"},
                        "reasoning": True,
                        "max_tool_calls": 3,
                    },
                }
            },
        }
        target = resolve_targets(cfg)["model"]
        self.assertEqual(target["pi"]["tools"], ["read", "grep"])
        self.assertEqual(target["pi"]["max_tool_calls"], 3)

    def test_pi_config_rejects_unknown_tools_and_keys(self):
        base = {"sources": {"Local": {}}, "models": {"model": {"source": "Local"}}}
        for pi in ({"tools": ["network"]}, {"not_a_pi_setting": True}):
            cfg = {**base, "models": {"model": {"source": "Local", "pi": pi}}}
            with self.assertRaises(ValueError):
                resolve_targets(cfg)

    def test_explicit_multi_runner_list_is_accepted_by_parser(self):
        args = build_parser().parse_args(["--runners", "pi,http"])
        self.assertEqual(args.runners, "pi,http")
        self.assertEqual(build_parser().parse_args(["--runner", "pi"]).runner, "pi")
        self.assertTrue(build_parser().parse_args(["--pi-probe"]).pi_probe)


if __name__ == "__main__":
    unittest.main()
