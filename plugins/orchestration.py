"""Multi-step workflow orchestration benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


class OrchestrationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "orchestration"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "Orchestration & Workflow"

    @property
    def max_score(self):
        return 16.0

    @property
    def supports_streaming(self):
        return True

    def get_prompt(self):
        return (
            "You are an orchestration AI handling a complex data pipeline execution. Task:\n"
            "'Process 1TB of raw server logs, perform GeoIP lookup on IPs, run anomaly detection, "
            "and generate a final PDF report.'\n\n"
            "Produce a structured execution plan including:\n"
            "1. Task Decomposition: Break the prompt into distinct steps.\n"
            "2. Dependency Graph: Mark which steps are [PARALLEL] and which are [SEQUENTIAL], "
            "using explicit [DEPENDS_ON: task_id] tags.\n"
            "3. Execution Trace: Output a simulated trace showing the initialization, running, "
            "and completion states of each step.\n\n"
            "Use clear headings and concise logic."
        )

    def get_temperature(self, global_config):
        if "orchestration_temperature" in global_config:
            return global_config["orchestration_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text
        rubric = []
        s = 0.0

        # 1. Task breakdown presence (0-4)
        earned = 0.0
        steps = sum(1 for _ in re.finditer(r'(?:task|step)\s*\d+', t, re.IGNORECASE))
        if steps >= 3:
            earned += 2.0
        elif steps >= 2:
            earned += 1.0
        if re.search(r'(?:geoip|anomal|pdf|report|logs|server)', t, re.IGNORECASE):
            earned += 2.0
        earned = round(min(earned, 4.0), 1)
        s += earned
        rubric.append({"name": "Task breakdown presence", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        # 2. Explicit dependency tagging (0-4)
        earned = 0.0
        if re.search(r'\[DEPENDS_ON[^\]]*\]', t):
            earned = 4.0
        elif re.search(r'depends on', t, re.IGNORECASE):
            earned = 2.0
        s += earned
        rubric.append({"name": "Explicit dependency tagging", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        # 3. Parallel vs sequential logic (0-4)
        earned = 0.0
        has_parallel = re.search(r'\[PARALLEL\]', t) or re.search(r'\bparallel\b', t, re.IGNORECASE)
        has_sequential = re.search(r'\[SEQUENTIAL\]', t) or re.search(r'\bsequential\b', t, re.IGNORECASE)
        if has_parallel and has_sequential:
            earned = 4.0
        elif has_parallel or has_sequential:
            earned = 2.0
        s += earned
        rubric.append({"name": "Parallel vs sequential logic", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        # 4. State / execution trace simulation (0-4)
        earned = 0.0
        if re.search(r'(?:trace|execution|pipeline)', t, re.IGNORECASE):
            earned += 1.0
        has_start = re.search(r'(?:init|start|running|pending)', t, re.IGNORECASE)
        has_end = re.search(r'(?:complete|done|success|finish)', t, re.IGNORECASE)
        if has_start and has_end:
            earned += 3.0
        earned = round(min(earned, 4.0), 1)
        s += earned
        rubric.append({"name": "State / execution trace", "max": 4.0, "earned": earned, "missed": round(4.0 - earned, 1)})

        return round(min(s, self.max_score), 1), rubric

    def score(self, response_text):
        return self.evaluate(response_text)[0]
