"""MoE versus dense architecture analysis challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._analysis import first_section, markdown_sections
from plugins.challenges._rubric import Rubric


class MoEDensePlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "moe-dense"

    @property
    def version(self) -> str:
        return "1.0.1"

    @property
    def name(self) -> str:
        return "MoE vs Dense"

    @property
    def max_score(self) -> int:
        return int(17.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self):
        return (
            "Write a technical comparison with headings Gating, Load Balancing, Training, "
            "Inference, Benchmarks, and References. Include a top-k routing equation, a load "
            "balancing equation with defined variables, two concrete MoE advantages and two "
            "concrete dense advantages tied to named tasks/models, two named references, and "
            "specific numeric trade-offs. Do not merely list keywords."
        )

    def get_temperature(self, global_config):
        return global_config.get("moe_dense_temperature")

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        sections = markdown_sections(text)
        rubric.record_validation(type("Validation", (), {
            "valid": len(sections) >= 4,
            "evidence": [{"kind": "section", "heading": section.heading} for section in sections],
            "errors": [],
        })())
        gating = first_section(text, ["Gating", "Routing"])
        load = first_section(text, ["Load Balancing", "Auxiliary Loss"])
        training = first_section(text, ["Training"])
        inference = first_section(text, ["Inference"])
        benchmarks = first_section(text, ["Benchmarks", "Comparison"])
        references = first_section(text, ["References", "Papers"])
        gating_text = gating.body if gating else ""
        load_text = load.body if load else ""
        required_sections = {
            "gating", "load balancing", "training", "inference", "benchmarks", "references",
        }
        present_sections = {section.normalized for section in sections}
        matched_sections = {
            required: next((present for present in present_sections if required in present), None)
            for required in required_sections
        }
        section_hits = sum(value is not None for value in matched_sections.values())
        rubric.add_criterion(
            "Required comparison sections", 2.0,
            2.0 * section_hits / len(required_sections),
            evidence=[{"kind": "section", "name": name, "heading": heading}
                      for name, heading in matched_sections.items() if heading],
            negative_findings=[{"finding": f"missing section: {name}"}
                              for name, heading in matched_sections.items() if heading is None],
        )
        gate_ok = bool(re.search(r"top\s*-?\s*k", gating_text, re.IGNORECASE) and re.search(r"softmax", gating_text, re.IGNORECASE) and re.search(r"(?:router|gate|expert)", gating_text, re.IGNORECASE) and re.search(r"(?:=|equation|formula)", gating_text, re.IGNORECASE))
        rubric.add_criterion("Gating/routing mechanism", 3.0, 3.0 if gate_ok else 0.0, negative_findings=[] if gate_ok else [{"finding": "section must contain top-k, softmax, and an equation"}])
        load_ok = bool(re.search(r"(?:load.?balanc|auxiliary)", load_text, re.IGNORECASE) and re.search(r"(?:f[_\s]?i|p[_\s]?i|importance|capacity)", load_text, re.IGNORECASE) and re.search(r"(?:=|equation|formula|L[_\s]?aux)", load_text, re.IGNORECASE))
        rubric.add_criterion("Load-balancing loss", 3.0, 3.0 if load_ok else 0.0)
        training_hits = sum(bool(re.search(pattern, (training.body if training else ""), re.IGNORECASE)) for pattern in (r"token.?drop", r"expert.?collapse|instab|capacity"))
        rubric.add_criterion("Training challenges", 2.0, float(training_hits))
        inference_hits = sum(bool(re.search(pattern, (inference.body if inference else ""), re.IGNORECASE)) for pattern in (r"memory|bandwidth|parallel|latency|throughput|compute"))
        rubric.add_criterion("Inference implications", 2.0, min(2.0, float(inference_hits)))
        benchmark_text = benchmarks.body if benchmarks else ""
        advantage_pairs = len(re.findall(r"(?:moe|mixture.of.experts).{0,150}(?:outperform|better|advantage|wins).{0,150}(?:dense|task|model)", benchmark_text, re.IGNORECASE))
        dense_pairs = len(re.findall(r"dense.{0,150}(?:outperform|better|advantage|wins).{0,150}(?:moe|mixture.of.experts|task|model)", benchmark_text, re.IGNORECASE))
        named_tasks = len(set(re.findall(r"\b(?:MMLU|GSM8K|HumanEval|MBPP|HellaSwag|ARC|coding|translation|classification)\b", benchmark_text, re.IGNORECASE)))
        rubric.add_criterion("Benchmarks/comparison", 2.0, 2.0 if advantage_pairs >= 2 and dense_pairs >= 2 and named_tasks >= 2 else min(2.0, float(advantage_pairs + dense_pairs) / 2.0))
        ref_text = references.body if references else ""
        refs = set(re.findall(r"\b(?:Mixtral|Switch Transformer|Shazeer|GLaM|DeepSeekMoE|Fedus|Sparsely-Gated|arXiv|technical report)\b", ref_text, re.IGNORECASE))
        rubric.add_criterion("Paper references", 2.0, 2.0 if len(refs) >= 2 else float(len(refs)))
        numeric = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|B|M|K|TFLOPs?|params?|tokens?/s)\b", text, re.IGNORECASE)
        side_by_side = bool(re.search(r"\b(?:moe|dense)\b.{0,120}\b(?:vs\.?|versus|compared|than)\b.{0,120}\b(?:moe|dense)\b", text, re.IGNORECASE))
        rubric.add_criterion("Quantitative trade-off", 3.0, 3.0 if len(numeric) >= 2 and side_by_side else min(3.0, float(len(numeric))))
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
