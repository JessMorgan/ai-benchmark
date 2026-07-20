"""MoE vs Dense architecture analysis benchmark task."""

from benchmark_plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric


class MoEDensePlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "moe-dense"

    @property
    def version(self):
        return "0.1.0"

    @property
    def name(self):
        return "MoE vs Dense"

    @property
    def max_score(self):
        return 15.0

    @property
    def supports_streaming(self):
        return False

    def get_prompt(self):
        return (
            "Write a detailed technical analysis comparing Mixture-of-Experts (MoE) architecture "
            "(used in Mixtral 8x7B, Qwen3-MoE, DeepSeekMoE) versus dense transformer architecture "
            "(used in Llama 3, Gemma, GPT-4o). Your analysis must cover:\n\n"
            "- The mathematical formulation of the sparse MoE gating/routing mechanism (include the top-k "
            "routing equation and softmax gating)\n"
            "- How the auxiliary load-balancing loss works — include the exact mathematical formulation\n"
            "- At least 2 specific training stability challenges unique to MoE (token dropping, expert "
            "collapse, or others)\n"
            "- Inference implications: memory bandwidth, expert parallelism, vs dense compute patterns\n"
            "- Specific benchmarks or tasks where MoE outperforms dense architectures, and where dense "
            "outperforms MoE (name at least 2 of each)\n"
            "- Reference at least 2 specific papers, technical reports, or model cards\n\n"
            "Be precise and technical — this is for an ML engineering audience. 4-5 paragraphs."
        )

    def get_temperature(self, global_config):
        if "moe_dense_temperature" in global_config:
            return global_config["moe_dense_temperature"]
        return None

    def evaluate(self, response_text):
        t = response_text.lower()
        rubric = Rubric(self.max_score)

        rubric.eval_regex(
            "Covers both architectures",
            2.0,
            t,
            [
                (r'(?:mixture.of.expert|moe|sparse.*moe)', 1.0),
                (r'(?:dense\s*(?:transformer|model|architecture)|standard\s*transformer)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Gating/routing mechanism",
            2.5,
            t,
            [
                (r'(?:gating|routing|gate|router|top.k|softmax.*gate)', 1.5),
                (r'(?:expert.*select|which.*expert|rout.*token)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Load-balancing loss",
            2.5,
            t,
            [
                (r'(?:load.balanc|auxiliary.*loss|aux.*loss|balance.*loss)', 1.5),
                (r'(?:importance|loss.*formula|load.*equation|L_aux)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Training challenges",
            2.0,
            t,
            [
                (r'(?:token.dropp|expert.collaps|instability|collapse|dropping)', 1.0),
                (r'(?:training.*challeng|difficult|problem|issue|stability)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Inference implications",
            2.0,
            t,
            [
                (r'(?:inference|memory.*bandwidth|expert.*parallel|sparse.*compute)', 1.0),
                (r'(?:throughput|latency|batch.*size|parameter.*efficien)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Benchmarks/comparison",
            2.0,
            t,
            [
                (r'(?:benchmark|mmlu|gsm8k|human-eval|mbpp|hellaswag|arc|truthful)', 1.0),
                (r'(?:outperform|better.*than|compared to|vs\.|versus|advantage)', 1.0),
            ],
        )

        rubric.eval_regex(
            "Paper references",
            2.0,
            t,
            [
                (r'(?:paper|report|arxiv|technical.*report)', 1.0),
                (r'(?:2023|2024|2025|et\s*al|vashwani|shazeer|fedus|lepikhin|du et al)', 1.0),
            ],
        )

        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text)[0]
