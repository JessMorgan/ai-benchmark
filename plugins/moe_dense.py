"""MoE vs Dense architecture analysis benchmark task."""
import re

from benchmark_plugin import BenchmarkTaskPlugin


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
        # Backward-compatible legacy key
        if "general_temperature" in global_config:
            return global_config["general_temperature"]
        return None

    def score(self, response_text):
        t = response_text.lower()
        s = 0.0

        # 1. Covers both architectures explicitly (0-2)
        if re.search(r'(?:mixture.of.expert|moe|sparse.*moe)', t):
            s += 1.0
        if re.search(r'(?:dense\s*(?:transformer|model|architecture)|standard\s*transformer)', t):
            s += 1.0

        # 2. Gating/routing mechanism (0-2.5)
        if re.search(r'(?:gating|routing|gate|router|top.k|softmax.*gate)', t):
            s += 1.5
        if re.search(r'(?:expert.*select|which.*expert|rout.*token)', t):
            s += 1.0

        # 3. Load-balancing loss (0-2.5)
        if re.search(r'(?:load.balanc|auxiliary.*loss|aux.*loss|balance.*loss)', t):
            s += 1.5
        if re.search(r'(?:importance|loss.*formula|load.*equation|L_aux)', t):
            s += 1.0

        # 4. Training challenges (0-2)
        if re.search(r'(?:token.dropp|expert.collaps|instability|collapse|dropping)', t):
            s += 1.0
        if re.search(r'(?:training.*challeng|difficult|problem|issue|stability)', t):
            s += 1.0

        # 5. Inference implications (0-2)
        if re.search(r'(?:inference|memory.*bandwidth|expert.*parallel|sparse.*compute)', t):
            s += 1.0
        if re.search(r'(?:throughput|latency|batch.*size|parameter.*efficien)', t):
            s += 1.0

        # 6. Specific benchmarks / performance comparison (0-2)
        if re.search(r'(?:benchmark|mmlu|gsm8k|human-eval|mbpp|hellaswag|arc|truthful)', t):
            s += 1.0
        if re.search(r'(?:outperform|better.*than|compared to|vs\.|versus|advantage)', t):
            s += 1.0

        # 7. Paper references (0-2)
        if re.search(r'(?:paper|report|arxiv|technical.*report)', t):
            s += 1.0
        if re.search(r'(?:2023|2024|2025|et\s*al|vashwani|shazeer|fedus|lepikhin|du et al)', t):
            s += 1.0

        return round(s, 1)
