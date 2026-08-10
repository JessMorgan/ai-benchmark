"""MoE vs Dense architecture analysis benchmark task."""

import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import validate_sections


class MoEDensePlugin(BenchmarkTaskPlugin):
    @property
    def id(self):
        return "moe-dense"

    @property
    def version(self):
        return "0.5.0"

    @property
    def name(self):
        return "MoE vs Dense"

    @property
    def max_score(self):
        # 15 points of existing rubric + 2 points for the new sharper
        # "Quantitative trade-off" criterion.  If we kept this at 15.0,
        # Rubric.results() would cap the displayed score at min(earned, 15)
        # and silently redistribute points without expanding the score range.
        return 17.0

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
        rubric.record_validation(validate_sections(t, [
            "gating", "load-balancing", "training", "inference", "benchmark",
        ], min_chars=20))

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

        # Quantitative trade-off (0-2): sharpen this task so merely listing
        # concepts earns less than articulating a numeric comparison.  We
        # require (a) at least one specific numeric measurement tied to
        # compute or scale, and (b) an explicit side-by-side comparison
        # anchored in numerics.  Pattern B uses re.DOTALL so multi-line
        # comparisons like "70.6% MMLU \nvs 69.8% MMLU" still match; the
        # default eval_regex flag is just IGNORECASE which would miss
        # those.
        rubric.eval_regex(
            "Quantitative measurement (specifics)",
            1.0,
            t,
            [
                (
                    # Word-boundary anchored so "30%" in prose still matches,
                    # but "\\d+ teraflops / \\d+x faster" must have a real
                    # unit.  `mixture[\\s\\-_]of[\\s\\-_]experts?` covers
                    # hyphens, spaces, underscores between tokens.  The
                    # `more\\s+compute` arm allows trailing adjectives like
                    # "more compute intensive / required / overhead".
                    (r'\b\d+(?:\.\d+)?\s*(?:%|x\b|'
                    r'TFLOPs?|GFLOPs?|FLOPs?|BFLOPs|'
                    r'billion|trillion|million|billion\s+parameters?|trillion\s+parameters?|'
                    r'TOPS(?:/s)?\b|'
                    r'(?:B|M|K)\s+params?|(?:B|M|K)\s+parameters?)|'
                    r'\b\d+\s*[xX]\s*(?:faster|slower|larger|smaller|cheaper|'
                    r'more\s+compute(?:\s+(?:intensive|required|overhead|operations|heavy))?)|'
                    r'\b(?:total\s+params?|active\s+params?|inference\s+params?|'
                    r'trainable\s+params?|'
                    r'total\s+parameters?|active\s+parameters?|inference\s+parameters?|'
                    r'trainable\s+parameters?|total\s+parameter\s+count)|'
                    r'\b\d+(?:\.\d+)?\s*(?:tokens?\s*[/]?\s*s\b|tokens?\s+per\s+second\b)'),
                    1.0,
                ),
            ],
        )
        rubric.eval_regex(
            "Quantitative side-by-side comparison",
            1.0,
            t,
            [
                (
                    # Four flavors of side-by-side comparison, all anchored
                    # by a numeric token on at least one side so a generic
                    # "MoE is faster than dense" without numerics cannot earn
                    # credit.  Flavors 3 and 4 are expressed as single
                    # subgraphs that contain both the digit and the
                    # comparative - rather than as variable-width
                    # lookbehinds, which Python's stdlib `re` rejects (it
                    # only supports fixed-width lookbehinds).
                    (r'\b\d+(?:\.\d+)?%[\s\S]{0,200}?(?:vs\.?|versus|compared\s+to|over|higher\s+than|lower\s+than)[\s\S]{0,200}?\b\d+(?:\.\d+)?%|'
                    r'\b\d+(?:\.\d+)?\s*(?:B|M|K)\s*(?:params?|parameters?)[\s\S]{0,200}?(?:vs\.?|versus|compared\s+to)[\s\S]{0,200}?\b\d+(?:\.\d+)?\s*(?:B|M|K)\s*(?:params?|parameters?)|'
                    r'\b\d+(?:\.\d+)?\s*[xX][\s\S]{0,80}?\b(?:faster|slower|cheaper|more\s+expensive|larger|smaller)\s+than[\s\S]{0,80}?\b(?:dense|MoE|mixture[\s\-_]of[\s\-_]experts?|mixtral|llama|gemma|gpt-4)\b|'
                    r'\b(?:dense|MoE|mixture[\s\-_]of[\s\-_]experts?|mixtral|llama|gemma|gpt-4)[\s\S]{0,80}?\b(?:faster|slower|cheaper|more\s+expensive|larger|smaller|denser|sparser)\s+than[\s\S]{0,200}?\b\d+(?:\.\d+)?'),
                    1.0,
                ),
            ],
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not re.search(r"(?:equation|formula|=|loss\s*=)", t):
            rubric.penalize_criterion("Gating/routing mechanism", 0.5, "no mathematical formulation was provided")
            rubric.penalize_criterion("Load-balancing loss", 0.5, "no load-balancing equation was provided")
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
