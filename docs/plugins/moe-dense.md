# MoE vs Dense Plugin

| Property | Value |
|---|---|
| ID | `moe-dense` |
| Name | MoE vs Dense |
| Version | `0.1.0` |
| Max Score | 15 |
| Streaming | No |

## Task

The model is asked to write a detailed technical analysis comparing Mixture-of-Experts (MoE) architecture with dense transformer architecture. The analysis must cover:

- Mathematical formulation of sparse MoE gating/routing (top-k routing, softmax gating)
- Auxiliary load-balancing loss formulation
- At least 2 training stability challenges unique to MoE
- Inference implications (memory bandwidth, expert parallelism)
- Specific benchmarks where MoE outperforms dense and vice versa
- At least 2 paper/technical report references

## Scoring Rubric

| Criterion | Max | Description |
|---|---|---|
| Both architectures covered | 2 | Explicitly discusses MoE and dense |
| Gating/routing mechanism | 2.5 | Top-k routing, softmax gating equations |
| Load-balancing loss | 2.5 | Auxiliary loss formulation |
| Training challenges | 2 | Token dropping, expert collapse, etc. |
| Inference implications | 2 | Memory bandwidth, expert parallelism |
| Specific benchmarks | 2 | MMLU, GSM8K, etc. with comparisons |
| Paper references | 2 | Specific papers, technical reports |

## Temperature

Default temperature can be set with:

```json
"moe-dense_temperature": 0.7
```

## Tips for Models

- Be precise with equations and notation.
- Cite specific papers (e.g., Shazeer et al., Fedus et al., Lepikhin et al.).
- Compare both strengths and weaknesses of each architecture.
