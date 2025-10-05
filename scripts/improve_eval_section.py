#!/usr/bin/env python3
"""Improve evaluation section with a compact table."""

import re

# Read the file
with open("my_reports/strategy_results.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace the evaluation section
old_eval = r"""### 📊 Evaluation Matrix

Please rate each model on a scale of 1-10 for each criterion:

### 1\. GPT-5 \(gpt-5\)
- \[ \] Technical Accuracy: Score 1-10
- \[ \] Completeness: Score 1-10
- \[ \] Nuance: Score 1-10
- \[ \] Actionability: Score 1-10
- \[ \] Overall Quality: Score 1-10

### 2\. Claude 4\.5 Sonnet \(claude-sonnet-4-5-20250929\)
- \[ \] Technical Accuracy: Score 1-10
- \[ \] Completeness: Score 1-10
- \[ \] Nuance: Score 1-10
- \[ \] Actionability: Score 1-10
- \[ \] Overall Quality: Score 1-10

### 3\. Gemini 2\.5 Pro \(vertex_ai/gemini-2\.5-pro\)
- \[ \] Technical Accuracy: Score 1-10
- \[ \] Completeness: Score 1-10
- \[ \] Nuance: Score 1-10
- \[ \] Actionability: Score 1-10
- \[ \] Overall Quality: Score 1-10

### 4\. Grok 4 \(xai/grok-4-latest\)
- \[ \] Technical Accuracy: Score 1-10
- \[ \] Completeness: Score 1-10
- \[ \] Nuance: Score 1-10
- \[ \] Actionability: Score 1-10
- \[ \] Overall Quality: Score 1-10

### 5\. Arbitrium Framework
- \[ \] Technical Accuracy: Score 1-10
- \[ \] Completeness: Score 1-10
- \[ \] Nuance: Score 1-10
- \[ \] Actionability: Score 1-10
- \[ \] Overall Quality: Score 1-10"""

new_eval = """### 📊 Evaluation Matrix

Rate each model on a scale of 1-10 for each criterion:

| Criterion | GPT-5 | Claude 4.5 | Gemini 2.5 | Grok 4 | Arbitrium |
|-----------|:-----:|:----------:|:----------:|:------:|:---------:|
| **Technical Accuracy** | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 |
| **Completeness** | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 |
| **Nuance** | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 |
| **Actionability** | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 |
| **Overall Quality** | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 | ⭐ __ / 10 |
| **TOTAL SCORE** | **__ / 50** | **__ / 50** | **__ / 50** | **__ / 50** | **__ / 50** |

---"""

content = re.sub(old_eval, new_eval, content, flags=re.DOTALL)

# Write back
with open("my_reports/strategy_results.md", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully improved evaluation section!")
