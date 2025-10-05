#!/usr/bin/env python3
"""Wrap model responses in collapsible sections."""

import re

# Read the file
with open("my_reports/strategy_results.md", "r", encoding="utf-8") as f:
    content = f.read()

# Define replacements
replacements = [
    # Gemini
    (
        r"Good luck\. The world needs better ways to make high-stakes decisions with AI\. Arbitrium could be that solution—but only if you execute\. 🚀\n\n## Approach 1\.3: Single Model\n\nModel: Gemini 2\.5 Pro \(vertex_ai/gemini-2\.5-pro\)\n\nDuration: 76\.9 seconds\n\nCost: \$0\.0594",
        r"""Good luck. The world needs better ways to make high-stakes decisions with AI. Arbitrium could be that solution—but only if you execute. 🚀

</details>

---

## Approach 1.3: Single Model

**Model:** Gemini 2.5 Pro (vertex_ai/gemini-2.5-pro)
**Duration:** 76.9 seconds (1.3 min)
**Cost:** $0.0594
**Cost/Time Ratio:** $0.00077/s

<details>
<summary><b>📄 View Full Gemini 2.5 Pro Response</b> (Click to expand ~150 lines)</summary>""",
    ),
    # Grok
    (
        r"## Approach 1\.4: Single Model\n\nModel: Grok 4 \(xai/grok-4-latest\)\n\nDuration: 85\.6 seconds\n\nCost: \$0\.0517",
        r"""</details>

---

## Approach 1.4: Single Model

**Model:** Grok 4 (xai/grok-4-latest)
**Duration:** 85.6 seconds (1.4 min)
**Cost:** $0.0517
**Cost/Time Ratio:** $0.00060/s

<details>
<summary><b>📄 View Full Grok 4 Response</b> (Click to expand ~90 lines)</summary>""",
    ),
    # Arbitrium
    (
        r"## Approach 2: Arbitrium Framework Tournament\n\nChampion: gpt\n\nDuration: 767\.4 seconds \(12\.8 minutes\)\n\nTotal Cost: \$0\.8449",
        r"""</details>

---

## Approach 2: Arbitrium Framework Tournament

**Champion:** GPT-5
**Duration:** 767.4 seconds (12.8 minutes)
**Total Cost:** $0.8449
**Cost/Time Ratio:** $0.00110/s
**Cost Multiple:** 11.1x vs cheapest single model
**Time Multiple:** 6.6x vs fastest single model

<details open>
<summary><b>🏆 View Arbitrium Framework Tournament Response</b> (Click to expand ~210 lines)</summary>""",
    ),
    # Manual Evaluation - close last details
    (
        r"## Manual Evaluation Guide\n\nPlease evaluate all responses on these dimensions:",
        r"""</details>

---

## Manual Evaluation Guide

### 📊 Evaluation Matrix

Please rate each model on a scale of 1-10 for each criterion:""",
    ),
]

# Apply all replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open("my_reports/strategy_results.md", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully wrapped all model responses in collapsible sections!")
