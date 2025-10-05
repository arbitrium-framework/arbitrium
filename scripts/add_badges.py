#!/usr/bin/env python3
"""Add visual badges to model sections."""

import re

# Read the file
with open("my_reports/strategy_results.md", "r", encoding="utf-8") as f:
    content = f.read()

# Define badge replacements
badges = [
    # Claude
    (
        r"(\*\*Cost/Time Ratio:\*\* \$0\.00066/s\n\n<details>\n<summary><b>📄 View Full Claude 4\.5 Sonnet Response</b>)",
        r"\1\n\n🏷️ `MOST-ACTIONABLE` `BEST-BALANCE` `30-DAY-PLAN`\n\n<details>\n<summary><b>📄 View Full Claude 4.5 Sonnet Response</b>",
    ),
    # Gemini
    (
        r"(\*\*Model:\*\* Gemini 2\.5 Pro.*?\n\*\*Duration:.*?\n\*\*Cost:.*?\n\*\*Cost/Time Ratio:.*?\n\n<details>)",
        r"\1\n\n🏷️ `FASTEST` `CONCISE` `EVIDENCE-FIRST`\n\n<details>",
    ),
    # Grok
    (
        r"(\*\*Model:\*\* Grok 4.*?\n\*\*Duration:.*?\n\*\*Cost:.*?\n\*\*Cost/Time Ratio:.*?\n\n<details>)",
        r"\1\n\n🏷️ `CHEAPEST` `MINIMAL` `BASIC-COVERAGE`\n\n<details>",
    ),
    # Arbitrium
    (
        r"(\*\*Time Multiple:\*\* 6\.6x vs fastest single model\n\n<details open>)",
        r"\1\n\n🏷️ `MULTI-PERSPECTIVE` `SYNTHESIZED` `TOURNAMENT-VALIDATED` `HIGH-STAKES`\n\n<details open>",
    ),
]

# Apply replacements
for pattern, replacement in badges:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open("my_reports/strategy_results.md", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully added badges to all model sections!")
