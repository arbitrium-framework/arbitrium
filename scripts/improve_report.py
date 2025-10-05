#!/usr/bin/env python3
"""
Script to improve strategy_results.md with collapsible sections.
"""

import re
from pathlib import Path


def wrap_responses_in_details(file_path: Path) -> None:
    """Wrap each model response in collapsible <details> tags."""

    content = file_path.read_text(encoding="utf-8")

    # Pattern to match each Approach section
    # Approach 1.1, 1.2, 1.3, 1.4
    single_model_pattern = r"(## Approach 1\.\d: Single Model\n\nModel: ([^\n]+)\n\nDuration: ([^\n]+)\n\nCost: ([^\n]+)\n\n### Response)"

    def replace_single_model(match):
        match.group(0)
        model = match.group(2)
        duration = match.group(3)
        cost = match.group(4)

        # Calculate minutes
        seconds = float(duration.split()[0])
        minutes = seconds / 60

        # Calculate cost/time ratio
        cost_val = float(cost.strip("$"))
        cost_time_ratio = cost_val / seconds

        return f"""## Approach 1.X: Single Model

**Model:** {model}
**Duration:** {duration} ({minutes:.1f} min)
**Cost:** {cost}
**Cost/Time Ratio:** ${cost_time_ratio:.5f}/s

<details>
<summary><b>📄 View Full {model.split('(')[0].strip()} Response</b> (Click to expand)</summary>

### Response"""

    # Replace single model sections
    content = re.sub(single_model_pattern, replace_single_model, content)

    # Add closing </details> before each "## Approach" (except the first one)
    # Split by ## Approach and add closing tags
    sections = re.split(r"(\n---\n\n## Approach)", content)

    result = []
    for i, section in enumerate(sections):
        if i > 0 and i % 2 == 0:  # After each approach section
            # Add closing tag before the separator
            result.append("\n</details>\n")
        result.append(section)

    # Write back
    file_path.write_text("".join(result), encoding="utf-8")
    print(f"✅ Updated {file_path}")


def improve_evaluation_section(file_path: Path) -> None:
    """Improve the evaluation section with a table format."""

    content = file_path.read_text(encoding="utf-8")

    # Find the Manual Evaluation Guide section
    eval_pattern = (
        r"## Manual Evaluation Guide\n\nPlease evaluate all responses on these dimensions:\n\n(### 1\. GPT-5.*?)(### Which Would You Actually Use\?)"
    )

    new_eval = """## Manual Evaluation Guide

Please evaluate all responses on these dimensions:

### 📊 Evaluation Matrix

| Criterion | GPT-5 | Claude 4.5 | Gemini 2.5 | Grok 4 | Arbitrium |
|-----------|-------|------------|------------|--------|-----------|
| **Technical Accuracy** (1-10) | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ |
| **Completeness** (1-10) | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ |
| **Nuance** (1-10) | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ |
| **Actionability** (1-10) | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ |
| **Overall Quality** (1-10) | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ | ⭐️ __ |
| **TOTAL** | __/50 | __/50 | __/50 | __/50 | __/50 |

### Which Would You Actually Use?

"""

    content = re.sub(eval_pattern, new_eval + r"\3", content, flags=re.DOTALL)

    file_path.write_text(content, encoding="utf-8")
    print(f"✅ Improved evaluation section in {file_path}")


def main():
    file_path = Path("my_reports/strategy_results.md")

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print("🚀 Improving strategy_results.md...")

    # Note: The ToC and comparison table are already added manually
    # Here we just add collapsible sections

    # wrap_responses_in_details(file_path)
    # improve_evaluation_section(file_path)

    print("✅ All improvements completed!")


if __name__ == "__main__":
    main()
