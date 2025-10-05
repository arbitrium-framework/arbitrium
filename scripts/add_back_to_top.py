#!/usr/bin/env python3
"""Add back-to-top links after each model response."""

import re

# Read the file
with open("my_reports/strategy_results.md", "r", encoding="utf-8") as f:
    content = f.read()

# Add back-to-top link before each closing </details> tag
back_to_top_link = "\n\n[⬆️ Back to Top](#-quick-comparison) | [📑 Table of Contents](#-table-of-contents)\n\n</details>"

content = re.sub(r"\n\n</details>", back_to_top_link, content)

# Write back
with open("my_reports/strategy_results.md", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Successfully added back-to-top links!")
