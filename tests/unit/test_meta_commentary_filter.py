#!/usr/bin/env python3
"""Test the meta-commentary filtering functionality."""

import logging

from arbitrium.core.comparison import _strip_meta_commentary
from arbitrium.logging import get_contextual_logger, setup_logging

# Setup logger
setup_logging(level=logging.DEBUG, enable_file_logging=False)
logger = get_contextual_logger("test")

print("=" * 80)
print("TEST 1: Remove 'Sure, here is' prefix")
print("=" * 80)

text1 = """Sure, here is an improved answer:

This is the actual improved content that should be kept.
It has multiple lines and important information."""

cleaned1 = _strip_meta_commentary(text1, logger=logger)
expected1 = """This is the actual improved content that should be kept.
It has multiple lines and important information."""

print(f"Original: {text1[:50]!r}")
print(f"Cleaned: {cleaned1[:50]!r}")
print(f"Expected: {expected1[:50]!r}")
assert cleaned1 == expected1, f"FAILED: {cleaned1!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 2: Remove greeting prefix")
print("=" * 80)

text2 = """Hello! I am here to help you.

Here's my improved response:
The actual content starts here."""

cleaned2 = _strip_meta_commentary(text2, logger=logger)
expected2 = "The actual content starts here."

print(f"Original: {text2[:40]!r}")
print(f"Cleaned: {cleaned2!r}")
print(f"Expected: {expected2!r}")
assert cleaned2 == expected2, f"FAILED: {cleaned2!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 3: Remove 'Okay,' and 'here is' prefixes")
print("=" * 80)

text3 = """Okay, here is my refined answer:

The core answer content.
More details here."""

cleaned3 = _strip_meta_commentary(text3, logger=logger)
expected3 = """The core answer content.
More details here."""

print(f"Original: {text3[:40]!r}")
print(f"Cleaned: {cleaned3!r}")
print(f"Expected: {expected3!r}")
assert cleaned3 == expected3, f"FAILED: {cleaned3!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 4: Keep content without meta-commentary")
print("=" * 80)

text4 = """This is a direct answer without any meta-commentary.
It just provides the information requested."""

cleaned4 = _strip_meta_commentary(text4, logger=logger)
expected4 = text4

print(f"Original: {text4[:40]!r}")
print(f"Cleaned: {cleaned4[:40]!r}")
print(f"Expected: {expected4[:40]!r}")
assert cleaned4 == expected4, f"FAILED: {cleaned4!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 5: Remove 'Certainly' prefix")
print("=" * 80)

text5 = """Certainly, I'll provide an improved response:

The main content follows here."""

cleaned5 = _strip_meta_commentary(text5, logger=logger)
expected5 = "The main content follows here."

print(f"Original: {text5[:40]!r}")
print(f"Cleaned: {cleaned5!r}")
print(f"Expected: {expected5!r}")
assert cleaned5 == expected5, f"FAILED: {cleaned5!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 6: Remove multiple meta-commentary lines")
print("=" * 80)

text6 = """Sure!
Here's an improved version:
Let me provide you with a better answer:

The actual answer content."""

cleaned6 = _strip_meta_commentary(text6, logger=logger)
expected6 = "The actual answer content."

print(f"Original: {text6[:40]!r}")
print(f"Cleaned: {cleaned6!r}")
print(f"Expected: {expected6!r}")
assert cleaned6 == expected6, f"FAILED: {cleaned6!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 7: Case insensitive matching")
print("=" * 80)

text7 = """SURE, HERE IS MY IMPROVED ANSWER:

The actual content."""

cleaned7 = _strip_meta_commentary(text7, logger=logger)
expected7 = "The actual content."

print(f"Original: {text7[:40]!r}")
print(f"Cleaned: {cleaned7!r}")
print(f"Expected: {expected7!r}")
assert cleaned7 == expected7, f"FAILED: {cleaned7!r}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 8: Empty or whitespace-only input")
print("=" * 80)

text8 = "   \n\n   "
cleaned8 = _strip_meta_commentary(text8, logger=logger)
expected8 = text8

print(f"Original: {text8!r}")
print(f"Cleaned: {cleaned8!r}")
print(f"Expected: {expected8!r}")
assert cleaned8 == expected8, f"FAILED: {cleaned8!r}"
print("✅ PASSED\n")

print("=" * 80)
print("ALL TESTS PASSED! 🎉")
print("=" * 80)
print("\nSummary:")
print("- Meta-commentary prefixes are successfully removed")
print("- Clean content is preserved")
print("- Case-insensitive matching works")
print("- Edge cases (empty input) are handled")
