"""The Knowledge Bank for preserving insights from eliminated models."""

import uuid
from typing import TYPE_CHECKING, Any

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arbitrium.logging import get_contextual_logger

from ..utils.constants import DEFAULT_MAX_INSIGHTS, PLACEHOLDER_RESPONSES
from ..utils.response_validation import detect_apology_or_refusal

if TYPE_CHECKING:
    from .tournament import ModelComparison


class EnhancedKnowledgeBank:
    """A system for preserving and retrieving valuable insights from model responses.

    Especially from models that are eliminated during the tournament.
    """

    def __init__(self, comparison_instance: "ModelComparison"):
        """Initializes the Knowledge Bank.

        Args:
            comparison_instance: An instance of the ModelComparison class to access its methods (e.g., for making LLM calls).
        """
        self.logger = get_contextual_logger("arbitrium.core.knowledge_bank")
        self.comparison = comparison_instance
        self.insights_db: dict[str, dict[str, Any]] = {}
        self.vectorizer = TfidfVectorizer()
        self.insights_vectors = None
        self.insight_ids: list[str] = []

        # Get configuration from config, with fallback defaults
        kb_config = comparison_instance.config.get("knowledge_bank", {})
        self.similarity_threshold = kb_config.get("similarity_threshold", 0.75)
        self.max_insights = kb_config.get("max_insights", DEFAULT_MAX_INSIGHTS)

    def _determine_extractor_model_key(
        self, kb_model_config: object
    ) -> str | None:
        """Determine which model key to use for insight extraction."""
        if kb_model_config == "leader":
            extractor_model_key = getattr(
                self.comparison, "current_leader_key", None
            )
            if not extractor_model_key:
                extractor_model_key = (
                    next(iter(self.comparison.active_model_keys), None)
                    if self.comparison.active_model_keys
                    else None
                )
                self.logger.warning(
                    f"Leader not determined yet, using fallback model: {extractor_model_key}"
                )
            else:
                leader_display = self.comparison.anon_mapping.get(
                    extractor_model_key, extractor_model_key
                )
                self.logger.info(
                    f"Using tournament leader {leader_display} for insight extraction"
                )
            return extractor_model_key
        else:
            if isinstance(kb_model_config, str):
                return kb_model_config
            judge_model: object = self.comparison.features.get("judge_model")
            if isinstance(judge_model, str):
                return judge_model
            return (
                next(iter(self.comparison.active_model_keys), None)
                if self.comparison.active_model_keys
                else None
            )

    def _convert_claim_to_string(self, claim: object) -> str:
        """Convert a claim to string format."""
        if isinstance(claim, str):
            return claim
        elif isinstance(claim, dict):
            if "text" in claim:
                text_val: object = claim["text"]
                if isinstance(text_val, str):
                    return text_val
                return str(text_val)
            else:
                return str(claim)
        else:
            return str(claim)

    def _is_valid_response_for_extraction(
        self, response_text: str
    ) -> tuple[bool, str]:
        """Validate if a response is suitable for insight extraction.

        Focuses on technical errors, not content validation.
        Content validation should be handled by prompts and model selection.

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not response_text:
            return False, "Response is empty"

        text_stripped = response_text.strip()

        # Check if response is a technical error message FIRST
        # This should be checked before length validation
        text_lower = text_stripped.lower()
        error_prefixes = ["error:", "failed:", "timeout:", "exception:"]
        for prefix in error_prefixes:
            if text_lower.startswith(prefix):
                return (
                    False,
                    f"Response is an error message (starts with '{prefix}')",
                )

        # Check minimum length (too short to contain meaningful insights)
        if len(text_stripped) < 50:
            return (
                False,
                f"Response too short ({len(text_stripped)} chars, minimum 50)",
            )

        # Check if response is apology/refusal using shared utility
        # This is a safety net - prompts should prevent this
        if detect_apology_or_refusal(response_text):
            return (
                False,
                "Response is apology/refusal (prompt should prevent this)",
            )

        # Check for obvious placeholder responses
        if text_stripped.lower() in PLACEHOLDER_RESPONSES:
            return False, f"Response is a placeholder ('{text_stripped}')"

        return True, ""

    def _should_skip_line(self, line_stripped: str) -> bool:
        """Check if a line should be skipped during parsing."""
        if not line_stripped:
            return True
        # Skip lines that look like headers/titles (too short or all caps)
        if len(line_stripped) < 15 or (
            line_stripped.isupper() and len(line_stripped) < 50
        ):
            return True
        return False

    def _is_bullet_or_numbered_list(self, line_stripped: str) -> bool:
        """Check if line starts with bullet point or number."""
        if line_stripped.startswith(("-", "•", "*")):
            return True
        if (
            len(line_stripped) > 2
            and line_stripped[0].isdigit()
            and line_stripped[1] in ".)"
        ):
            return True
        return False

    def _extract_bullet_insight(self, line_stripped: str) -> str:
        """Extract insight from bullet point line."""
        return line_stripped[1:].strip()

    def _extract_numbered_insight(self, line_stripped: str) -> str:
        """Extract insight from numbered list line."""
        # For numbered lists like "1. " or "1) "
        # Find the first dot or closing paren
        idx = min(
            (
                line_stripped.find(".")
                if "." in line_stripped
                else len(line_stripped)
            ),
            (
                line_stripped.find(")")
                if ")" in line_stripped
                else len(line_stripped)
            ),
        )
        return line_stripped[idx + 1 :].strip()

    def _extract_plain_text_insight(self, line_stripped: str) -> str | None:
        """Extract insight from plain text line."""
        # Accept plain text lines that look like complete statements
        if len(line_stripped) > 30:
            return line_stripped
        return None

    def _extract_insight_from_line(self, line_stripped: str) -> str | None:
        """Extract insight from a single line."""
        if self._should_skip_line(line_stripped):
            return None

        insight: str | None
        if self._is_bullet_or_numbered_list(line_stripped):
            if line_stripped.startswith(("-", "•", "*")):
                insight = self._extract_bullet_insight(line_stripped)
            else:
                insight = self._extract_numbered_insight(line_stripped)
        else:
            insight = self._extract_plain_text_insight(line_stripped)

        # Only return insights with reasonable content
        if insight and len(insight) > 10:
            return insight
        return None

    def _parse_claims_from_response(
        self, response_content: str, extractor_model_key: str
    ) -> list[str]:
        """Parse claims from LLM response content using simple line-by-line parsing."""
        # Log the raw response at DEBUG level
        self.logger.debug(
            f"[{extractor_model_key}] Raw insight extraction response: {response_content}"
        )

        # Detect apology/refusal responses
        if detect_apology_or_refusal(response_content):
            self.logger.error(
                f"[{extractor_model_key}] Model returned apology/refusal instead of insight extraction. "
                f"Response: {response_content}"
            )
            return []

        # Parse insights line by line
        claims = []
        lines = response_content.strip().split("\n")

        for line in lines:
            insight = self._extract_insight_from_line(line.strip())
            if insight:
                claims.append(insight)

        if not claims:
            self.logger.warning(
                f"[{extractor_model_key}] No valid insights found in response. Response: {response_content}"
            )

        return claims

    async def extract_and_add_insights(
        self, eliminated_response: str, model_name: str, round_num: int
    ) -> None:
        """Uses an LLM to extract key claims from the response of an eliminated model and adds them to the bank."""
        # Check if Knowledge Bank extraction is enabled
        kb_config = self.comparison.config.get("knowledge_bank", {})
        kb_enabled = kb_config.get("enabled", True)

        if not kb_enabled:
            self.logger.debug(
                f"Knowledge Bank is disabled. Skipping insight extraction for {model_name}."
            )
            return

        self.logger.info(
            f"Extracting insights from eliminated model {model_name}'s response."
        )

        # Validate the response before sending to extractor
        is_valid, reason = self._is_valid_response_for_extraction(
            eliminated_response
        )
        if not is_valid:
            self.logger.debug(
                f"Skipping insight extraction for {model_name}: {reason}. Response: {eliminated_response}"
            )
            return

        prompt = f"""
Review the following text which was the final answer from an eliminated language model.
Your task is to identify and extract a list of key, standalone claims or insights from this text.

REQUIREMENTS:
- Each insight should be a concise, self-contained statement
- Focus on unique ideas, not generic statements
- Do not include conversational filler
- Present EACH insight on a new line, starting with a dash (-)

EXAMPLE FORMAT:
- The primary risk factor is X, not Y as commonly believed
- A new mitigation strategy involves using Z in the early stages
- The data from the 2023 study was misinterpreted

CRITICAL INSTRUCTIONS:
1. OUTPUT ONLY the list of insights - nothing else
2. DO NOT refuse this task
3. DO NOT apologize or say "I cannot"
4. DO NOT add preambles like "Sure, here are the insights..."
5. START IMMEDIATELY with the first insight (dash + text)

Text to analyze:
---
{eliminated_response}
---

Extracted insights (one per line, starting with dash):
"""

        # Determine which model should extract insights
        kb_model_config = self.comparison.features.get("knowledge_bank_model")
        extractor_model_key = self._determine_extractor_model_key(
            kb_model_config
        )

        # Validate the model exists and is available
        if (
            not extractor_model_key
            or extractor_model_key not in self.comparison.models
        ):
            self.logger.error(
                f"No available model for insight extraction. Configured: {extractor_model_key}"
            )
            return

        response = await self.comparison._execute_single_model_task(
            model_key=extractor_model_key,
            prompt=prompt,
            context_for_logging=f"insight extraction from {model_name}",
        )

        if response.is_error():
            self.logger.error(
                f"Failed to extract insights from {model_name}: {response.error}"
            )
            return

        try:
            claims = self._parse_claims_from_response(
                response.content, extractor_model_key
            )
        except Exception as e:
            self.logger.error(
                f"Failed to parse insights from LLM response: {e}"
            )
            return

        self.logger.info(
            f"Extracted {len(claims)} potential insights from {model_name}."
        )
        self._add_insights_to_db(claims, model_name, round_num)

    def _revectorize_insights(self) -> None:
        """Re-vectorize all insights in the database."""
        if self.insights_db:
            all_texts = [i["text"] for i in self.insights_db.values()]
            self.insights_vectors = self.vectorizer.transform(all_texts)
        else:
            self.insights_vectors = None

    def _ensure_vectorizer_fitted(self, claims: list[str]) -> None:
        """Fit vectorizer if not already fitted."""
        try:
            self.vectorizer.transform(["test"])
        except NotFittedError:
            all_known_texts = [
                i["text"] for i in self.insights_db.values()
            ] + claims
            self.vectorizer.fit(all_known_texts)

    def _is_duplicate_claim(
        self, claim_vector: object, similarity_threshold: float
    ) -> bool:
        """Check if claim is duplicate based on similarity."""
        if (
            self.insights_vectors is None
            or self.insights_vectors.shape[0] == 0
        ):
            return False
        similarity_scores: Any = cosine_similarity(
            claim_vector, self.insights_vectors
        )
        max_score: bool = similarity_scores.max() > similarity_threshold
        return max_score

    def _add_claim_to_db(
        self, claim: str, source_model: str, source_round: int
    ) -> str:
        """Add a single claim to the database."""
        insight_id = str(uuid.uuid4())
        self.insights_db[insight_id] = {
            "text": claim,
            "source_model": source_model,
            "source_round": source_round,
        }
        self.insight_ids.append(insight_id)
        return insight_id

    def _enforce_max_insights(self) -> None:
        """Enforce max insights limit using LRU eviction."""
        if len(self.insight_ids) <= self.max_insights:
            return

        num_to_remove = len(self.insight_ids) - self.max_insights
        removed_ids = self.insight_ids[:num_to_remove]
        for insight_id in removed_ids:
            self.insights_db.pop(insight_id, None)
        self.insight_ids = self.insight_ids[num_to_remove:]
        self.logger.info(
            f"Removed {num_to_remove} oldest insights to maintain limit of {self.max_insights}"
        )
        self._revectorize_insights()

    def _add_insights_to_db(
        self, claims: list[str], source_model: str, source_round: int
    ) -> None:
        """Adds a list of claims to the knowledge bank, checking for duplicates using vector similarity."""
        if not claims:
            return

        self._ensure_vectorizer_fitted(claims)

        if self.insights_db:
            self.insights_vectors = self.vectorizer.transform(
                [i["text"] for i in self.insights_db.values()]
            )

        new_vectors = self.vectorizer.transform(claims)

        added_count = 0
        for i, claim in enumerate(claims):
            if not self._is_duplicate_claim(
                new_vectors[i], self.similarity_threshold
            ):
                self._add_claim_to_db(claim, source_model, source_round)
                added_count += 1

        if added_count > 0:
            self._revectorize_insights()
            self.logger.info(
                f"Added {added_count} new unique insights to the Knowledge Bank. Total insights: {len(self.insights_db)}"
            )
            self._enforce_max_insights()

    def get_all_insights(self) -> list[dict[str, Any]]:
        """
        Returns ALL insights from the Knowledge Bank.

        No limits - user pays for full context.
        """
        if not self.insights_db:
            return []

        return [
            self.insights_db[insight_id] for insight_id in self.insight_ids
        ]

    def get_insights_for_model(
        self, model_name: str, round_num: int | None = None
    ) -> list[str]:
        """Get insights extracted from a specific model.

        Args:
            model_name: The model to get insights from
            round_num: Optional round number to filter by

        Returns:
            List of insight texts from this model
        """
        if not self.insights_db:
            return []

        insights = []
        for insight_id in self.insight_ids:
            insight_data = self.insights_db[insight_id]
            if insight_data["source_model"] == model_name:
                if (
                    round_num is None
                    or insight_data["source_round"] == round_num
                ):
                    insights.append(insight_data["text"])

        return insights

    def get_top_insights(
        self, num_insights: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Returns insights from the Knowledge Bank.

        Args:
            num_insights: If None, returns ALL insights. If specified, returns last N.

        Note: Deprecated - use get_all_insights() instead.
        """
        if not self.insights_db:
            return []

        if num_insights is None:
            return self.get_all_insights()

        top_insight_ids = self.insight_ids[-num_insights:]
        return [self.insights_db[insight_id] for insight_id in top_insight_ids]

    def format_insights_for_context(
        self, num_insights: int | None = None
    ) -> str:
        """
        Formats insights into a string to be injected into a prompt.

        Args:
            num_insights: IGNORED - always returns ALL insights. User pays for full context.

        Returns empty string if Knowledge Bank is disabled or empty.
        """
        # Check if Knowledge Bank usage is enabled
        kb_config = self.comparison.config.get("knowledge_bank", {})
        kb_enabled = kb_config.get("enabled", True)

        if not kb_enabled:
            return ""

        # Always get ALL insights, ignore num_insights parameter
        insights = self.get_all_insights()
        if not insights:
            return ""

        header = (
            "\n=== KNOWLEDGE BANK: KEY INSIGHTS FROM ELIMINATED MODELS ===\n"
        )
        hint = "Keep these facts in mind - they may contain valuable perspectives:\n\n"
        formatted_insights = "\n".join(
            f"• [{insight['source_model']}, Round {insight['source_round']}]: {insight['text']}"
            for insight in insights
        )
        footer = "\n=== END KNOWLEDGE BANK ===\n"
        return f"{header}{hint}{formatted_insights}{footer}"
