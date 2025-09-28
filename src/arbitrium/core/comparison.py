"Core comparison functionality for Arbitrium Framework."

import asyncio
import json
import random
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arbitrium.interfaces.event_handler import EventHandler
from arbitrium.interfaces.host import HostEnvironment
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import BaseModel, ModelResponse, run_with_retry

from ..utils.constants import SCORE_EXTRACTION_PATTERNS
from ..utils.markdown import sanitize_content_dict
from .knowledge_bank import EnhancedKnowledgeBank
from .prompt_templates import (
    LOG_EVALUATOR_RESPONSE,
    LOG_FEEDBACK,
    LOG_JUDGE_EVALUATION,
    LOG_PROMPT,
    LOG_RESPONSE,
    RESPONSE_WRAPPER,
)
from .prompting import PromptBuilder


def _indent_text(text: str, indent: str = "    ") -> str:
    """Add indentation to each line of text and remove empty lines for better log readability."""
    lines = [indent + line for line in text.splitlines() if line.strip()]
    return "\n" + "\n".join(lines)


def _strip_meta_commentary(text: str, logger: Any | None = None) -> str:
    """
    Remove meta-commentary from model responses.

    Models often add unwanted prefixes like:
    - "Sure, here is an improved answer:"
    - "Hello! I am here to help you..."
    - "Here's my improved response:"

    This function detects and strips these patterns while preserving the actual content.

    Args:
        text: The text to clean
        logger: Optional logger for debug messages

    Returns:
        Cleaned text with meta-commentary removed
    """
    if not text or not text.strip():
        return text

    original_text = text
    lines = text.split("\n")

    # Patterns that indicate meta-commentary (case-insensitive)
    meta_patterns = [
        # Common prefixes - including standalone acknowledgments
        r"^\s*(?:sure|okay|alright|certainly|absolutely)(?:[,!\s]+|$)",
        r"^\s*(?:here\s+is|here\'s)\s+",
        r"^\s*(?:improved|refined|enhanced|better)\s+(?:answer|response|version)",
        r"^\s*(?:let me|i will|i\'ll)\s+(?:provide|give|present)",
        r"^\s*(?:my\s+)?(?:improved|refined|enhanced)\s+(?:answer|response)",
        # Greetings
        r"^\s*(?:hello|hi|hey|greetings)!?\s*,?\s*",
        r"^\s*(?:i am|i\'m)\s+(?:here to )?help",
        # Meta phrases at the start
        r"^\s*(?:as requested|as you asked)",
        r"^\s*(?:below is|following is)",
    ]

    # Compile patterns
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in meta_patterns]

    # Track if we removed anything
    removed_lines = []

    # Process lines from the beginning, removing meta-commentary
    clean_lines = []
    started_content = False

    for line in lines:
        stripped_line = line.strip()

        # Skip empty lines at the beginning
        if not started_content and not stripped_line:
            continue

        # Check if this line is meta-commentary
        is_meta = False
        if not started_content and stripped_line:
            for pattern in compiled_patterns:
                if pattern.match(stripped_line):
                    is_meta = True
                    removed_lines.append(stripped_line)
                    break

        # If not meta-commentary, it's actual content
        if not is_meta:
            started_content = True
            clean_lines.append(line)

    # Join the clean lines
    cleaned_text = "\n".join(clean_lines).strip()

    # Log if we removed meta-commentary
    if removed_lines and logger:
        logger.debug(
            f"Stripped meta-commentary from response. Removed lines: {removed_lines[:3]}"
            + (f" (and {len(removed_lines) - 3} more)" if len(removed_lines) > 3 else "")
        )

    # Return original if we accidentally removed everything
    if not cleaned_text and original_text.strip():
        if logger:
            logger.warning("Meta-commentary filter would have removed all content, returning original")
        return original_text

    return cleaned_text


class ModelAnonymizer:
    """Centralizes model anonymization and response shuffling logic."""

    def __init__(self, event_handler: EventHandler, deterministic_mode: bool = False):
        """Initialize the anonymizer."""
        self.event_handler = event_handler
        self.logger = get_contextual_logger("arbitrium.comparison.anonymizer")
        self.rng = random.Random(42) if deterministic_mode else random.Random()

    @staticmethod
    def anonymize_model_keys(model_keys: list[str]) -> dict[str, str]:
        """Creates an anonymized mapping of model keys to LLM1, LLM2, etc."""
        return {key: f"LLM{i + 1}" for i, key in enumerate(model_keys)}

    def anonymize_responses(self, responses: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        """Creates anonymized labels for model responses (deterministic by alphabetical order)."""
        model_names = sorted(responses.keys())
        code_names = [f"LLM{i + 1}" for i in range(len(model_names))]
        forward_mapping = {name: code_names[i] for i, name in enumerate(model_names)}
        reverse_mapping = {v: k for k, v in forward_mapping.items()}
        anonymized_responses = {forward_mapping[name]: responses[name] for name in model_names}

        self.logger.debug("Anonymization mapping (deterministic by alphabetical order):")
        for real_name, anon_name in forward_mapping.items():
            self.logger.debug(f"  {real_name} -> {anon_name} (in prompt)")

        return anonymized_responses, reverse_mapping


class ScoreExtractor:
    """Extracts scores from evaluation text in JSON format."""

    def __init__(self, event_handler: EventHandler):
        """Initialize the score extractor."""
        self.event_handler = event_handler
        self.logger = get_contextual_logger("arbitrium.comparison.score_extractor")

    def extract_scores(self, evaluation_text: str, model_names: list[str]) -> dict[str, float]:
        """Extracts scores from evaluation text using ONLY pattern matching (no JSON)."""
        # Try pattern matching with exact model names
        scores = self._extract_scores_using_pattern_matching(evaluation_text, model_names)

        # If we didn't get all scores, try alternative names (LLM1, Response 1, etc.)
        if len(scores) < len(model_names):
            self.logger.debug(f"Pattern matching found {len(scores)}/{len(model_names)} scores, trying alternative names")
            alternative_scores = self._extract_scores_with_alternative_names(evaluation_text, model_names)
            scores.update(alternative_scores)

        if len(scores) >= len(model_names):
            self.logger.debug(f"Pattern matching successfully extracted all {len(scores)} scores")
        else:
            self.logger.warning(f"Pattern matching found only {len(scores)}/{len(model_names)} scores")

        return scores

    def _extract_numeric_score(self, score_value: object) -> float | None:
        """Extracts a numeric score from various formats using regex."""
        if isinstance(score_value, list):
            if len(score_value) > 0:
                score_value = score_value[0]
            else:
                return None
        if isinstance(score_value, (int, float)):
            return float(score_value)
        score_str = str(score_value)
        patterns = [
            r"(\d+\.?\d*)\s*/\s*10",
            r"(\d+\.?\d*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, score_str)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _match_model_name(self, key: str, model_names: list[str]) -> str | None:
        """Match a key to a model name (exact or fuzzy)."""
        if key in model_names:
            return key
        for model_name in model_names:
            if model_name in key or key in model_name:
                self.logger.debug(f"Fuzzy matched '{key}' to '{model_name}'")
                return model_name
        return None

    def _parse_scores_dict(
        self,
        scores_data: dict[str, object],
        model_names: list[str],
    ) -> dict[str, float]:
        """Parse scores from dict format."""
        scores = {}
        for key, score in scores_data.items():
            numeric_score = self._extract_numeric_score(score)
            if numeric_score is None:
                self.logger.warning(f"Invalid score value '{score}' for {key}")
                continue
            matched_model = self._match_model_name(key, model_names)
            if matched_model:
                scores[matched_model] = numeric_score
        return scores

    def _parse_scores_list(
        self,
        scores_data: list[object],
        model_names: list[str],
    ) -> dict[str, float]:
        """Parse scores from list format."""
        scores = {}
        for item in scores_data:
            if not isinstance(item, dict):
                continue
            model_key = item.get("model") or item.get("name") or item.get("model_name")
            score_value = item.get("score") or item.get("scores")
            if not model_key or score_value is None:
                continue
            numeric_score = self._extract_numeric_score(score_value)
            if numeric_score is None:
                continue
            matched_model = self._match_model_name(model_key, model_names)
            if matched_model:
                scores[matched_model] = numeric_score
        return scores

    def _parse_json_block(
        self,
        json_block: str,
        model_names: list[str],
    ) -> dict[str, float]:
        """Parses a JSON block and extracts scores."""
        scores: dict[str, float] = {}
        try:
            # Attempt to fix unescaped newlines within the 'reasoning' string
            # This is a common error from LLMs.
            # This regex captures the start of reasoning, the content, and the end quote.
            match = re.search(r'("reasoning":\s*")(.*)(")', json_block, re.DOTALL)
            if match:
                # The content of the reasoning string is group 2
                reasoning_content = match.group(2)
                # Escape newlines within the reasoning content
                fixed_reasoning = reasoning_content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                # Reconstruct the json_block
                json_block = json_block[: match.start(2)] + fixed_reasoning + json_block[match.end(2) :]
                self.logger.debug("Attempted to fix control characters in 'reasoning' field.")

            data = json.loads(json_block)
            if not isinstance(data, dict) or "scores" not in data:
                return scores
            scores_data = data["scores"]
            if isinstance(scores_data, dict):
                scores = self._parse_scores_dict(scores_data, model_names)
            elif isinstance(scores_data, list):
                scores = self._parse_scores_list(scores_data, model_names)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON block: {e}")
            self.logger.debug(f"Problematic JSON block:\n{json_block}")
        return scores

    def _extract_scores_from_json(
        self,
        evaluation_text: str,
        model_names: list[str],
    ) -> dict[str, float]:
        """Extracts scores from JSON blocks in the evaluation text."""
        scores = {}
        json_pattern = r"```json\s*\n(.*?)\n\s*```"
        json_blocks = [match.group(1).strip() for match in re.finditer(json_pattern, evaluation_text, re.DOTALL | re.IGNORECASE)]
        if not json_blocks:
            raw_json_pattern = r'\{[^{}]*"scores"[^{}]*[:\s]*[\[{].*?[\]}][^{}]*\}'
            json_blocks.extend([match.group(0).strip() for match in re.finditer(raw_json_pattern, evaluation_text, re.DOTALL)])
        for json_block in json_blocks:
            scores.update(self._parse_json_block(json_block, model_names))
        return scores

    def _extract_score_for_model(
        self,
        evaluation_text: str,
        model_name: str,
        patterns: list[str],
    ) -> float | None:
        """Extracts score for a single model using regex patterns."""
        for pattern in patterns:
            formatted_pattern = pattern.format(model_name=re.escape(model_name))
            match = re.search(
                formatted_pattern,
                evaluation_text,
                re.IGNORECASE | re.DOTALL | re.MULTILINE,
            )
            if match:
                try:
                    # Паттерны имеют формат: группа 1 = название модели, группа 2 = оценка
                    # Но некоторые старые паттерны могут иметь формат: группа 1 = numerator, группа 2 = denominator  # noqa: RUF003
                    # Проверяем, является ли группа 1 числом
                    if len(match.groups()) >= 2 and match.group(2):
                        try:
                            # Пытаемся интерпретировать как дробь (numerator/denominator)
                            numerator = float(match.group(1))
                            denominator = float(match.group(2))
                            score_value = (numerator / denominator) * 10.0
                            self.logger.debug(f"Extracted fractional score {numerator}/{denominator} = {score_value}/10 for {model_name}")
                            return score_value
                        except ValueError:
                            # Группа 1 не число (вероятно, название модели), берем группу 2 как оценку
                            score_value = float(match.group(2))
                            self.logger.debug(f"Extracted score {score_value} for {model_name} using pattern (group 2)")
                            return score_value
                        except ZeroDivisionError:
                            continue
                    score_value = float(match.group(1) if len(match.groups()) == 1 else match.group(2))
                    self.logger.debug(f"Extracted score {score_value} for {model_name} using pattern")
                    return score_value
                except (ValueError, TypeError, IndexError):
                    self.logger.warning(f"Invalid score value in pattern match for {model_name}")
                    continue
        return None

    def _extract_scores_using_pattern_matching(
        self,
        evaluation_text: str,
        model_names: list[str],
    ) -> dict[str, float]:
        """Extracts scores using regex patterns as a fallback."""
        scores = {}
        for model_name in model_names:
            score = self._extract_score_for_model(evaluation_text, model_name, SCORE_EXTRACTION_PATTERNS)
            if score is not None:
                scores[model_name] = score
        return scores

    def _extract_scores_with_alternative_names(
        self,
        evaluation_text: str,
        model_names: list[str],
    ) -> dict[str, float]:
        """Extracts scores by trying numbered model aliases (LLM1, Model 1, Response 1, etc.)."""
        scores = {}
        numbered_mapping = {f"LLM{idx + 1}": model_name for idx, model_name in enumerate(sorted(model_names))}
        response_mapping = {f"Response {idx + 1}": model_name for idx, model_name in enumerate(sorted(model_names))}
        numbered_mapping.update(response_mapping)
        alternative_names = list(numbered_mapping.keys())
        extracted = self._extract_scores_using_pattern_matching(evaluation_text, alternative_names)
        for alt_name, score in extracted.items():
            if alt_name in numbered_mapping:
                original_name = numbered_mapping[alt_name]
                if original_name not in scores:
                    scores[original_name] = score
                    self.logger.debug(f"Mapped alternative name '{alt_name}' to '{original_name}' with score {score}")
        return scores

    def _detect_apology_or_refusal(self, evaluation_text: str) -> bool:
        """Detect if the response is an apology or refusal instead of proper evaluation."""
        if not evaluation_text:
            return False

        # Convert to lowercase for case-insensitive matching
        text_lower = evaluation_text.lower().strip()

        # Check if the response starts with common apology/refusal patterns
        refusal_patterns = [
            "i cannot",
            "i can't",
            "i'm sorry",
            "i am sorry",
            "i apologize",
            "sorry, i",
            "sorry but",
            "i'm unable",
            "i am unable",
            "i don't have",
            "i do not have",
            "as an ai",
            "i'm an ai",
            "i am an ai",
        ]

        # Check if response starts with refusal (first 200 chars)
        text_start = text_lower[:200]
        return any(pattern in text_start for pattern in refusal_patterns)

    def extract_scores_from_evaluation(
        self,
        evaluation_text: str,
        model_names: list[str],
        evaluator_name: str = "Unknown",
    ) -> dict[str, float]:
        """Extracts numerical scores from an evaluation text."""
        # Log the raw response at DEBUG level
        self.logger.debug(f"[{evaluator_name}] Raw evaluation response to parse: {evaluation_text}")

        # Detect apology/refusal responses
        if self._detect_apology_or_refusal(evaluation_text):
            self.logger.error(f"[{evaluator_name}] Model returned apology/refusal instead of evaluation. Response: {evaluation_text}")
            return {}

        scores = self.extract_scores(evaluation_text, model_names)
        missing_models = set(model_names) - set(scores.keys())
        if missing_models:
            self.logger.warning(
                f"[{evaluator_name}] Could not extract scores for {len(missing_models)} models: {', '.join(sorted(missing_models))}. "
                f"The evaluator may have failed to score all models. "
                f"Response: {evaluation_text}"
            )
            # Return empty dict when evaluation is incomplete to avoid unfair penalties
            return {}
        return scores


class ReportGenerator:
    """Generates and saves reports."""

    def __init__(self, host: HostEnvironment, event_handler: EventHandler):
        """Initialize the report generator."""
        self.host = host
        self.event_handler = event_handler
        self.logger = get_contextual_logger("arbitrium.comparison.report_generator")

    async def save_report_to_file(
        self,
        content_type: str,
        round_number: int | None = None,
        content: dict[str, Any] | None = None,
    ) -> bool:
        """Helper method to save reports to files."""
        if not content:
            self.logger.warning(f"No content provided to save for {content_type}")
            return False

        prefix = f"round{round_number}_{content_type}" if round_number is not None else content_type
        filename = f"{prefix}.md"  # Simplified filename generation
        sanitized_content = sanitize_content_dict(content, preserve_markdown=True)

        clean_title = content_type.replace("_", " ").replace("champion solution", "Champion Solution")
        if round_number:
            clean_title += f" Round {round_number}"

        report_title = f"# {clean_title}"
        report_sections = []
        for key, value in sanitized_content.items():
            clean_key = key.replace("_", " ").title()
            # Note: tournament_history is already formatted by sanitize_content_dict
            if key == "champion_solution":
                from ..utils.markdown import adjust_markdown_headers

                adjusted_solution = adjust_markdown_headers(value, start_level=3)
                report_sections.append(f"## {clean_key}\n\n{adjusted_solution}")
            else:
                report_sections.append(f"## {clean_key}\n\n{value}")

        report_body = "\n\n".join(report_sections)
        file_content = f"{report_title}\n\n{report_body}"

        try:
            await self.host.write_file(f"reports/{filename}", file_content)
            self.logger.info(f"Saved {content_type} to reports/{filename}")
            return True
        except Exception as e:
            self.logger.error(f"Unexpected error saving {content_type}: {e!s}")
            return False


class TournamentRunner:
    """Orchestrates the phases of the tournament."""

    def __init__(self, comparison_instance: "ModelComparison") -> None:
        """Initialize with reference to ModelComparison instance."""
        self.comp = comparison_instance
        self.event_handler = comparison_instance.event_handler
        self.logger = comparison_instance.logger

    async def _run_tournament_phases(self, initial_question: str) -> str:
        """Runs the main phases of the tournament."""
        if not await self._run_initial_phase(initial_question):
            return "No valid initial responses. Tournament cannot proceed."
        if not await self._run_phase_2(initial_question):
            return "Phase 2 failed. Tournament cannot proceed."
        await self._run_elimination_rounds(initial_question)
        return await self._finalize_tournament(initial_question)

    async def run(self, initial_question: str) -> str:
        """Runs the complete model comparison tournament."""
        # Set run_id for the entire tournament
        self.logger.set_run()
        self.logger.info("Starting model comparison tournament", question=initial_question)

        self.logger.info(f"Starting model comparison tournament: {initial_question}")
        self.comp.previous_answers = []
        self.comp.eliminated_models = []
        self.comp.evaluation_history = []
        self.comp.feedback_history = []
        self.comp.criticism_history = []

        try:
            return await self._run_tournament_phases(initial_question)
        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user.")
            return "Process interrupted by user."
        except Exception as e:
            self.logger.error(f"Unexpected error in tournament: {e!s}", exc_info=True)
            return f"Tournament error: {e!s}"

    async def _run_initial_phase(self, initial_question: str) -> bool:
        """Runs the initial response phase."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 PHASE 1: Initial Answers - Each model answers independently")
        self.logger.info("=" * 80)
        initial_responses = await self.comp.run_initial_round(initial_question)
        if not initial_responses:
            return False
        self.comp.previous_answers.append(initial_responses)
        self.logger.info(f"✅ PHASE 1 COMPLETE: Got {len(initial_responses)} initial responses")
        return True

    async def _run_phase_2(self, initial_question: str) -> bool:
        """Runs Phase 2: Improvement phase using unified workflow."""
        improvement_phase_config = self.comp.config.get("improvement_phase", {})

        if not improvement_phase_config.get("enabled", True):
            self.logger.info("Phase 2 is disabled. Skipping.")
            return True

        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔄 PHASE 2: Improvement Phase")
        self.logger.info("=" * 80)

        feedback_context: dict[str, dict[str, str]] | None = None
        if improvement_phase_config.get("feedback_enabled", False):
            self.logger.info("📝 Collecting feedback from models...")
            feedback_context = await self.comp.run_feedback(
                initial_question,
                self.comp.previous_answers[0],
                feedback_instruction=improvement_phase_config.get("feedback_instruction", "Provide feedback for this answer."),
            )
            if not feedback_context:
                self.logger.warning("No feedback collected, proceeding without it")
                feedback_context = None

        self.logger.info("💡 Generating improved responses...")
        improved_responses = await self.comp.run_improvement(
            initial_question,
            self.comp.previous_answers[0],
            improvement_instruction=improvement_phase_config.get("improvement_instruction", "Improve your answer."),
            improvement_context=feedback_context,
            other_responses=(self.comp.previous_answers[0] if improvement_phase_config.get("share_responses", True) else None),
        )

        if not improved_responses:
            return False

        self.comp.previous_answers.append(improved_responses)
        self.logger.info(f"✅ PHASE 2 COMPLETE: Got {len(improved_responses)} improved responses")
        return True

    async def _run_elimination_rounds(self, initial_question: str) -> None:
        """Runs the elimination tournament loop."""
        round_num = 1
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"📊 Starting elimination rounds with {len(self.comp.active_model_keys)} models")
        self.logger.info("=" * 80)

        while len(self.comp.active_model_keys) > 1:
            self.logger.info("\n" + "-" * 80)
            self.logger.info(f"🔍 ROUND {round_num}: Cross-Evaluation Phase")
            self.logger.info("-" * 80)

            evaluations = await self.comp.run_cross_evaluation(initial_question, self.comp.previous_answers[-1], round_num)
            if not evaluations:
                self.logger.error(f"No evaluations in round {round_num}. Stopping elimination.")
                break

            self.comp.evaluation_history.append(
                {
                    "round": round_num,
                    "evaluations": evaluations.copy(),
                    "scores": (self.comp.evaluation_scores.copy() if hasattr(self.comp, "evaluation_scores") else {}),
                }
            )

            (
                eliminated_model,
                _leader_model,
            ) = await self.comp.determine_lowest_and_highest_ranked_models()
            if not eliminated_model:
                self.logger.warning("Could not determine model to eliminate. Ending tournament.")
                break

            await self._handle_elimination(eliminated_model, round_num)

            if len(self.comp.active_model_keys) <= 1:
                break

            if not await self._run_refinement_strategy(initial_question, round_num):
                break

            round_num += 1

    async def _run_refinement_strategy(
        self,
        initial_question: str,
        round_num: int,
    ) -> bool:
        """Runs refinement round using unified workflow."""
        refinement_phase_config = self.comp.config.get("refinement_phase", {})

        if not refinement_phase_config.get("enabled", True):
            self.logger.info(f"Refinement is disabled. Skipping round {round_num} refinement.")
            return True

        self.logger.info(f"\n🔄 ROUND {round_num}: Refinement Phase")

        feedback_context: dict[str, dict[str, str]] | None = None
        if refinement_phase_config.get("feedback_enabled", False):
            self.logger.info("📝 Collecting feedback from models...")
            feedback_context = await self.comp.run_feedback(
                initial_question,
                self.comp.previous_answers[-1],
                feedback_instruction=refinement_phase_config.get("feedback_instruction", "Provide feedback for this answer."),
            )

        self.logger.info("💡 Generating refined responses...")
        refined_responses = await self.comp.run_improvement(
            initial_question,
            self.comp.previous_answers[-1],
            improvement_instruction=refinement_phase_config.get("improvement_instruction", "Refine your answer."),
            improvement_context=feedback_context,
            other_responses=(self.comp.previous_answers[-1] if refinement_phase_config.get("share_responses", True) else None),
        )

        if not refined_responses:
            return False

        self.comp.previous_answers.append(refined_responses)
        self.logger.info(f"✅ ROUND {round_num} COMPLETE: Got {len(refined_responses)} refined responses")
        return True

    async def _handle_elimination(self, eliminated_model: str, round_num: int) -> None:
        """Handles the logic for eliminating a model."""
        eliminated_response = self.comp.previous_answers[-1].get(eliminated_model)
        if eliminated_response:
            await self.comp.knowledge_bank.extract_and_add_insights(eliminated_response, eliminated_model, round_num)

        # Store elimination info with reason and score for provenance
        elimination_info: dict[str, object] = {
            "model": eliminated_model,
            "round": round_num,
            "reason": getattr(self.comp, "elimination_reason", "Lowest score in evaluation"),
            "score": getattr(self.comp, "elimination_score", None),
        }
        self.comp.eliminated_models.append(elimination_info)  # type: ignore[arg-type]

        model_key_to_remove = next(
            (key for key, name in self.comp.anon_mapping.items() if name == eliminated_model),
            None,
        )

        if model_key_to_remove:
            self.comp.active_model_keys.remove(model_key_to_remove)
            self.comp.anon_mapping.pop(model_key_to_remove)

        self.logger.info(f"❌ ELIMINATED: {eliminated_model} - {len(self.comp.active_model_keys)} models remaining")
        self.logger.info(f"   Reason: {elimination_info['reason']}")

    async def _finalize_tournament(self, initial_question: str) -> str:
        """Returns the final champion's answer or a message if no champion is found."""
        if len(self.comp.active_model_keys) == 0:
            self.logger.error("All models failed during tournament. No champion can be determined.")
            return "Tournament ended prematurely: All models failed or were eliminated due to errors."
        elif len(self.comp.active_model_keys) == 1:
            final_model_key = self.comp.active_model_keys[0]
            final_model_anon = self.comp.anon_mapping[final_model_key]

            champion_answer = self.comp.previous_answers[-1].get(final_model_anon, "")

            if not champion_answer:
                self.logger.error(f"Could not find final answer for champion {final_model_anon}")
                return f"Champion {final_model_anon} determined but answer not found."

            self.logger.info(f"🏆 CHAMPION: {final_model_anon} - Using their final refined answer")

            if self.comp.features.get("save_reports_to_disk", True):
                await self.comp._save_champion_report(
                    initial_question=initial_question,
                    final_model_anon=final_model_anon,
                    champion_answer=champion_answer,
                    all_previous_answers=self.comp.previous_answers,
                )

            self.logger.info("Champion's Final Answer", extra={"display_type": "section_header"})
            self.logger.info(champion_answer, extra={"display_type": "model_response", "model_name": "success"})

            return champion_answer
        else:
            msg = f"Tournament ended with {len(self.comp.active_model_keys)} models remaining. No single champion determined."
            self.logger.warning(f"Tournament ended with {len(self.comp.active_model_keys)} models remaining (no single champion).")
            return msg


@dataclass
class InitialCosts:
    total_cost: float = 0.0
    cost_by_model: dict[str, float] = field(default_factory=dict)


class ModelComparison:
    """Manages the comparison of different LLM models."""

    def __init__(
        self,
        config: dict[str, Any],
        models: dict[str, BaseModel],
        event_handler: EventHandler,
        host: HostEnvironment,
    ):
        """Initialize the model comparison."""
        self.config = config
        self.models = models
        self.event_handler = event_handler
        self.host = host

        # Initialize contextual logger for correlation IDs
        self.logger = get_contextual_logger("arbitrium.comparison")

        self.total_cost = 0.0
        self.cost_by_model: dict[str, float] = {}

        self.retry_settings = config["retry"]
        self.features = config["features"]
        self.prompts = config["prompts"]

        self.previous_answers: list[dict[str, str]] = []
        self.eliminated_models: list[str] = []
        self.evaluation_history: list[dict[str, Any]] = []
        self.evaluation_scores: dict[str, float] | dict[str, dict[str, float]] = {}
        self.all_evaluations: dict[str, str] = {}
        self.feedback_history: list[dict[str, Any]] = []
        self.criticism_history: list[dict[str, Any]] = []

        deterministic_mode = self.features.get("deterministic_mode", False)
        if deterministic_mode:
            self.logger.info("Running in deterministic mode with fixed random seed")

        self.anonymizer = ModelAnonymizer(self.event_handler, deterministic_mode)
        self.score_extractor = ScoreExtractor(self.event_handler)
        self.report_generator = ReportGenerator(self.host, self.event_handler)
        self.prompt_builder = PromptBuilder(self.prompts)

        self.active_model_keys = list(models.keys())
        self.judge_model_key = self._identify_and_remove_judge()

        self.anon_mapping = self.anonymizer.anonymize_model_keys(self.active_model_keys)

        self.knowledge_bank = EnhancedKnowledgeBank(self)

        max_concurrent = config.get("model_defaults", {}).get("concurrency_limits", {}).get("max_concurrent_requests", 2)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.runner = TournamentRunner(self)

    def _identify_and_remove_judge(self) -> str | None:
        """Identifies the judge model from config and removes it from active participants."""
        judge_model_config_key = self.features.get("judge_model")
        if not judge_model_config_key:
            return None

        judge_model_key = None
        for key, model_instance in self.models.items():
            if key == judge_model_config_key or model_instance.display_name == judge_model_config_key:
                judge_model_key = key
                break

        if not judge_model_key:
            self.logger.warning(f"Judge model '{judge_model_config_key}' not found in available models")
            return None

        if judge_model_key in self.active_model_keys:
            self.logger.info(
                f"⚠️  Judge model '{self.models[judge_model_key].display_name}' will only act as judge and will not participate in the tournament.",
                extra={"display_type": "colored_text", "color": "warning"},
            )
            self.active_model_keys.remove(judge_model_key)
            self.logger.info(f"Removed judge model {judge_model_key} from tournament participants")

        return judge_model_key

    def _get_knowledge_bank_context(self) -> str:
        """Get formatted Knowledge Bank insights if enabled, empty string otherwise."""
        kb_config = self.config.get("knowledge_bank", {})
        max_insights = kb_config.get("max_insights_to_inject", 5)
        return self.knowledge_bank.format_insights_for_context(num_insights=max_insights)

    def _filter_valid_responses(
        self,
        results: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        """Drop empty/obviously error responses and return failing model keys."""
        valid = {}
        failed_keys = []
        for key, value in results.items():
            if not value:
                failed_keys.append(key)
                continue
            txt = value.strip()
            if not txt or txt.lower().startswith("error:"):
                failed_keys.append(key)
                continue

            if len(txt) < 10 or txt.strip() in ["###", "...", "N/A"]:
                self.logger.warning(f"Filtered out placeholder/invalid response from {key}: '{txt}'")
                failed_keys.append(key)
                continue

            valid[key] = txt
        return valid, failed_keys

    def _decode_shuffled_names(
        self,
        text: str,
        reverse_shuffle_mapping: dict[str, str],
    ) -> str:
        """Replace shuffled code names with original anonymous names in text."""
        decoded_text = text
        for code_name, orig_name in reverse_shuffle_mapping.items():
            pattern = r"\b" + re.escape(code_name) + r"\b"
            decoded_text = re.sub(pattern, f"({orig_name})", decoded_text)
        return decoded_text

    def _display_model_score(
        self,
        model_name: str,
        score: float,
        score_type: str = "Score",
    ) -> None:
        """Display a model's score in consistent format."""
        self.logger.info(f"{model_name}: {score_type} {score:.2f}/10", extra={"display_type": "colored_text", "color": model_name})

    def _handle_model_failure(self, model_key: str, reason: str) -> None:
        """Centralized logic to remove a failed model from the tournament."""
        display_name = self.anon_mapping.get(model_key, model_key)
        self.logger.error(f"❌ Removing {display_name} from tournament: {reason}")

        if model_key in self.active_model_keys:
            self.active_model_keys.remove(model_key)
        self.anon_mapping.pop(model_key, None)

    async def _execute_single_model_task(
        self,
        model_key: str,
        prompt: str,
        context_for_logging: str,
    ) -> ModelResponse:
        """Executes a single task for a model with retry logic and timeout."""
        from ..utils.constants import DEFAULT_MODEL_TIMEOUT

        model = self.models[model_key]
        self.anon_mapping.get(model_key, model.display_name)

        # Use task context for correlation IDs
        with self.logger.task_context(phase=context_for_logging, model=model_key):
            self.logger.debug(f"Preparing to execute task for model: {model_key}")

            self.logger.debug(f"Executing {context_for_logging}", model=model.display_name, model_id=model.model_name)

            # Log the full prompt at DEBUG level
            log_message = LOG_PROMPT.format(model=model.display_name, content=prompt)
            self.logger.debug(_indent_text(log_message))

            try:
                async with self.semaphore:
                    response = await asyncio.wait_for(
                        run_with_retry(
                            model=model,
                            prompt=prompt,
                            max_attempts=self.retry_settings.get("max_attempts", 3),
                            initial_delay=None,
                            max_delay=None,
                            logger=self.event_handler,
                        ),
                        timeout=DEFAULT_MODEL_TIMEOUT * 2,
                    )
            except asyncio.TimeoutError:
                self.logger.error(f"Task timeout for {model.full_display_name} after {DEFAULT_MODEL_TIMEOUT * 2}s")
                return ModelResponse.create_error(f"Task timeout after {DEFAULT_MODEL_TIMEOUT * 2} seconds")

            if hasattr(response, "cost"):
                if response.cost > 0:
                    self.total_cost += response.cost
                    model_display_name = model.display_name
                    if model_display_name not in self.cost_by_model:
                        self.cost_by_model[model_display_name] = 0.0
                    self.cost_by_model[model_display_name] += response.cost
                    self.logger.info(f"💰 Added ${response.cost:.4f} for {model_display_name}, total now: ${self.total_cost:.4f}")
                else:
                    self.logger.debug(f"💰 Zero cost response from {model.display_name}")
            else:
                self.logger.debug(f"💰 No cost attribute in response from {model.display_name}")

            # Apply meta-commentary filtering for improvement responses
            if response and response.content and context_for_logging == "IMPROVEMENT":
                cleaned_content = _strip_meta_commentary(response.content, logger=self.logger)
                # Update the response object with cleaned content
                response.content = cleaned_content

            # Log response immediately to ensure it's saved even if subsequent tasks fail
            if response and response.content and context_for_logging:
                display_name = self.anon_mapping.get(model_key, model.display_name)
                log_message = LOG_RESPONSE.format(
                    response_type=context_for_logging.upper(),
                    model=display_name,
                    content=response.content,
                )
                self.logger.info(_indent_text(log_message))

            return response

    def _prepare_parallel_tasks(
        self,
        model_keys_to_run: list[str],
        prompt_builder: Callable[[str, BaseModel], str],
        context_for_logging: str,
    ) -> tuple[list[tuple[str, Any]], dict[str, str]]:
        """Prepare tasks for parallel execution."""
        tasks = []
        display_names = {}

        for model_key in model_keys_to_run:
            if model_key not in self.active_model_keys:
                continue

            model = self.models[model_key]
            display_name = self.anon_mapping[model_key]
            display_names[model_key] = display_name

            prompt = prompt_builder(model_key, model)
            task = self._execute_single_model_task(
                model_key=model_key,
                prompt=prompt,
                context_for_logging=context_for_logging,
            )
            tasks.append((model_key, task))

        return tasks, display_names

    def _process_single_response(
        self,
        response: Any,
        model_key: str,
        display_name: str,
        context_for_logging: str,
        results: dict[str, str],
    ) -> None:
        """Process a single response from a model."""
        if isinstance(response, Exception):
            self._handle_model_failure(model_key, f"Error in {context_for_logging}: {response!s}")
            return

        if not isinstance(response, ModelResponse):
            return

        if response.is_error():
            error_msg = response.error or "Unknown error"
            self._handle_model_failure(model_key, f"Error in {context_for_logging}: {error_msg}")
            return

        # Content is already cleaned by _execute_single_model_task if needed
        results[display_name] = response.content

    def _handle_failed_models(self, failed_keys: list[str]) -> None:
        """Remove models that produced invalid responses."""
        for display_name in failed_keys:
            model_key_to_remove = next((k for k, v in self.anon_mapping.items() if v == display_name), None)
            if model_key_to_remove:
                self._handle_model_failure(model_key_to_remove, "Invalid/empty response")

    def _check_tournament_viability(
        self,
        valid_results: dict[str, str],
        context_for_logging: str,
    ) -> bool:
        """Check if tournament can continue with current results."""
        if not valid_results:
            self.logger.critical(f"No valid responses for {context_for_logging} from any models.")
            self.logger.warning(f"Tournament cannot continue. Active models remaining: {len(self.active_model_keys)}")
            return False

        if len(valid_results) < 2 and context_for_logging in [
            "INITIAL",
            "IMPROVEMENT",
        ]:
            self.logger.warning(f"Only {len(valid_results)} model(s) responded. Tournament may end prematurely.")

        return True

    async def _execute_parallel_model_tasks(
        self,
        model_keys_to_run: list[str],
        prompt_builder: Callable[[str, BaseModel], str],
        context_for_logging: str,
    ) -> dict[str, str]:
        """Executes tasks in parallel for multiple models and collects their responses."""
        tasks, display_names = self._prepare_parallel_tasks(model_keys_to_run, prompt_builder, context_for_logging)
        results: dict[str, str] = {}

        try:
            responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            for i, response in enumerate(responses):
                model_key = tasks[i][0]
                display_name = display_names[model_key]
                self._process_single_response(response, model_key, display_name, context_for_logging, results)
        except Exception as e:
            self.logger.error(f"Unexpected error during parallel {context_for_logging} calls: {e!s}")

        valid_results, failed_keys = self._filter_valid_responses(results)
        self._handle_failed_models(failed_keys)

        if not self._check_tournament_viability(valid_results, context_for_logging):
            return {}

        return valid_results

    async def run_initial_round(self, initial_question: str) -> dict[str, str]:
        """Run the Individual Response Generation phase."""
        self.logger.info("Individual Response Generation", extra={"display_type": "section_header"})

        def build_initial_prompt(model_key: str, model: BaseModel) -> str:
            return self.prompt_builder.build_initial_prompt(initial_question)

        valid_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_initial_prompt,
            context_for_logging="INITIAL",
        )

        if not valid_responses:
            self.logger.error("No valid responses from any models.")
            return {}

        self.logger.info("Initial Model Responses", extra={"display_type": "section_header"})

        return valid_responses

    async def run_feedback(
        self,
        initial_question: str,
        current_responses: dict[str, str],
        feedback_instruction: str,
    ) -> dict[str, dict[str, str]]:
        """Unified feedback phase - models provide feedback for each other's answers."""
        active_responses = {name: resp for name, resp in current_responses.items() if name in [self.anon_mapping[k] for k in self.active_model_keys]}

        feedback_context: dict[str, dict[str, str]] = {model_name: {} for model_name in active_responses.keys()}

        tasks = []
        task_metadata = []

        for target_model, target_answer in active_responses.items():
            reviewer_models = [m for m in self.active_model_keys if self.anon_mapping[m] != target_model]

            for reviewer_key in reviewer_models:
                prompt = self.prompt_builder.build_feedback_prompt(
                    initial_question=initial_question,
                    target_answer=target_answer,
                    feedback_instruction=feedback_instruction,
                )
                task = self._execute_single_model_task(
                    model_key=reviewer_key,
                    prompt=prompt,
                    context_for_logging="FEEDBACK",
                )
                tasks.append(task)
                task_metadata.append((reviewer_key, target_model))

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                reviewer_key, target_model = task_metadata[i]
                reviewer_name = self.anon_mapping[reviewer_key]

                if isinstance(response, Exception):
                    self.logger.error(f"Error getting feedback from {reviewer_name} for {target_model}: {response!s}")
                    continue

                if not isinstance(response, ModelResponse):
                    continue

                if response.is_error():
                    self.logger.error(f"Error getting feedback from {reviewer_name} for {target_model}: {response.error}")
                    continue

                feedback_text = response.content
                feedback_context[target_model][reviewer_name] = feedback_text

                log_message = LOG_FEEDBACK.format(reviewer=reviewer_name, target=target_model, content=feedback_text)
                self.logger.debug(_indent_text(log_message))
                self.logger.info(
                    f"\n{reviewer_name}'s feedback for {target_model}:",
                    extra={"display_type": "colored_text", "color": reviewer_name},
                )
                self.logger.info(feedback_text, extra={"display_type": "colored_text", "color": reviewer_name})
                self.logger.info("-" * 20, extra={"display_type": "colored_text"})

        self.feedback_history.append({"round": 0, "feedback": feedback_context})
        return feedback_context

    async def run_improvement(
        self,
        initial_question: str,
        current_responses: dict[str, str],
        improvement_instruction: str,
        improvement_context: dict[str, dict[str, str]] | None = None,
        other_responses: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Unified improvement phase - models improve their answers."""
        self.logger.info("Improvement Phase", extra={"display_type": "section_header"})

        def build_improvement_prompt(model_key: str, model: BaseModel) -> str:
            display_name = self.anon_mapping[model_key]
            own_answer = current_responses[display_name]
            kb_context = self._get_knowledge_bank_context()

            return self.prompt_builder.build_improvement_prompt(
                initial_question=initial_question,
                own_answer=own_answer,
                improvement_instruction=improvement_instruction,
                kb_context=kb_context,
                improvement_context=improvement_context,
                other_responses=other_responses,
                model=model,
                display_name=display_name,
            )

        improved_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_improvement_prompt,
            context_for_logging="IMPROVEMENT",
        )

        self.logger.info("Improved Responses", extra={"display_type": "section_header"})
        for _display_name, _improved_text in improved_responses.items():
            # Response already logged immediately after generation
            pass

        return improved_responses

    def _select_largest_model_as_judge(self) -> str | None:
        """Selects the largest model based on context_window and max_tokens as emergency judge."""
        if not self.active_model_keys:
            return None

        best_model_key = None
        best_score = -1

        for model_key in self.active_model_keys:
            model = self.models[model_key]
            score = (model.context_window or 0) + (model.max_tokens or 0)
            if score > best_score:
                best_score = score
                best_model_key = model_key

        if best_model_key is not None:
            model = self.models[best_model_key]
            self.logger.info(
                f"Selected {model.display_name} as emergency judge " f"(context_window={model.context_window}, max_tokens={model.max_tokens})"
            )

        return best_model_key

    async def run_cross_evaluation(
        self,
        initial_question: str,
        responses: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        """Run the Cross-Evaluation Phase."""
        self.logger.info(f"Phase 3: Cross-Evaluation (Round {round_num})", extra={"display_type": "section_header"})

        if self.judge_model_key:
            self.logger.info(f"Using dedicated judge model for evaluation: {self.judge_model_key}")
            return await self._run_judge_evaluation(self.judge_model_key, initial_question, responses, round_num)
        else:
            self.logger.info("Using cross-evaluation (peer review). All models will evaluate each other.")
            result = await self._run_peer_evaluation(initial_question, responses, round_num)

            has_valid_scores = False
            if hasattr(self, "evaluation_scores") and self.evaluation_scores and isinstance(next(iter(self.evaluation_scores.values()), None), dict):
                has_valid_scores = any(scores for scores in self.evaluation_scores.values())

            if not has_valid_scores:
                self.logger.warning("⚠️  ALL peer evaluators failed to provide valid scores. " "Falling back to JUDGE MODE with largest model.")

                emergency_judge = self._select_largest_model_as_judge()
                if emergency_judge:
                    self.logger.info(f"🏛️  EMERGENCY JUDGE MODE: Using {self.models[emergency_judge].display_name}")
                    result = await self._run_judge_evaluation(emergency_judge, initial_question, responses, round_num)
                else:
                    self.logger.error("❌ CRITICAL: No model available for emergency judge fallback")

            return result

    async def _run_judge_evaluation(
        self,
        judge_model_key: str,
        initial_question: str,
        collaborative_responses: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        """Runs the evaluation using a single, designated judge model."""
        judge_display_name = self.models[judge_model_key].display_name
        self.logger.info(f"Using {judge_display_name} as the judge.", extra={"display_type": "colored_text", "color": "info"})

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {v: k for k, v in self.anon_mapping.items()}

        self.logger.info(f"Judge {judge_display_name} will evaluate:")
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info(f"  {anon_name} (actually {real_name})")

        formatted_responses = "\n\n".join(
            RESPONSE_WRAPPER.format(name=resp_name, response=resp_text) for resp_name, resp_text in shuffled_responses.items()
        )

        code_names = list(shuffled_responses.keys())
        evaluation_template = self.prompts.get("evaluate", "")
        prompt = self.prompt_builder.build_evaluation_prompt(initial_question, evaluation_template, formatted_responses, code_names)

        response = await self._execute_single_model_task(
            model_key=judge_model_key,
            prompt=prompt,
            context_for_logging="JUDGE_EVAL",
        )

        if response.is_error():
            self.logger.error(f"Error getting judge evaluation from {judge_model_key}: {response.error}")
            self.evaluation_scores = {}
            self.all_evaluations = {}
            return {}

        evaluation_text = response.content
        decoded_eval = self._decode_shuffled_names(evaluation_text, reverse_shuffle_mapping)

        log_message = LOG_JUDGE_EVALUATION.format(judge=judge_display_name, content=decoded_eval)
        self.logger.info(_indent_text(log_message))

        self.logger.info(
            f"\nEvaluation from Judge {judge_display_name}:",
            extra={"display_type": "colored_text", "color": judge_display_name},
        )
        self.logger.info(decoded_eval, extra={"display_type": "colored_text", "color": judge_display_name})

        code_names = list(shuffled_responses.keys())
        raw_scores = self.score_extractor.extract_scores_from_evaluation(
            evaluation_text=evaluation_text, model_names=code_names, evaluator_name=judge_display_name
        )

        mapped_scores = {reverse_shuffle_mapping.get(code, code): score for code, score in raw_scores.items()}

        self.all_evaluations = {judge_display_name: decoded_eval}
        self.evaluation_scores = mapped_scores

        self.logger.info(f"Judge {judge_display_name} provided scores: {mapped_scores}")
        return self.all_evaluations

    async def _run_peer_evaluation(
        self,
        initial_question: str,
        collaborative_responses: dict[str, str],
        round_num: int,
    ) -> dict[str, str]:
        """Runs the cross-evaluation phase where each model evaluates all other models."""
        self.prompts.get("evaluate", "")
        all_evaluations = {}
        evaluation_scores = {}

        shuffled_responses = collaborative_responses
        reverse_shuffle_mapping = {v: k for k, v in self.anon_mapping.items()}

        self.logger.info("Peer evaluation anonymization mapping:")
        for anon_name, real_name in reverse_shuffle_mapping.items():
            self.logger.info(f"  {anon_name} = {real_name}")

        def get_responses_for_evaluator(evaluator_anon_name: str) -> dict[str, str]:
            self.logger.debug(f"Evaluator {evaluator_anon_name} will evaluate {len(shuffled_responses)} responses: {list(shuffled_responses.keys())}")
            return shuffled_responses

        def build_peer_review_prompt(model_key: str, model: BaseModel) -> str:
            evaluator_anon_name = self.anon_mapping[model_key]
            evaluator_display_name = self.models[model_key].display_name

            self.logger.info(f"Evaluator {evaluator_display_name} (as {evaluator_anon_name}) is scoring:")
            for anon_name, real_name in reverse_shuffle_mapping.items():
                self.logger.info(f"  {anon_name} (labeled as {anon_name} in prompt, actually {real_name})")

            responses_to_evaluate = get_responses_for_evaluator(evaluator_anon_name)
            formatted_responses = "\n\n".join(
                RESPONSE_WRAPPER.format(name=resp_name, response=resp_text) for resp_name, resp_text in responses_to_evaluate.items()
            )
            code_names_to_evaluate = list(responses_to_evaluate.keys())
            self.logger.debug(
                f"Building evaluation prompt for {evaluator_anon_name}. "
                f"Formatted responses length: {len(formatted_responses)}, "
                f"Models to evaluate: {code_names_to_evaluate}"
            )
            evaluation_template = self.prompts.get("evaluate", "")
            prompt = self.prompt_builder.build_evaluation_prompt(
                initial_question,
                evaluation_template,
                formatted_responses,
                code_names_to_evaluate,
            )
            self.logger.debug(f"Full evaluation prompt for {evaluator_anon_name}:\n{prompt}")
            return prompt

        evaluator_responses = await self._execute_parallel_model_tasks(
            model_keys_to_run=self.active_model_keys,
            prompt_builder=build_peer_review_prompt,
            context_for_logging="PEER_EVAL",
        )

        for display_name, evaluation_text in evaluator_responses.items():
            responses_to_evaluate = get_responses_for_evaluator(display_name)
            code_names = list(responses_to_evaluate.keys())

            log_message = LOG_EVALUATOR_RESPONSE.format(evaluator=display_name, content=evaluation_text)
            self.logger.info(_indent_text(log_message))
            raw_scores = self.score_extractor.extract_scores_from_evaluation(
                evaluation_text=evaluation_text, model_names=code_names, evaluator_name=display_name
            )

            # Keep scores with anonymized keys - _get_aggregated_scores() expects them
            # raw_scores already has anonymized keys (LLM1, LLM2, etc.) from code_names
            if raw_scores:
                self.logger.info(f"Extracted scores from {display_name}: {raw_scores}")

            evaluation_scores[display_name] = raw_scores
            decoded_eval = self._decode_shuffled_names(evaluation_text, reverse_shuffle_mapping)

            all_evaluations[display_name] = decoded_eval
            self.logger.info(f"\nEvaluations from {display_name}:", extra={"display_type": "colored_text", "color": display_name})
            self.logger.info(decoded_eval, extra={"display_type": "colored_text", "color": display_name})

        self.evaluation_scores = evaluation_scores
        self.all_evaluations = all_evaluations
        return all_evaluations

    def _normalize_score(self, score: float, evaluator: str) -> float | None:
        """Normalizes a score to a 1-10 scale, or rejects invalid scores."""
        # Reject scores outside the valid range [0.5, 10.5] to avoid garbage
        if score < 0.5 or score > 10.5:
            self.logger.error(f"Rejecting invalid score from {evaluator}: {score} (must be between 1.0 and 10.0)")
            return None  # Signal invalid score

        # Normalize scores that are slightly out of bounds
        if score > 10:
            normalized = min(score / 10.0, 10.0)  # Cap at 10.0
            self.logger.warning(f"Normalizing score from {evaluator}; received {score}, converted to {normalized:.2f}.")
            return normalized
        if 0 < score < 1:
            normalized = max(score * 10.0, 1.0)  # Floor at 1.0
            self.logger.warning(f"Normalizing score from {evaluator}; received {score}, converted to {normalized:.2f}.")
            return normalized

        # Score is already in valid range
        return score

    def _get_aggregated_scores(self) -> dict[str, float]:
        """Aggregates evaluation scores from peer review or single judge."""
        first_score_value = next(iter(self.evaluation_scores.values()))
        is_peer_review = isinstance(first_score_value, dict)

        if is_peer_review:
            self.logger.info("Aggregating scores from peer-review format.")
            aggregated_scores: dict[str, float] = {}
            valid_evaluators = [evaluator for evaluator, evaluations in self.evaluation_scores.items() if evaluations]
            invalid_evaluators = [evaluator for evaluator, evaluations in self.evaluation_scores.items() if not evaluations]

            if invalid_evaluators:
                self.logger.warning(
                    f"⚠️  {len(invalid_evaluators)} evaluator(s) provided invalid evaluations and were excluded: {', '.join(invalid_evaluators)}"
                )

            for model_display_name in [self.anon_mapping[k] for k in self.active_model_keys]:
                scores: list[float] = []
                self.logger.info(f"Collecting scores for {model_display_name}:")
                for evaluator in valid_evaluators:
                    evaluations = self.evaluation_scores[evaluator]
                    if isinstance(evaluations, dict) and model_display_name in evaluations:
                        score_val = evaluations[model_display_name]
                        if isinstance(score_val, (int, float)):
                            normalized_score = self._normalize_score(float(score_val), evaluator)
                            if normalized_score is not None:  # Only use valid scores
                                scores.append(normalized_score)
                                self.logger.info(
                                    f"  - {evaluator} gave {model_display_name} a score of {score_val} (normalized to {normalized_score:.2f})"
                                )
                            else:
                                self.logger.warning(f"  - {evaluator} gave {model_display_name} an invalid score of {score_val} (rejected)")

                if scores:
                    median = statistics.median(scores)
                    aggregated_scores[model_display_name] = median
                    self.logger.info(f"  → Median score for {model_display_name}: {median:.2f} (from {len(scores)} valid evaluations)")
                    self._display_model_score(
                        model_display_name,
                        aggregated_scores[model_display_name],
                        "Median score",
                    )
                else:
                    self.logger.warning(f"No valid scores found for {model_display_name}. Assigning a penalty score of 0.0.")
                    aggregated_scores[model_display_name] = 0.0
            return aggregated_scores
        else:
            self.logger.info("Processing scores from single-judge format.")
            result_scores: dict[str, float] = {}
            evaluator_name = next(iter(self.all_evaluations.keys()), "judge")
            for model_name, score in self.evaluation_scores.items():
                if isinstance(score, (int, float)):
                    score_float = float(score)
                    normalized_score = self._normalize_score(score_float, evaluator_name)
                    if normalized_score is not None:  # Only use valid scores
                        self._display_model_score(model_name, normalized_score, "Judge score")
                        result_scores[model_name] = normalized_score
                    else:
                        self.logger.warning(f"Judge gave {model_name} an invalid score of {score_float} (rejected)")
            return result_scores

    async def determine_lowest_and_highest_ranked_models(self) -> tuple[str, str]:
        """Aggregates evaluation scores to determine the lowest and highest performing models.

        Also sets self.elimination_reason to track if elimination was random or score-based.
        """
        if not hasattr(self, "evaluation_scores") or not self.evaluation_scores:
            self.logger.error("No evaluation scores available for elimination decision. Falling back to random selection.")
            active_names = [self.anon_mapping[k] for k in self.active_model_keys]
            chosen = self.anonymizer.rng.choice(active_names)
            self.elimination_reason = "Random selection (no evaluation scores available)"
            self.elimination_score = None
            return chosen, self.anonymizer.rng.choice(active_names)

        self.logger.info("Aggregating Evaluation Scores", extra={"display_type": "section_header"})
        aggregated_scores = self._get_aggregated_scores()

        all_active_models = {self.anon_mapping[k] for k in self.active_model_keys}

        # active_scores only contains models that received at least one valid score
        active_scores = {k: v for k, v in aggregated_scores.items() if k in all_active_models and v is not None}

        unscored_models = all_active_models - set(active_scores.keys())

        if unscored_models:
            lowest_model_name = self.anonymizer.rng.choice(list(unscored_models))
            self.logger.warning(
                f"Models {list(unscored_models)} were not scored by any evaluator. " f"Randomly selecting {lowest_model_name} for elimination."
            )
            self.elimination_reason = "Random selection (model was not scored by any evaluator)"
            self.elimination_score = None

            if not active_scores:
                # All models were unscored, so pick a random leader from the rest
                remaining_models = list(all_active_models - {lowest_model_name})
                highest_model_name = self.anonymizer.rng.choice(remaining_models) if remaining_models else lowest_model_name
                self.logger.warning("No models received scores. Randomly selecting leader.")
            else:
                # At least one model was scored, so find the best among them
                max_score = max(active_scores.values())
                models_with_max = [name for name, score in active_scores.items() if score == max_score]
                highest_model_name = self.anonymizer.rng.choice(models_with_max)

        elif not active_scores:
            self.logger.error("CRITICAL: No models have scores and no unscored models found. Cannot determine ranking.")
            # Fallback to random choice among all active models
            active_names = list(all_active_models)
            chosen = self.anonymizer.rng.choice(active_names)
            self.elimination_reason = "Random selection (critical error - no valid scores)"
            self.elimination_score = None
            return chosen, self.anonymizer.rng.choice(active_names)

        else:  # All models were scored
            self.logger.info("All models received scores. Determining winner and loser based on scores.")
            min_score = min(active_scores.values())
            max_score = max(active_scores.values())

            models_with_min = [name for name, score in active_scores.items() if score == min_score]
            models_with_max = [name for name, score in active_scores.items() if score == max_score]

            if len(models_with_min) > 1:
                lowest_model_name = self.anonymizer.rng.choice(models_with_min)
                self.logger.warning(
                    f"Tie for lowest score ({min_score:.2f}): {models_with_min}. " f"Randomly selected {lowest_model_name} for elimination."
                )
                self.elimination_reason = f"Random selection among tied models (tied at score {min_score:.2f})"
                self.elimination_score = min_score
            else:
                lowest_model_name = models_with_min[0]
                self.elimination_reason = "Lowest score in evaluation"
                self.elimination_score = min_score

            if len(models_with_max) > 1:
                highest_model_name = self.anonymizer.rng.choice(models_with_max)
                self.logger.info(f"Tie for highest score ({max_score:.2f}): {models_with_max}. Randomly selected {highest_model_name} as leader.")
            else:
                highest_model_name = models_with_max[0]

            if lowest_model_name == highest_model_name and len(active_scores) > 1:
                self.logger.warning(f"All models tied with score {min_score:.2f}. Randomly selecting for elimination and leadership.")
                other_models = [name for name in active_scores.keys() if name != lowest_model_name]
                if other_models:
                    highest_model_name = self.anonymizer.rng.choice(other_models)
                self.logger.info(f"Randomly selecting {lowest_model_name} to be eliminated and {highest_model_name} to lead.")
                self.elimination_reason = f"Random selection (all models tied at {min_score:.2f})"
                self.elimination_score = min_score

        self.current_leader_key = next(
            (key for key, name in self.anon_mapping.items() if name == highest_model_name),
            None,
        )

        highest_score_val = active_scores.get(highest_model_name, float("nan"))
        lowest_score_val = active_scores.get(lowest_model_name, 0.0)  # Unscored models have a de-facto 0.0

        self.logger.info(
            f"\n🏆 Highest-ranked model: {highest_model_name} with score {highest_score_val:.2f}/10",
            extra={"display_type": "colored_text", "color": "success"},
        )
        self.logger.info(
            f"❌ Lowest-ranked model: {lowest_model_name} with score {lowest_score_val:.2f}/10",
            extra={"display_type": "colored_text", "color": "warning"},
        )

        return lowest_model_name, highest_model_name

    def _get_phase_2_criticisms(self) -> Any | None:
        """Extract Phase 2 criticisms from history."""
        if not self.criticism_history:
            return None
        for crit_entry in self.criticism_history:
            if crit_entry.get("round") == 0:
                return crit_entry.get("criticisms")
        return None

    def _get_phase_2_feedback(self) -> Any | None:
        """Extract Phase 2 feedback from history."""
        if not self.feedback_history:
            return None
        for feedback_entry in self.feedback_history:
            if feedback_entry.get("round") == 0:
                return feedback_entry.get("feedback")
        return None

    def _build_phase_2_data(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Build Phase 2 data for tournament history."""
        phase_2_criticisms = self._get_phase_2_criticisms()
        phase_2_feedback = self._get_phase_2_feedback()

        phase_2_data = {}
        if phase_2_criticisms:
            phase_2_data["criticisms"] = phase_2_criticisms
            phase_2_data["improved_answers"] = all_previous_answers[1].copy()
            return {"Phase 2: Cross-Criticism & Self-Improvement": phase_2_data}
        elif phase_2_feedback:
            phase_2_data["feedback"] = phase_2_feedback
            phase_2_data["enhanced_answers"] = all_previous_answers[1].copy()
            return {"Phase 2: Positive Reinforcement & Strength Amplification": phase_2_data}
        else:
            return {"Phase 2: Collaborative Analysis": all_previous_answers[1].copy()}

    def _add_round_evaluations(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        """Add evaluations for a round if available."""
        if hasattr(self, "evaluation_history") and elimination_round - 1 < len(self.evaluation_history):
            eval_data = self.evaluation_history[elimination_round - 1]
            round_data["evaluations"] = eval_data.get("evaluations", {})
            round_data["scores"] = eval_data.get("scores", {})

    def _add_round_criticisms(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        """Add criticisms for a round if available."""
        if not hasattr(self, "criticism_history"):
            return
        for crit_entry in self.criticism_history:
            if crit_entry.get("round") == elimination_round:
                round_data["criticisms"] = crit_entry.get("criticisms", {})
                break

    def _add_round_feedback(
        self,
        round_data: dict[str, Any],
        elimination_round: int,
    ) -> None:
        """Add feedback for a round if available."""
        if not hasattr(self, "feedback_history"):
            return
        for feedback_entry in self.feedback_history:
            if feedback_entry.get("round") == elimination_round:
                round_data["feedback"] = feedback_entry.get("feedback", {})
                break

    def _build_elimination_rounds(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Build elimination rounds data for tournament history."""
        tournament_data = {}
        for elimination_round, i in enumerate(range(2, len(all_previous_answers)), start=1):
            round_data: dict[str, Any] = {}
            self._add_round_evaluations(round_data, elimination_round)
            self._add_round_criticisms(round_data, elimination_round)
            self._add_round_feedback(round_data, elimination_round)
            round_data["refined_answers"] = all_previous_answers[i].copy()
            tournament_data[f"Elimination Round {elimination_round}"] = round_data
        return tournament_data

    def _build_tournament_history(
        self,
        all_previous_answers: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Build complete tournament history with all phases."""
        tournament_history: dict[str, Any] = {}

        if len(all_previous_answers) > 0:
            tournament_history["Phase 1: Initial Answers"] = all_previous_answers[0].copy()

        if len(all_previous_answers) > 1:
            tournament_history.update(self._build_phase_2_data(all_previous_answers))

        tournament_history.update(self._build_elimination_rounds(all_previous_answers))

        return tournament_history

    async def _save_champion_report(
        self,
        initial_question: str,
        final_model_anon: str,
        champion_answer: str,
        all_previous_answers: list[dict[str, str]],
    ) -> None:
        """Saves the final champion report to disk with complete tournament history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tournament_history = self._build_tournament_history(all_previous_answers)

        report_content = {
            "initial_question": initial_question,
            "champion_model": f"{final_model_anon} (model: {self.models[self.active_model_keys[0]].full_display_name})",
            "champion_solution": champion_answer,
            "total_cost": f"${self.total_cost:.4f}",
            "cost_by_model": {k: f"${v:.4f}" for k, v in self.cost_by_model.items()},
            "eliminated_models": self.eliminated_models.copy(),
            "complete_tournament_history": tournament_history,
        }

        await self.report_generator.save_report_to_file(content_type=f"champion_solution_{timestamp}", content=report_content)

        from arbitrium.utils.provenance import generate_provenance_report

        champion_model_str = str(report_content["champion_model"])
        provenance_paths = generate_provenance_report(
            question=initial_question,
            champion_model=champion_model_str,
            champion_answer=champion_answer,
            tournament_data=report_content,
            output_dir="reports",  # This should be from the host
        )
        self.logger.info(f"📋 Provenance report saved: {provenance_paths['markdown']}")

    def _display_cost_summary(self) -> None:
        """Display cost summary for the tournament."""
        self.logger.info("Cost Summary", extra={"display_type": "section_header"})
        self.logger.info(f"💰 Total Cost: ${self.total_cost:.4f}", extra={"display_type": "colored_text"})

        if self.cost_by_model:
            self.logger.info("\n📊 Cost by Model:", extra={"display_type": "colored_text"})
            for model_name, cost in sorted(self.cost_by_model.items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / self.total_cost) * 100 if self.total_cost > 0 else 0
                self.logger.info(f"  {model_name}: ${cost:.4f} ({percentage:.1f}%)", extra={"display_type": "colored_text"})
        else:
            self.logger.info(
                "📊 No cost information available (cost tracking may be disabled)",
                extra={"display_type": "colored_text"},
            )

        self.logger.info(f"💰 Tournament total cost: ${self.total_cost:.4f}")

    async def run(self, initial_question: str) -> str:
        """Runs the complete model comparison tournament."""
        try:
            return await self.runner.run(initial_question)
        finally:
            self._display_cost_summary()
            self.logger.info("Arbitrium Framework Tournament Complete", extra={"display_type": "section_header"})
            # self.display.reset() # This needs to be handled by the event handler
