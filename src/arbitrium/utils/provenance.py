"""
Provenance tracking and report generation for Arbitrium Framework tournaments.

Provides audit trail showing:
- Which model proposed each idea
- How ideas were critiqued and refined
- Why competing ideas were rejected
- What insights came from eliminated models
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ProvenanceReport:
    """Generate provenance reports showing idea evolution through tournament."""

    def __init__(
        self,
        question: str,
        champion_model: str,
        champion_answer: str,
        tournament_data: dict[str, Any],
    ):
        """
        Initialize provenance report.

        Args:
            question: The initial question posed to the tournament
            champion_model: The winning model's display name
            champion_answer: The final champion answer
            tournament_data: Complete tournament history data
        """
        self.question = question
        self.champion_model = champion_model
        self.champion_answer = champion_answer
        self.tournament_data = tournament_data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json(self) -> dict[str, Any]:
        """Generate structured JSON provenance report."""
        report = {
            "tournament_id": self.timestamp,
            "question": self.question,
            "champion_model": self.champion_model,
            "final_answer": self.champion_answer,
            "phases": self._extract_phases(),
            "eliminations": self._extract_eliminations(),
            "cost_summary": self.tournament_data.get("cost_summary", {}),
            "generated_at": datetime.now().isoformat(),
        }
        return report

    def _format_phase_initial(self, md: list[str], phase: dict[str, Any]) -> None:
        """Format initial phase responses."""
        md.append("All models provided initial responses independently.")
        md.append("")
        for model, answer in phase.get("responses", {}).items():
            md.append(f"{model}:")
            md.append("```")
            md.append(f"{answer}")
            md.append("```")
            md.append("")

    def _format_phase_improvement(self, md: list[str], phase: dict[str, Any]) -> None:
        """Format improvement phase with critiques and feedback."""
        md.append(f"Models improved their answers based on {phase.get('strategy', 'feedback')}.")
        md.append("")

        # Show criticisms if available
        if "criticisms" in phase and phase["criticisms"] is not None:
            md.append("#### Critiques Exchanged")
            md.append("")
            for target, critics in phase["criticisms"].items():
                md.append(f"{target} received feedback from:")
                for critic_model, critique in critics.items():
                    md.append(f"- *{critic_model}:* {critique}")
                md.append("")

        # Show positive feedback if available
        if "feedback" in phase and phase["feedback"] is not None:
            md.append("#### Positive Feedback Exchanged")
            md.append("")
            for target, supporters in phase["feedback"].items():
                md.append(f"{target} received support from:")
                for supporter, feedback in supporters.items():
                    md.append(f"- *{supporter}:* {feedback}")
                md.append("")

    def _format_phase_evaluation(self, md: list[str], phase: dict[str, Any]) -> None:
        """Format evaluation phase with scores."""
        md.append("Models were evaluated and ranked.")
        md.append("")

        scores = phase.get("scores", {})
        if scores:
            md.append("Scores:")
            md.append("")
            # Filter out non-numeric scores and sort
            numeric_scores = {model: score for model, score in scores.items() if isinstance(score, (int, float))}
            if numeric_scores:
                for model, score in sorted(numeric_scores.items(), key=lambda x: x[1], reverse=True):
                    md.append(f"- {model}: {score:.2f}/10")
                md.append("")

    def _format_eliminations(self, md: list[str]) -> None:
        """Format elimination information."""
        eliminations = self._extract_eliminations()
        if not eliminations:
            return

        md.append("## Eliminations")
        md.append("")

        for elim in eliminations:
            md.append(f"### Round {elim['round']}: {elim['model']} Eliminated")
            md.append("")
            md.append(f"Score: {elim.get('score', 'N/A')}")
            md.append("")
            if "reason" in elim:
                md.append(f"Reason: {elim['reason']}")
                md.append("")

            if elim.get("insights_preserved"):
                md.append("Insights Preserved from this Model:")
                md.append("")
                for insight in elim["insights_preserved"][:3]:
                    md.append(f"- {insight}")
                md.append("")

            md.append("---")
            md.append("")

    def _format_cost_summary(self, md: list[str]) -> None:
        """Format cost summary information."""
        cost = self.tournament_data.get("total_cost")
        if not cost:
            return

        md.append("## Cost Summary")
        md.append("")
        # Handle cost as either float or string
        if isinstance(cost, str):
            md.append(f"Total: {cost}")
        else:
            md.append(f"Total: ${cost:.4f}")
        md.append("")

        cost_by_model = self.tournament_data.get("cost_by_model", {})
        if cost_by_model:
            md.append("Cost by Model:")
            md.append("")

            # model_cost уже отформатирован как строка "$X.XXXX" в comparison.py
            # Извлекаем числовое значение для сортировки
            def get_numeric_cost(item: tuple[str, Any]) -> float:
                cost_str = item[1]
                if isinstance(cost_str, str):
                    # Убираем "$" и конвертируем в float
                    return float(cost_str.replace("$", ""))
                return float(cost_str)

            for model, model_cost in sorted(cost_by_model.items(), key=get_numeric_cost, reverse=True):
                md.append(f"- {model}: {model_cost}")
            md.append("")

    def generate_markdown(self) -> str:
        """Generate human-readable Markdown provenance report."""
        md = []

        # Header
        md.append("# Tournament Provenance Report")
        md.append("")
        md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")

        # Question
        md.append("## Question")
        md.append("")
        md.append(f"{self.question}")
        md.append("")

        # Champion
        md.append(f"## Champion: {self.champion_model}")
        md.append("")
        md.append("### Final Answer")
        md.append("")
        md.append(f"{self.champion_answer}")
        md.append("")

        # Tournament Evolution
        md.append("## Tournament Evolution")
        md.append("")

        phases = self._extract_phases()
        for phase in phases:
            md.append(f"### {phase['name']}")
            md.append("")

            if phase["type"] == "initial":
                self._format_phase_initial(md, phase)
            elif phase["type"] == "improvement":
                self._format_phase_improvement(md, phase)
            elif phase["type"] == "evaluation":
                self._format_phase_evaluation(md, phase)

            md.append("---")
            md.append("")

        # Eliminations
        self._format_eliminations(md)

        # Cost Summary
        self._format_cost_summary(md)

        return "\n".join(md)

    def _extract_phases(self) -> list[dict[str, Any]]:
        """Extract tournament phases from history data."""
        phases = []
        history = self.tournament_data.get("complete_tournament_history", {})

        for phase_name, phase_data in history.items():
            if "Initial" in phase_name:
                phases.append(
                    {
                        "name": phase_name,
                        "type": "initial",
                        "responses": phase_data,
                    }
                )
            elif "Cross-Criticism" in phase_name or "Positive Reinforcement" in phase_name or "Collaborative" in phase_name:
                phase_type = "improvement"
                phases.append(
                    {
                        "name": phase_name,
                        "type": phase_type,
                        "strategy": (
                            "cross-criticism"
                            if "Criticism" in phase_name
                            else "positive reinforcement" if "Positive" in phase_name else "collaborative"
                        ),
                        "criticisms": phase_data.get("criticisms"),
                        "feedback": phase_data.get("feedback"),
                        "improved_answers": phase_data.get("improved_answers") or phase_data.get("enhanced_answers"),
                    }
                )
            elif "Elimination Round" in phase_name:
                phases.append(
                    {
                        "name": phase_name,
                        "type": "evaluation",
                        "scores": phase_data.get("scores"),
                        "evaluations": phase_data.get("evaluations"),
                        "refined_answers": phase_data.get("refined_answers"),
                    }
                )

        return phases

    def _extract_eliminations(self) -> list[dict[str, Any]]:
        """Extract elimination events from tournament data."""
        eliminations = []
        eliminated_models = self.tournament_data.get("eliminated_models", [])

        for i, elim_data in enumerate(eliminated_models, 1):
            if isinstance(elim_data, dict):
                eliminations.append(
                    {
                        "round": i,
                        "model": elim_data.get("model", "Unknown"),
                        "score": elim_data.get("score"),
                        "reason": elim_data.get("reason", "Lowest score in evaluation"),
                        "insights_preserved": elim_data.get("insights_preserved", []),
                    }
                )
            else:
                # Legacy format: just model name
                eliminations.append(
                    {
                        "round": i,
                        "model": str(elim_data),
                        "score": None,
                        "reason": "Lowest score in evaluation",
                        "insights_preserved": [],
                    }
                )

        return eliminations

    def save_to_file(self, output_dir: str) -> dict[str, str]:
        """
        Save provenance report to both JSON and Markdown formats.

        Args:
            output_dir: Directory where reports should be saved

        Returns:
            Dict with paths to saved files
        """
        output_path = Path(output_dir)

        # Save JSON
        json_filename = f"arbitrium_provenance_{self.timestamp}.json"
        json_path = output_path / json_filename
        with open(json_path, "w") as f:
            json.dump(self.generate_json(), f, indent=2)

        # Save Markdown
        md_filename = f"arbitrium_provenance_{self.timestamp}.md"
        md_path = output_path / md_filename
        with open(md_path, "w") as f:
            f.write(self.generate_markdown())

        return {
            "json": str(json_path),
            "markdown": str(md_path),
        }


def generate_provenance_report(
    question: str,
    champion_model: str,
    champion_answer: str,
    tournament_data: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    """
    Generate and save provenance report.

    Args:
        question: The initial question
        champion_model: Winning model name
        champion_answer: Final champion answer
        tournament_data: Complete tournament history
        output_dir: Directory where reports should be saved

    Returns:
        Dict with paths to saved files
    """
    report = ProvenanceReport(
        question=question,
        champion_model=champion_model,
        champion_answer=champion_answer,
        tournament_data=tournament_data,
    )

    return report.save_to_file(output_dir)
