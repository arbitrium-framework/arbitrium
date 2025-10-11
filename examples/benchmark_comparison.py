#!/usr/bin/env python3
"""
Arbitrium Framework - Benchmark Comparison Example

Compare single-model vs tournament performance on your own questions.
Perfect for: Evaluating whether tournament is worth the cost.

Expected runtime: ~10-20 minutes
Expected cost: ~$1-3 (compares multiple approaches)
"""

import asyncio
import time

from arbitrium import Arbitrium


async def main():
    """Run a micro-benchmark: Single model vs Tournament."""

    arbitrium = await Arbitrium.from_config("config.example.yml")

    print("=" * 80)
    print("⚖️  SINGLE MODEL vs TOURNAMENT BENCHMARK")
    print("=" * 80)

    # Test question
    question = """
    Evaluate whether we should build our MVP with React or Vue.js.

    Context:
    - Small team (2 developers)
    - Timeline: 3 months to MVP
    - Need to hire more devs later
    - Product: B2B SaaS dashboard
    """

    print(f"\n❓ Test Question:\n{question}\n")

    # ------------------------------------------------------------------
    # APPROACH 1: Single Model (fastest, cheapest)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("🤖 APPROACH 1: Single Model (GPT-5)")
    print("=" * 80)

    start = time.time()
    single_response = await arbitrium.run_single_model("gpt", question)
    single_time = time.time() - start

    print(f"\n✅ Completed in {single_time:.1f}s")
    print(f"💰 Cost: ${single_response.cost:.4f}")
    print("\n📝 Response Preview:")
    print(single_response.content[:300] + "...")

    # ------------------------------------------------------------------
    # APPROACH 2: All Models (no tournament, just comparison)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔄 APPROACH 2: All Models (No Tournament)")
    print("=" * 80)

    start = time.time()
    all_responses = await arbitrium.run_all_models(question)
    all_time = time.time() - start

    all_cost = sum(r.cost for r in all_responses.values() if not r.is_error())

    print(f"\n✅ Completed in {all_time:.1f}s")
    print(f"💰 Total Cost: ${all_cost:.4f}")
    print("\n📊 Individual Costs:")
    for model_key, response in all_responses.items():
        if not response.is_error():
            print(f"   {model_key}: ${response.cost:.4f}")

    # ------------------------------------------------------------------
    # APPROACH 3: Tournament (synthesis + elimination)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏆 APPROACH 3: Tournament")
    print("=" * 80)

    start = time.time()
    _tournament_result, tournament_metrics = await arbitrium.run_tournament(question)
    tournament_time = time.time() - start

    print(f"\n✅ Completed in {tournament_time:.1f}s")
    print(f"💰 Total Cost: ${tournament_metrics['total_cost']:.4f}")
    print(f"🏆 Champion: {tournament_metrics['champion_model']}")
    print(f"🗑️  Eliminated: {len(tournament_metrics['eliminated_models'])} models")

    # ------------------------------------------------------------------
    # COMPARISON TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Approach':<30} {'Time':>10} {'Cost':>10} {'Models':>10}")
    print("-" * 80)
    print(f"{'1. Single Model (GPT-5)':<30} {single_time:>9.1f}s ${single_response.cost:>8.4f} {1:>10}")
    print(f"{'2. All Models (Independent)':<30} {all_time:>9.1f}s ${all_cost:>8.4f} {len(all_responses):>10}")
    print(f"{'3. Tournament (Synthesis)':<30} {tournament_time:>9.1f}s ${tournament_metrics['total_cost']:>8.4f} {arbitrium.healthy_model_count:>10}")

    # Cost multipliers
    tournament_cost_mult = tournament_metrics["total_cost"] / single_response.cost
    tournament_time_mult = tournament_time / single_time

    print(f"\n💡 Tournament Cost Multiple: {tournament_cost_mult:.1f}x")
    print(f"⏱️  Tournament Time Multiple: {tournament_time_mult:.1f}x")

    # ------------------------------------------------------------------
    # WHEN TO USE EACH
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)

    print(
        f"""
    Use Single Model when:
    ✅ Budget < $0.20 per query
    ✅ Time sensitive (< 1 minute)
    ✅ Exploratory questions
    ✅ Reversible decisions

    Use All Models when:
    ✅ Need multiple perspectives
    ✅ But don't need synthesis
    ✅ Will manually review responses

    Use Tournament when:
    ✅ Decision value > $1,000 ({1000/tournament_metrics['total_cost']:.0f}x cost)
    ✅ Irreversible decision
    ✅ Stakeholder buy-in needed (synthesis helps)
    ✅ Worth extra {tournament_time-single_time:.0f} seconds for quality
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
