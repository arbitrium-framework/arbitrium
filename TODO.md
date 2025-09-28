# Arbitrium™ Framework - Development Roadmap

> **Status:** Pre-MVP, solo development | **Last updated:** 2025-01-05

---

## I. IMMEDIATE ACTIONS (Pre-MVP Phase)

### Goal
Validate core hypothesis with minimal resources before investing in extensive validation.

### Quick Validation Protocol (Solo Developer, Zero Budget)

#### Week 1-2: Free Model Benchmarks

**Setup:**
- Use **Ollama** (llama3, mistral, gemma) - completely free
- 5-10 test questions across domains
- Compare outputs: Single model vs CoT vs Arbitrium tournament

**Evaluation:**
- Self-eval: read outputs blind, score qualitatively
- Document where Arbitrium clearly wins vs doesn't

**Tools:**
- `benchmarks/micro_benchmark.py`
- `benchmarks/standard_benchmarks.py`

**Success:** Tournament noticeably better in ≥3/5 questions

#### Week 3-4: Knowledge Bank Critical Test

**Question:** Does KB actually add value?

**Test:**
- Same questions, tournaments with KB on/off
- Compare outputs side-by-side

**Decision:** If KB doesn't help → REMOVE IT (reduce complexity)

**Tool:** `tests/integration/test_kb_quick_validation.py`

---

### First Release Checklist (v0.1.0)

**Pre-release:**
- [x] Rebranding complete (Arbitrium Framework)
- [x] EUIPO trademark filed (№ 019256589, 900€ paid)
- [x] GitHub org created (arbitrium-framework/arbitrium)
- [ ] 5 benchmark runs documented
- [ ] README with clear use cases
- [ ] All tests pass

**Release:**
- [ ] Tag v0.1.0
- [ ] GitHub Release notes
- [ ] Publish to PyPI (`arbitrium-framework`)
- [ ] Announcement (Dev.to/Habr - optional)

**Legal:**
- [x] MIT License
- [x] TRADEMARKS.md
- [x] NOTICE file
- [x] Disclaimer in README
- [x] Arbitrium™ symbol everywhere

---

### Technical TODOs (Prioritized)

**High (Pre-MVP):**
- [ ] Clean up remaining `agentsphere` in comments
- [ ] Add full tournament integration test
- [ ] Better error messages
- [ ] Add `--version` flag
- [ ] Reduce logging verbosity

**Medium (v0.2):**
- [ ] Progress bar for tournaments
- [ ] Parallel model execution
- [ ] Tournament checkpoints/resume
- [ ] Export to markdown/PDF
- [ ] More config examples

**Low (Future):**
- [ ] KB semantic clustering
- [ ] Provenance visualization
- [ ] Interactive judge mode
- [ ] Web UI

---

## II. LONG-TERM VALIDATION (When Resources Allow)

> **Note:** This is the FULL scientific validation plan. Only execute when you have:
> - Budget: $500-2000 for human evaluators
> - Time: 6-8 weeks
> - Goal: Publish results as evidence

### Scientific Validation Framework

#### Core Hypothesis

```
Tournament-based multi-model elimination with Knowledge Bank produces
demonstrably superior results for complex, multi-faceted tasks compared
to single-model approaches, and this improvement justifies higher costs.
```

#### Validation Protocol (Rigorous)

##### Stage 1: Question Set Development (Week 1)

Create 25 questions across 5 validated domains:

1. **Technical Architecture (5)** - Example in `micro_benchmark.py`
2. **Multi-stakeholder Policy Analysis (5)** - Example in `test_kb_quick_validation.py`
3. **Strategic Business (5)** - Example in `test_kb_quick_validation.py`
4. **Research Synthesis (5)** - e.g., "Scientific consensus on intermittent fasting given 50+ contradictory studies"
5. **Ethical Dilemmas (5)** - e.g., "Should autonomous vehicles prioritize passenger safety or minimize total casualties?"

##### Stage 2: Experimental Conditions (Weeks 2-3) - Ablation Study

**Philosophy:** Each condition isolates one variable for causal attribution

**Condition A: Best Single Model (Baseline 1)**
- Setup: Claude 3.5 Sonnet, question only, temp 0.7
- Goal: Establish baseline

**Condition B: Single Model + Chain-of-Thought (Baseline 2)**
- Setup: Claude 3.5 Sonnet + "Think step by step..."
- Goal: Test if simple prompting achieves similar results

**Condition C: Arbitrium Tournament (Main Test)**
- Setup: GPT-4 Turbo, Claude 3.5 Sonnet, Gemini 1.5 Pro, Grok 2; external judge; KB enabled
- Goal: Test full product functionality

**Condition D: Arbitrium without Knowledge Bank (Control)**
- Setup: Same as C, but `knowledge_bank.enabled: false`
- Goal: Isolate KB contribution - **CRITICAL:** If no value, remove KB

**Condition E: Arbitrium with Peer Review (Judge Ablation)**
- Setup: Same as C, but `judge_model: null` (models judge each other)
- Goal: Test if external judge is necessary

**Condition F: Arbitrium with Multiple Judges (Judge Reliability)**
- Setup: Same as C, but `judge_model: ["claude", "gpt-4", "gemini"]`
- Goal: Test single judge bias

##### Stage 3: Blind Human Evaluation (Week 4)

**Protocol:**
1. **Recruit:** 5 evaluators per question (compensated, Upwork/Prolific)
2. **Blind:** Shuffle outputs from conditions A-F, anonymize
3. **Criteria (1-10 scale):** Accuracy, Completeness, Depth, Actionability, Overall Quality
4. **Analysis:** ANOVA, t-tests, Cohen's d effect size

**Success Criteria (ALL must be met):**
1. Condition C ≥15% better than B (p < 0.05)
2. Cohen's d ≥ 0.5 (medium effect) on Overall Quality
3. Improvement in ≥3/5 domains
4. Cost per quality justified for ≥$1,000 decisions

**Feature Removal Criteria (Ablation-based):**
1. If C ≈ D (p ≥ 0.05): **Remove Knowledge Bank**
2. If C ≈ E (p ≥ 0.05): Use peer review (simpler)
3. If F ≈ C (p ≥ 0.05): Single judge sufficient

---

## III. LEGAL & COMPLIANCE

### EUIPO Trademark

**Application Details:**
- Number: **019256589**
- Mark: **ARBITRIUM** (word mark)
- Classes: **9, 42**
- Status: Filed, payment confirmed (900€)
- Payment ref: EEFEM202500001699668

**Timeline:**
- Formal examination → Publication
- 3-month opposition period → Registration

**Actions:**
- [x] Payment transferred
- [x] Confirmation saved
- [ ] Set calendar reminder for opposition period
- [ ] Monitor EUIPO email alerts

**Usage:**
- Until registration: **Arbitrium™**
- After registration: **Arbitrium®**

**Descriptions used (classes 9/42):**
- **Class 9:** Downloadable computer software; open-source software libraries and frameworks for orchestration of large language model agents; recorded software for multi-agent tournaments, agent elimination workflows, and knowledge management.
- **Class 42:** Software as a Service (SaaS) featuring frameworks for LLM agent orchestration; design and development of computer software; providing online non-downloadable software for tournament-based decision synthesis, agent elimination, and knowledge bank injection.

### CrowdStrike COI Compliance

**Context:** Personal side project, must maintain complete separation from employer.

**COI Disclosure Template (for Day 1 onboarding):**
```
Project: Arbitrium Framework — open-source tournament-based LLM orchestration
        (agent elimination + knowledge bank)

Nature: Personal OSS, MIT license; non-commercial at this time

Separation: Personal laptop, personal accounts/domains, no company systems/VPN;
           outside working hours only

Overlap: None with CrowdStrike confidential information, products, or roadmap;
        NOT a cybersecurity product; no engagements with competitors/customers

Time: ≤8 hours/week, evenings/weekends

Branding: Includes trademark filing for the name; purely personal
```

**Red Lines (NEVER cross):**
- ❌ Security use cases or integrations
- ❌ Work during business hours
- ❌ Company equipment/VPN/accounts
- ❌ Recruiting colleagues
- ❌ Paid consulting in cybersecurity domain

**Evidence of Separation:**
- Personal laptop, email (nikolay.eremeev@outlook.com)
- Personal GitHub org (arbitrium-framework)
- Personal API keys, domains
- Commit timestamps outside work hours
- Explicit disclaimer in README

**What to watch:**
- IP assignment: No company code/methodologies in project
- Scope creep: If adding security features → update COI first
- Time management: No project responses during core hours

---

## IV. OPEN QUESTIONS / DECISIONS NEEDED

1. **Validation approach?**
   - Option A: Start with free Ollama models (reproducible, no cost)
   - Option B: Use paid APIs (closer to production quality)
   - **Decision:** Start with A, add B later

2. **Default tournament size?**
   - Current: 4 models
   - Question: Is 3 enough? Does 5+ add value?
   - **Need:** Quick benchmark to test

3. **Knowledge Bank fate?**
   - Depends on validation
   - If no improvement → REMOVE (reduce complexity)
   - **Decision point:** After Week 3-4 testing

4. **Judge strategy?**
   - Current: External judge (Claude)
   - Alternative: Peer evaluation
   - **Need:** A/B test (Condition C vs E)

---

## V. SUCCESS METRICS (Realistic for Solo Dev)

**End of Phase 1 (2-4 weeks):**
- [x] Codebase rebranded
- [x] Trademark filed
- [ ] 5 documented benchmarks
- [ ] Clear answer: "When does Arbitrium help vs single model?"
- [ ] KB decision (keep/remove)

**v0.1.0 release:**
- [ ] Published on PyPI
- [ ] 3-5 GitHub stars
- [ ] 1-2 people try it and give feedback
- [ ] No major bugs

**v0.2.0 (if continuing):**
- [ ] 10+ stars
- [ ] 1-2 external contributors
- [ ] One real use case documented
- [ ] Community starting

---

## VI. REPOSITORY STATUS

**Links:**
- GitHub: https://github.com/arbitrium-framework/arbitrium
- PyPI: `arbitrium-framework` (not yet published)
- Docs: https://arbitrium-framework.github.io/arbitrium (to be set up)
- License: MIT
- Python: 3.10+

**Key Files:**
- `src/arbitrium/` - Core codebase
- `benchmarks/` - Validation scripts
- `tests/` - Unit & integration tests
- `docs/calculator.html` - ROI calculator
- `TRADEMARKS.md` - Trademark policy
- `NOTICE` - Trademark notice

**Useful Commands:**
```bash
# Install dev
pip install -e .[dev]

# Test
pytest tests/ -v

# Benchmark
python -m arbitrium.benchmarks.micro_benchmark --config config.yml

# Format
black src/ tests/ benchmarks/
ruff check .

# Build
python -m build
```

---

## VII. PARKING LOT (Ideas for Future)

- Async tournament execution for speed
- Export "provenance graph" of idea evolution
- Research: Optimal elimination strategy
- Web UI with real-time visualization
- Marketing: "LLM ensemble, but competitive not cooperative"
- Consider: Domain-specific judge models
- Advanced KB: Semantic clustering, insight quality scoring

---

## VIII. WHAT WAS ALREADY DONE (Archive)

✅ **Completed:**
- Rebranding from AgentSphere → Arbitrium Framework
- Git remote updated to arbitrium-framework/arbitrium
- All imports changed (agentsphere → arbitrium)
- All logger names updated
- pyproject.toml package name: arbitrium-framework
- CLI command: `arbitrium`
- All URLs updated to new repo
- README badges updated with CI badge
- .gitignore updated (logs, benchmarks, temp files)
- TRADEMARKS.md created (trademark policy)
- NOTICE created (trademark notice)
- Disclaimer added to README (personal project, no affiliation)
- Exception classes fixed (ArbitriumError)
- All benchmark function names updated
- All documentation URLs updated
- GitHub organization created: arbitrium-framework
- EUIPO trademark application submitted and paid

---

**Next milestone:** Complete 5 benchmark runs and make KB decision
**Next review:** 2025-01-15
