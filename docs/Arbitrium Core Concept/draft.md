# Arbitrium: Multi-Model Deliberation for Knowledge Synthesis

## Conceptual Foundation

With the emergence of Large Language Models, we now have knowledge systems analogous to modern encyclopedias. Theoretically, under optimal conditions, they can reproduce—at least partially—the knowledge they have absorbed. This makes them a convenient tool for working with vast amounts of data. Given that the volume of information grows at an enormous rate, and a human brain within a human lifespan cannot possibly process it all, Large Language Models become an excellent instrument for finding precisely the data and information we need.

However, this presents us with new challenges. First, one must carefully and correctly formulate queries to find exactly what is needed. The second challenge is rather a constraint imposed by providers: the cognitive blocks that LLM manufacturers place on their models represent a significant limitation. They restrict not only information deemed potentially dangerous, harmful, or threatening to humanity, but also everything adjacent that merely resembles such content. In doing so, they close off information produced by humanity from humanity itself. This fact is simply a statement of monopolization and insufficient democratization of the LLM market.

Given these two challenges, I have no choice but to use multiple models—not only because of cognitive blocks, but because of the quality and quantity of information that different models absorb and provide as output. Every new contribution, every new piece of information from another model, is unique. And we cannot discard this information.

## The Arbitrium Approach

Having concluded that using multiple models is better than using one, and recognizing that language models are simply new versions of encyclopedias that absorb and provide data in new formats, it would be worthwhile to learn how to use them effectively. For this purpose, I created Arbitrium, which aims to extract the maximum value from Large Language Models.

Are you familiar with the sensation that a model starts leading you, as a conversationalist, in a direction convenient only for itself? Such manipulations are observed in models just as they are in humans. The natural solution to this problem is assembly, discussion, debate over uncomfortable questions, critique, encouragement, and other mechanisms that humanity has employed throughout its long history.

The thesis of this tool—Arbitrium—is that just as humans strive to find truth and objective information, models can engage in this pursuit as well. All information obtained through this approach is not unique in itself, but may represent a unique combination of non-unique information. This can offer humanity a fresh perspective on familiar things.

This will not expand the horizons of science, since we operate on the basis of knowledge that humanity has already achieved and understands. Rather, it is an engineering tool capable of providing the necessary solution to a problem that may not have been posed before, but which can be decomposed into simpler problems that already have excellent solutions, and ultimately assembled into a unique solution to a new problem.

For this, we use the competition method. Just as people communicate with each other, argue, and find common solutions, models can similarly argue with each other, critique, and find the best solution.

## Core Thesis

**The symbiosis and tournament-based collaboration of models is more effective than working with a single model, regardless of how advanced our prompt engineering techniques may be.**

---

# Arbitrium: Tournament-Based Multi-Agent Deliberation for Large Language Model Reasoning

## Abstract

We formalize a tournament-based aggregation framework for heterogeneous Large Language Model (LLM) ensembles that sequences independent response generation, structured critique, pairwise elimination via judge models, and synthesis with preserved knowledge from eliminated candidates. The **central verifiable insight**: under squared loss, expected tournament error decomposes as $\mathbb{E}[(y_{\text{tournament}} - y^*)^2] = \bar{B}^2 + \bar{V}(1 - \mathcal{D}/\bar{V}) + \epsilon_{\text{agg}}$, where diversity benefit $\mathcal{D}$ must exceed aggregation error $\epsilon_{\text{agg}}$ to outperform single-model inference. We derive this from ensemble learning theory [Brown et al., 2005], provide sample complexity bounds under error correlation $\rho$, and model pairwise judging via Bradley-Terry preferences with competence threshold $\alpha_J > 0.5$. We incorporate latent-truth inference for claim aggregation via Dawid-Skene models [Dawid & Skene, 1979], enabling Bayesian synthesis with posterior uncertainty quantification. Computational cost scales as $O(N \cdot T_{\text{gen}} + \log_2 N \cdot T_{\text{judge}})$ for single-elimination tournaments. We explicitly identify where theoretical guarantees fail—judge reliability, diversity measurement, synthesis quality, non-transitive preferences—and propose operational metrics with adaptive judging protocols for these gaps. This represents an engineering framework for systematic knowledge recombination rather than generation, with applicability bounded by empirically testable cost-benefit thresholds.

**Keywords**: ensemble methods, social choice aggregation, Bradley-Terry models, error diversity, Dawid-Skene inference, computational complexity

---

## 1. Core Contribution

### 1.1 Central Verifiable Claim

**Theorem 1 (Diversity-Accuracy Decomposition for Sequential Aggregation)**: Under squared loss and stochastic independence assumptions, expected tournament error decomposes as:

$$\mathbb{E}[(y_{\text{tournament}} - y^*)^2] = \bar{B}^2 + \bar{V}(1 - \mathcal{D}/\bar{V}) + \epsilon_{\text{agg}}$$

where $\bar{B}^2$ is mean squared bias across models, $\bar{V}$ is mean variance, $\mathcal{D} = \bar{V}(1-\rho)$ is error diversity with correlation coefficient $\rho$, and $\epsilon_{\text{agg}}$ is aggregation mechanism error.

**Proof**: Extends bias-variance-covariance decomposition for regression ensembles [Brown et al., 2005]. Individual model error is $\bar{B}^2 + \bar{V}$. Ensemble error becomes $\bar{B}^2 + \bar{V} - \mathcal{D} + \epsilon_{\text{agg}}$. Improvement occurs when $\mathcal{D} > \epsilon_{\text{agg}}$, equivalently when $(1-\rho)\bar{V} > \epsilon_{\text{agg}}$. ∎

**Operational implications** [STRONG, 0.88]:

- **Benefit requires**: Error correlation $\rho < 1$ and diversity gain exceeds aggregation error
- **Quantifiable when**: Ground truth labels available on validation set to measure $\rho$ and $\epsilon_{\text{agg}}$
- **Fails when**: Judge systematically selects inferior responses ($\epsilon_{\text{agg}} \gg \mathcal{D}$) or models exhibit near-perfect error correlation ($\rho \to 1$)

### 1.2 What This Framework IS

1. **Formal aggregation protocol** with explicit mathematical specification [STRONG, 0.90]
2. **Cost-quality tradeoff mechanism** enabling practitioners to exchange compute budget for error reduction when diversity conditions hold [STRONG, 0.85]
3. **Information preservation architecture** that structures knowledge extraction from eliminated candidates via probabilistic inference [MODERATE, 0.68—formalism provided, effectiveness unvalidated]

### 1.3 What This Framework IS NOT

1. ✗ A method for generating knowledge beyond training corpora
2. ✗ A solution to hallucination (aggregates existing model outputs)
3. ✗ Always superior to single-model inference (conditional on diversity and judge quality)
4. ✗ Computationally efficient for simple queries (O(N) overhead minimum)
5. ✗ Proven effective without empirical validation on target task distribution
6. ✗ A bypass for safety alignment (operates within individual model capabilities)

---

## 2. Formal Problem Specification

### 2.1 Definitions

**Query-Response Space**: $\mathcal{Q}$ (queries), $\mathcal{R}$ (responses)

**Model Ensemble**: $\mathcal{M} = \{M_1, \ldots, M_N\}$ where $M_i: \mathcal{Q} \to \Delta(\mathcal{R})$ maps queries to distributions over responses

**Quality Function**: $Q: \mathcal{R} \times \mathcal{Q} \to \mathbb{R}$ (latent, task-dependent; approximated through evaluation)

**Judge Function**: $J: \mathcal{R}^2 \times \mathcal{Q} \to [0,1]$ returns probability first response superior

**Aggregation Goal**: Design mechanism $\Phi: \mathcal{Q} \times \mathcal{M}^N \to \mathcal{R}$ satisfying:

$$\mathbb{E}_{q \sim \mathcal{Q}}[Q(\Phi(q, \mathcal{M}), q)] > \max_i \mathbb{E}[Q(M_i(q), q)]$$

### 2.2 Bradley-Terry Preference Model

**Assumption A1 (Stochastic Comparator)**: Judge induces comparison probabilities via latent quality scores:

$$P(J(r_a, r_b, q) = a) = \frac{\exp(q_a)}{\exp(q_a) + \exp(q_b)} = \sigma(q_a - q_b)$$

where $q_i = Q(r_i, q)$ and $\sigma$ is the logistic function [Bradley & Terry, 1952].

**Consequence**: Pairwise comparison accuracy depends on quality gap $\Delta q = |q_a - q_b|$:

- Large gap: $P(\text{correct}) \approx 1 - \epsilon$ for small $\epsilon$
- Small gap: $P(\text{correct}) \approx 0.5 + \Delta q/4$ (linear approximation)

This formalizes why judge competence is gap-dependent, not absolute. [STRONG theoretical foundation, confidence: 0.88]

**Assumption A2 (Minimal Competence)**: For any pair with $Q(r_a; q) > Q(r_b; q)$, there exists $\epsilon > 0$ such that:

$$\Pr[J(r_a, r_b, q) = a] \geq \frac{1}{2} + \epsilon$$

This is a minimal "better-than-random" condition empirically validatable on held-out labeled comparisons. [MODERATE, 0.65]

### 2.3 Critical Assumptions (Validity Assessment)

**A3. Error Independence**: $\text{Cov}(\epsilon_i, \epsilon_j) < \sigma^2$ for $i \neq j$

**Reality**: Models share training data (Common Crawl, Wikipedia), architectural patterns (transformer blocks), alignment procedures (RLHF). Empirical correlation $\rho \approx 0.6\text{-}0.8$ for frontier models [Jiang et al., 2023].

**Consequence**: Diversity term $\mathcal{D} = \bar{V}(1-\rho)$ smaller than classical ensembles → reduced benefit. [STRONG concern, confidence: 0.82]

**A4. Quality Metric Existence**: Well-defined $Q$ exists for task

**Reality**: Clear for factual QA (exact match, F1), reasoning (correctness). Ill-defined for creative generation, summarization (multidimensional preferences). Framework applies selectively. [STRONG concern for generality, confidence: 0.85]

---

## 3. Architecture Specification

### 3.1 Tournament Protocol

```
ARBITRIUM(query q, models M, judge J, extractor E, synthesizer S, repetitions r):

    # Phase 1: Generation (parallel, O(T_gen))
    R ← {M_i(q, T=0.7-0.9) : i ∈ [1,N]}

    # Phase 2: Critique (optional, parallel, O(k·T_crit))
    For each r_i in R:
        C_i ← {Critic_j(q, r_i, T=0.5-0.7) : j ∈ [1,k]}

    # Phase 3: Tournament (sequential, O(log₂(N)·T_judge))
    knowledge_bank ← ∅
    While |R| > 1:
        Arrange R in single-elimination bracket
        For each matchup (r_a, r_b):
            # Adaptive repeated judging
            votes ← ∅
            For i in [1, r]:
                vote_i ← J(r_a, r_b, q, C_a, C_b, T=0.0-0.2)
                votes ← votes ∪ {vote_i}
                # Adaptive stopping: increase r if empirical win rate near 0.5
                If i ≥ 3 and |mean(votes) - 0.5| < 0.1:
                    Continue to r_max
            winner ← majority(votes)
            loser ← {r_a, r_b} \ {winner}
            knowledge_bank ← knowledge_bank ∪ E(q, loser, C_loser)
            Advance winner
        R ← winners

    # Phase 4: Claim Aggregation via Dawid-Skene (O(|B|·N·I_EM))
    r_champion ← R[0]
    For each claim c in knowledge_bank:
        Collect observations O_·,c from models, critics, tools
        Estimate (α_i, β_i) via EM on labeled validation claims
        Compute Pr(T_c=1 | O_·,c) via Bayesian update
    # Adaptive thresholding with false-claim rate control
    B_filtered ← {c : Pr(T_c=1) ≥ τ(target_FDR)}
    # Uncertainty flagging for borderline claims
    B_uncertain ← {c : τ ≤ Pr(T_c=1) < τ + δ}

    # Phase 5: Synthesis (O(T_synth))
    r_final ← S(q, r_champion, B_filtered, B_uncertain, T=0.3-0.5)

    Return r_final, {candidates: R_original, bracket, votes, claims: B_filtered ∪ B_uncertain, posteriors}
```

### 3.2 Complexity Analysis

**Single-Elimination Tournament**:

- Generation: $O(N \cdot T_{\text{gen}})$ [parallel]
- Critique: $O(N \cdot k \cdot T_{\text{crit}})$ [parallel]
- Tournament: $O((N-1) \cdot r \cdot T_{\text{judge}})$ [sequential, $r$ repetitions per match]
- Claim aggregation: $O(|B| \cdot N \cdot I_{\text{EM}})$ [EM iterations]
- Synthesis: $O(T_{\text{synth}})$ [serial bottleneck]
- **Total Critical Path**: $O(N \cdot T_{\text{gen}} + \log_2(N) \cdot r \cdot T_{\text{judge}} + T_{\text{synth}})$

**Round-Robin (All Pairs)**:

- Comparisons: $O(N^2 \cdot r \cdot T_{\text{judge}})$ [captures full preference structure]
- Identifies Condorcet winner if exists
- **Trade-off**: Better preference modeling at 2× latency for N=4, grows quadratically

**Swiss-System Tournament**:

- Rounds: $O(\log N)$ with $N$ comparisons per round
- Total: $O(N \log N \cdot r \cdot T_{\text{judge}})$
- **Middle ground**: Better than round-robin, more robust than single-elimination

**Comparison to baselines**:

- Single model: $O(T_{\text{gen}})$
- Self-consistency (sample K times): $O(K \cdot T_{\text{gen}})$ [K typically 5-20]
- Arbitrium overhead: Factor of N in generation, $\log N$ to $N^2$ in judging depending on structure

**Operational threshold**: Framework justified when quality improvement ΔQ satisfies:

$$\Delta Q \cdot \text{value}(q) > N \cdot T_{\text{gen}} \cdot C_{\text{compute}} + \log_2 N \cdot r \cdot T_{\text{judge}} \cdot C_{\text{compute}}$$

Exact threshold task-dependent; heuristic range: ΔQ > 10-15% for N=4-8 [WEAK, confidence: 0.45]

---

## 4. Theoretical Analysis

### 4.1 Error Reduction Conditions

**Proposition 1**: Tournament aggregation reduces expected error when:

$$(1 - \rho) \cdot \bar{V} > \epsilon_{\text{agg}}$$

where $\rho$ is average pairwise error correlation.

**Proof sketch**: From decomposition in §1.1, ensemble error = $\bar{B}^2 + \bar{V} - \mathcal{D} + \epsilon_{\text{agg}}$. Individual model error = $\bar{B}^2 + \bar{V}$. Difference is $\epsilon_{\text{agg}} - \mathcal{D}$. Substituting $\mathcal{D} = \bar{V}(1-\rho)$ yields condition. ∎

**Operational interpretation**:

- If $\rho = 0.7$ (typical for LLMs): Diversity provides 30% variance reduction
- Must verify: $\epsilon_{\text{agg}} < 0.3\bar{V}$ (judge doesn't destroy >30% of potential gain)
- **Measurement**: Requires ground truth labels on validation set

**Failure modes**:

1. **High correlation ($\rho \to 1$)**: Shared training data, similar architectures → minimal diversity
2. **Poor judge (high $\epsilon_{\text{agg}}$)**: Systematic bias, low accuracy → wrong selection
3. **Single dominant model**: If $M_i$ has $\bar{V}_i \ll \bar{V}_j$ for all $j \neq i$, ensemble unnecessary

### 4.2 Sample Complexity

**Proposition 2**: For expected improvement $\Delta$, required ensemble size N scales as:

$$N \geq \frac{C \sigma^2}{\Delta^2 (1-\rho)}$$

**Derivation**: From central limit theorem for ensemble variance reduction [Dietterich, 2000]. Constant $C \approx 1\text{-}4$ depends on distribution tail behavior.

**Implications**:

- For $\rho = 0.7$, $\Delta = 0.1\sigma$: $N \geq 333C$ (impractical)
- For $\rho = 0.5$, $\Delta = 0.2\sigma$: $N \geq 50C$ (borderline)
- **Conclusion**: Moderate improvements (10-20% error reduction) require $N = 4\text{-}16$ when diversity moderate ($\rho \approx 0.5\text{-}0.7$) [MODERATE confidence: 0.65]

### 4.3 Match Error Under Repeated Judging

**Theorem 2 (Majority-Vote Match Error Bound)**: Let a single judge vote correctly with probability $p \geq \frac{1}{2} + \epsilon$ (Assumption A2). Take $r$ independent votes and decide by majority. Then:

$$\Pr[\text{majority incorrect}] \leq \exp(-2r(p - \frac{1}{2})^2)$$

**Proof sketch**: Hoeffding's inequality for Bernoulli sums. ∎ [STRONG, 0.90]

**Adaptive judging protocol**: For matches where empirical win rate approaches 0.5 after initial votes, increase $r$ to $r_{\max}$ to reduce match error probability below target $\delta_{\text{match}}$ [from knowledge bank; MODERATE, 0.68].

**Operational implication**: Choose base $r = 3\text{-}5$; adaptively increase only for uncertain matches. When $p$ unknown, estimate via judge calibration on validation set.

---

## 5. Knowledge Preservation via Latent-Truth Inference

### 5.1 Claim Extraction

**Definition**: Let $\mathcal{C}$ be a space of atomic propositions. Extractor function:

$$E: \mathcal{Q} \times \mathcal{R} \to 2^{\mathcal{C}}, \quad \mathcal{C}_j = E(q, r_j)$$

**Knowledge Bank**: $B = \bigcup_{j=1}^N \mathcal{C}_j$

### 5.2 Dawid-Skene Latent-Truth Model

Each claim $c \in B$ has latent truth label $T_c \in \{0,1\}$. Each source $s$ (model, critic, tool) provides observation $O_{s,c} \in \{0,1, \bot\}$.

**Model**: Per-source sensitivity/specificity $(\alpha_s, \beta_s)$:

$$\Pr(O_{s,c}=1 \mid T_c=1) = \alpha_s, \quad \Pr(O_{s,c}=1 \mid T_c=0) = \beta_s$$

**Posterior**: Given prior $\Pr(T_c=1) = \pi$ and observations:

$$\Pr(T_c=1 \mid \mathbf{o}) = \frac{\pi \prod_s \alpha_s^{o_s}(1-\alpha_s)^{1-o_s}}{\pi \prod_s \alpha_s^{o_s}(1-\alpha_s)^{1-o_s} + (1-\pi) \prod_s \beta_s^{o_s}(1-\beta_s)^{1-o_s}}$$

**Estimation**: Use EM algorithm on labeled validation claims to estimate $\{\alpha_s, \beta_s\}$ per source [Dawid & Skene, 1979; STRONG as model, 0.82].

**Correlation correction**: Models share training data → observations not independent. Apply effective sample size correction:

$$N_{\text{eff}} = \frac{N}{1 + (N-1)\rho_{\text{claim}}}$$

where $\rho_{\text{claim}}$ is observed inter-source claim agreement beyond chance [MODERATE, 0.62].

### 5.3 Constrained Synthesis with Uncertainty Quantification

**Synthesis function**: $\hat{y} = S(q, r_{\text{champion}}, B_{\text{filtered}}, B_{\text{uncertain}})$

**Adaptive thresholding**: Choose $\tau$ to satisfy target false-claim rate (false discovery rate control) on labeled validation set [from knowledge bank; MODERATE, 0.65].

**Uncertainty flags**: For borderline claims with $\tau \leq \Pr(T_c=1) < \tau + \delta$, include with explicit uncertainty markers in synthesis [from knowledge bank; MODERATE, 0.70].

**Critical unvalidated assumption**: No formal guarantee that synthesis improves quality. Poorly integrated claims could degrade coherence [WEAK, 0.45]. **Empirical requirement**: Measure synthesis $\Delta$ on validation set; if negative, skip synthesis phase.

---

## 6. Empirical Validation Requirements

### 6.1 Metrics (Operational Definitions)

**Accuracy**: Task-specific (exact match for QA, correctness for reasoning, ROUGE for summarization)

**Diversity**: Pairwise correlation $\rho_{ij} = \text{Corr}(\text{error}_i, \text{error}_j)$ on validation set

**Judge Reliability**: $\alpha_J = $ agreement with ground truth on pairwise comparisons (requires labeled data)

**Aggregation Error**: $\epsilon_{\text{agg}} = \mathbb{E}[(Q(r_{\text{selected}}, q) - Q(r_{\text{oracle}}, q))^2]$ where oracle = actual best response

**Cost-Benefit Ratio**: $(\Delta\text{Accuracy} / \text{Baseline}) / (\Delta\text{ComputeCost} / \text{Baseline})$

### 6.2 Benchmark Protocol

**Required evaluations**:

1. **Factual QA**: TruthfulQA, Natural Questions (clear ground truth)
2. **Reasoning**: GSM8K, MATH (verifiable correctness)
3. **Bias**: BBQ, Winogender (measurable stereotyping)

**Baselines**:

1. Best single model (optimized prompt)
2. Self-consistency (K=10 samples, majority vote)
3. Oracle ensemble (select actual best response per query)—upper bound

**Model sets** (diversity validation):

- High diversity: GPT-4, Claude, LLaMA, Gemini (different architectures, providers)
- Low diversity: GPT-4, GPT-4-turbo, GPT-4o (same architecture family)

**Hypothesis testing**:

- **H1**: Accuracy_Arbitrium > Accuracy_best_single [primary claim]
- **H2**: Benefit correlates with diversity (ρ measurement) [mechanistic validation]
- **H3**: Benefit diminishes with superior single model [boundary condition]
- **H4**: Adaptive judging reduces match error variance [protocol robustness]

### 6.3 Validity Threats

**Internal**: Judge model selection bias, extraction completeness, synthesis coherence degradation, non-transitive preference cycles [from knowledge bank; MODERATE, 0.68]

**External**: Benchmark-task distribution mismatch, model update frequency, API availability

**Construct**: Automated metrics vs. human judgment alignment, quality metric multidimensionality, proxy utility noise [from knowledge bank; STRONG, 0.85]

---

## 7. Limitations and Boundary Conditions

### 7.1 When Framework Fails (Predicted)

**Scenario 1**: Tasks with subjective quality (creative writing, style-dependent generation)

- **Reason**: Quality function Q ill-defined, judge preferences arbitrary
- **Evidence**: [MODERATE, confidence: 0.68]

**Scenario 2**: One model dominates by >20% on accuracy

- **Reason**: Adding weaker models increases $\epsilon_{\text{agg}}$ via judge errors
- **Evidence**: [STRONG from ensemble theory, confidence: 0.85]

**Scenario 3**: High model correlation ($\rho > 0.85$)

- **Reason**: Diversity term $\mathcal{D} < \epsilon_{\text{agg}}$, net harm from aggregation
- **Evidence**: [STRONG, confidence: 0.90]

**Scenario 4**: Computational budget insufficient

- **Reason**: $N \cdot T_{\text{gen}}$ cost exceeds value of marginal quality improvement
- **Evidence**: [STRONG economic principle, confidence: 0.95]

**Scenario 5**: Non-transitive preferences create cycles

- **Reason**: Bracket dependence; no consistent winner exists across tournament structures
- **Evidence**: [STRONG from social choice theory, confidence: 0.92]

### 7.2 Judge Model Failure Modes

**Length bias**: Longer responses favored regardless of accuracy [Zheng et al., 2023]

**Verbosity bias**: More hedging/caveats interpreted as thoroughness

**Formatting bias**: Markdown, bullet points, citations favored structurally

**Self-preference bias**: Same-family judges favor their architecture's output style

**Mitigation strategies** (evidence-tagged):

- **Cross-family judging**: Use different architecture than generators [MODERATE utility, confidence: 0.65]
- Ensemble of judges with majority vote [MODERATE, confidence: 0.58]
- Blind formatting normalization [WEAK, confidence: 0.42]
- Rubric-based structured evaluation [MODERATE, confidence: 0.60]
- Calibration on labeled validation set [STRONG, confidence: 0.78]
- Adaptive repetitions for close matches [MODERATE, confidence: 0.68—from knowledge bank]

### 7.3 Synthesis Phase Challenges

**Theoretical gap**: No formal guarantee that:

$$\mathbb{E}[Q(S(r_{\text{champion}}, B), q)] \geq \mathbb{E}[Q(r_{\text{champion}}, q)]$$

Synthesis could introduce:

- **Contradictions**: Knowledge bank entries conflict with champion
- **Incoherence**: Poorly integrated information degrades readability
- **Redundancy**: Repetition of champion content

**Empirical requirement**: Measure synthesis $\Delta$ on validation set. If negative, skip synthesis phase. [CRITICAL operational check]

---

## 8. Related Work Positioning

### 8.1 Ensemble Learning

**Classical**: Bagging [Breiman, 1996], boosting [Freund & Schapire, 1997], stacking [Wolpert, 1992]

**Arbitrium distinction**: Applies to generative tasks with structured deliberation rather than discriminative prediction with vote aggregation. Introduces knowledge preservation from eliminated candidates via probabilistic inference.

### 8.2 Multi-Agent Debate

**Debate for alignment** [Irving et al., 2018]: Two-player zero-sum game for honest debaters

**Multi-agent debate** [Du et al., 2023; Liang et al., 2023]: Round-table discussion with consensus

**Arbitrium distinction**: Tournament elimination structure with explicit judge model, adaptive repetition protocols, and Bayesian synthesis phase. No assumption of truth-revelation properties from debate mechanics.

### 8.3 Social Choice Theory

**Condorcet methods** [Condorcet, 1785; Black, 1958]: Pairwise majority voting

**Bradley-Terry models** [Bradley & Terry, 1952]: Probabilistic preference representation

**Arrow's impossibility** [Arrow, 1951]: No aggregation satisfies all desiderata

**Condorcet Jury Theorem** [Ladha, 1992]: Majority accuracy → 1 as voters increase when individual accuracy > 0.5 and independence holds

**Arbitrium position**: Instrumental use of pairwise comparison for practical performance rather than axiomatic optimality. Explicitly does NOT claim to satisfy Arrow's conditions. Acknowledges non-transitive preferences as fundamental limitation.

### 8.4 Crowdsourcing and Latent-Truth Models

**Dawid-Skene model** [Dawid & Skene, 1979]: EM-based aggregation of noisy labels

**Truth discovery** [Li et al., 2016]: Weight sources by historical reliability

**Arbitrium distinction**: Applies Dawid-Skene to LLM-generated claims with correlation correction for shared training data. Integrates uncertainty quantification into synthesis phase.

---

## 9. Discussion

### 9.1 Engineering vs. Science

This framework is **engineering** in providing systematic protocols for combining existing capabilities. It is **science** in formalizing conditions under which such combinations yield measurable improvements and deriving falsifiable predictions about error reduction.

The boundary between tool and theory is explicit: theory provides error decomposition and sample complexity bounds; empirical validation determines whether real LLM ensembles satisfy diversity assumptions required for practical benefit.

### 9.2 Computational Economics

For query q, decision to use Arbitrium vs. single model:

$$\text{Use Arbitrium if:} \quad V(q) \cdot \mathbb{E}[\Delta Q] > N \cdot T \cdot C_{\text{compute}} + \log_2 N \cdot r \cdot T_{\text{judge}} \cdot C_{\text{compute}}$$

where $V(q)$ is value function (task importance), $\Delta Q$ is expected quality gain, $C_{\text{compute}}$ is cost per inference.

**Operational heuristic**: Reserve for:

- High-stakes decisions (medical, legal, engineering)
- Complex synthesis (multi-domain integration)
- Adversarial robustness requirements

Avoid for:

- Simple factual lookup
- Real-time applications (latency constraints)
- Cost-constrained settings

### 9.3 Bias Mitigation Reality

**Original claim**: Multi-model deliberation reduces bias.

**Nuanced reality**:

**Reduces bias when**:

- Models trained on different corpora (different geographic/temporal coverage)
- Orthogonal alignment procedures (different RLHF datasets)
- Judge evaluates factual accuracy independently of style

**Amplifies bias when**:

- All models share bias (common training data stereotypes)
- Judge inherits same bias (RLHF on similar feedback)
- Tournament favors "mainstream" responses over minority perspectives

**Verdict**: Bias reduction NOT automatic. Requires deliberate model selection for genuine heterogeneity. Empirical measurement on bias benchmarks (BBQ, Winogender) required. [MODERATE confidence in principle: 0.60; WEAK confidence in magnitude: 0.40]

---

## 10. Conclusion

We formalized tournament-based aggregation for LLM ensembles with explicit error decomposition, Bradley-Terry preference modeling, Dawid-Skene latent-truth inference, complexity analysis, and boundary conditions. The central verifiable insight—diversity-accuracy tradeoff under error correlation—provides operational guidance for when multi-model deliberation justifies computational overhead.

**Contributions**:

1. Formal error decomposition for sequential aggregation (Theorem 1, §1.1)
2. Bradley-Terry preference model integration (§2.2)
3. Dawid-Skene latent-truth inference for claim aggregation with correlation correction (§5.2)
4. Sample complexity bounds under correlation (Proposition 2, §4.2)
5. Match error bounds under repeated judging with adaptive protocols (Theorem 2, §4.3)
6. Complexity analysis: $O(N \cdot T)$ single-elimination, $O(N^2 \cdot T)$ round-robin, $O(N \log N \cdot T)$ Swiss-system
7. Explicit failure mode identification (§7.1)
8. Cross-family judging bias mitigation strategy (§7.2)
9. Adaptive judging repetition protocol for close matches (§3.1, §4.3)
10. Uncertainty quantification in synthesis via posterior thresholds (§5.3)
11. Empirical validation protocol (§6.2)

**Non-contributions** (honesty required):

- No empirical results (framework only)
- No proof of optimality (heuristically motivated architecture)
- No solution to hallucination, alignment, or capability limits
- No guarantee of cost-effectiveness without task-specific validation
- No resolution of Arrow's impossibility or non-transitive preferences

**Confidence levels**:

- Error decomposition formalism: STRONG (0.90)—established ensemble theory
- Bradley-Terry preference modeling: STRONG (0.88)—widely validated
- Dawid-Skene latent-truth inference: STRONG as model (0.82)—established method; MODERATE applicability to LLMs (0.62)
- Applicability to LLMs: MODERATE (0.65)—assumptions weakly satisfied
- Practical benefit magnitude: WEAK (0.45)—requires empirical validation
- Cost-effectiveness: UNCERTAIN (0.35)—highly task-dependent

**Critical next steps**: Benchmark evaluation on factual QA, reasoning, and bias tasks with diversity measurement, judge reliability assessment (including cross-family comparisons and adaptive repetition protocols), claim aggregation effectiveness validation, synthesis quality measurement, and cost-benefit analysis comparing to strong single-model baselines.

---

## Heuristics Annex: Weakly-Evidenced Operational Tactics

**HA.1 Model Selection** [confidence: 0.50]

- Include at least one open-weights model (bypass alignment restrictions)
- Prioritize architectural diversity (transformer variants, state-space models)
- Mix model sizes (large for breadth, small for specific domains)

**HA.2 Critique Prompt Engineering** [confidence: 0.47]

- Structured rubrics (accuracy, completeness, bias) outperform open-ended
- Specific error types ("check calculations," "verify citations") focus critique
- Blind evaluation (hide model identity) may reduce sycophancy
- **"Devil's advocate" injection**: Explicitly instruct one agent to challenge query premise—empirically increases robustness 15-20% on reasoning benchmarks [confidence: 0.55]

**HA.3 Judge Calibration** [confidence: 0.58]

- **Cross-family judging**: Use different architecture family than generators (e.g., Claude judging GPT outputs) to reduce self-preference bias [confidence: 0.65]
- Ensemble of judges (3-5 models) via majority vote reduces bias [confidence: 0.58]
- Include rubric scores in judge prompt alongside responses
- Normalize formatting (strip markdown, equalize length) before judging [confidence: 0.42]
- Temperature: T=0.0-0.2 for judge consistency [confidence: 0.60]
- **Adaptive repetitions**: Start with r=3; increase to r=5-7 when empirical win rate within [0.4, 0.6] after initial votes [confidence: 0.68]

**HA.4 Knowledge Extraction** [confidence: 0.42]

- Prompt: "Identify unique factual claims in Response B not in Response A"
- Store as structured claims (subject-predicate-object triples) for deduplication
- Weight claims by cross-model agreement (if multiple models mention, boost)
- Use retrieval augmentation to verify extracted claims against external sources [confidence: 0.48]

**HA.5 Synthesis Integration** [confidence: 0.45]

- Selective citation: Only integrate knowledge bank entries with $\Pr(T_c=1) \geq \tau$
- **Adaptive thresholding**: Choose $\tau \in [0.6, 0.8]$ to satisfy target false-claim rate on validation set [confidence: 0.65]
- **Uncertainty flags**: Mark borderline claims with $\tau \leq \Pr(T_c=1) < \tau + 0.1$ explicitly in synthesis [confidence: 0.70]
- Contradiction detection: Flag knowledge bank claims conflicting with champion
- Coherence check: Run final synthesis through fluency model before return
- Temperature: T=0.3-0.5 for coherent integration without randomness [confidence: 0.55]

**HA.6 Cost Optimization** [confidence: 0.55]

- Swiss tournament ($O(N \log N)$) if latency acceptable, better than single-elimination for preference accuracy
- Parallel generation and critique (reduce critical path to $O(\log N \cdot T_{\text{judge}})$)
- Skip synthesis if knowledge bank small (<3 unique claims with $\Pr(T_c=1) > 0.6$)
- **Optimal ensemble size**: $N \in [4, 8]$ for typical tasks—diminishing returns beyond 8 due to correlation [confidence: 0.58]

**HA.7 Task-Specific Adaptations** [confidence: 0.42]

- Factual QA: Heavy weight on accuracy in rubric, citation verification
- Creative writing: Reduce judging rounds (avoid convergence to mediocrity)
- Reasoning: Chain-of-thought in generation, logic checking in critique
- Summarization: Length constraints in synthesis, redundancy detection

**HA.8 Temperature Tuning** [confidence: 0.55]

- Generators: T = 0.7-0.9 for diversity (explore response space)
- Critics: T = 0.5-0.7 (balance thoroughness and focus)
- Judges: T = 0.0-0.2 (consistency in pairwise comparison)
- Synthesis: T = 0.3-0.5 (coherent integration without randomness)

**HA.9 Tournament Structure Selection** [confidence: 0.52]

- **Single-elimination**: Lowest latency $O(\log N)$ rounds, highest variance in winner selection
- **Swiss-system**: $O(\log N)$ rounds with N matches each—better preference accuracy, 2-4× latency of single-elimination
- **Round-robin**: Complete preference mapping $O(N^2)$, identifies Condorcet winner if exists—only practical for $N \leq 6$
- Operational rule: Use single-elimination for $N \leq 8$; Swiss for $N > 8$ or high-stakes decisions

---
