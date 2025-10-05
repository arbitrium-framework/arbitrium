# Arbitrium Framework: Tournament-Based AI Decision Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/arbitrium-framework/arbitrium/actions/workflows/ci.yml/badge.svg)](https://github.com/arbitrium-framework/arbitrium/actions)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

> **Name Note:** Arbitrium Framework is not related to Arbitrum (Ethereum L2) or Arbitrium RAT. This is an open-source LLM tournament framework.

**A specialized framework for high-stakes decisions where quality, auditability, and synthesis of diverse perspectives matter more than speed.**

---

<p align="center">
  <a href="https://colab.research.google.com/github/arbitrium-framework/arbitrium/blob/main/examples/interactive_demo.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/Try%20Now-Interactive%20Demo-blue?style=for-the-badge&logo=google-colab" alt="Try Interactive Demo" height="50"/>
  </a>
  <br/>
  <sub><b>Run real tournaments in your browser • No installation • ~15 minutes • Requires API keys</b></sub>
</p>

---

## What Makes Arbitrium Framework Different?

Unlike general-purpose agent frameworks (AutoGen, CrewAI, LangGraph) designed for task automation and workflow orchestration, **Arbitrium Framework solves a specific problem: synthesizing a single, defensible, high-quality solution from competing AI perspectives.**

The framework uses a **competitive tournament structure** where AI agents:
1. Generate independent solutions to complex problems
2. Critique and learn from each other's approaches
3. Compete in evaluation rounds where weaker solutions are eliminated
4. **Preserve valuable insights from eliminated agents** via the Knowledge Bank
5. Progressively refine until a champion solution emerges

### The Knowledge Bank: Learning from Every Perspective

**This is Arbitrium Framework's core innovation.** When an agent is eliminated, the system:

- Extracts unique insights and valuable reasoning from its solution
- Vectorizes and deduplicates these insights using cosine similarity
- Injects preserved knowledge into surviving agents' context
- Tracks provenance: which ideas came from which eliminated agent

**Result:** The final solution combines the strongest overall approach with the best minority insights from across all competing perspectives. No valuable idea is lost.

## When Should You Use Arbitrium Framework?

### Ideal Use Cases

**Arbitrium Framework is purpose-built for high-stakes analytical decisions:**

- **Strategic Business Decisions** ($5K-$50K+ impact): Market entry analysis, investment theses, strategic planning
- **Multi-Stakeholder Synthesis**: Policy development, complex project planning where diverse viewpoints must be integrated
- **Compliance & Legal Analysis**: Scenarios requiring defensible audit trails and traceable reasoning
- **Risk-Sensitive Problem Solving**: Decisions where wrong answers have significant financial, operational, or reputational consequences
- **Research & Academic Analysis**: Complex problems requiring synthesis of competing theoretical frameworks

**→ [Calculate if Arbitrium Framework is right for your decision](https://arbitrium-framework.github.io/arbitrium/calculator.html) ←**

### Decision Stakes Guidance

| Decision Value   | Recommended Approach                  | Why                                              |
| ---------------- | ------------------------------------- | ------------------------------------------------ |
| < $500           | Single model (Claude/GPT-4)           | Tournament cost exceeds decision value           |
| $500 - $5,000    | Single model + careful prompting      | Cost-effective for most use cases                |
| $5,000 - $50,000 | **Arbitrium Framework recommended**           | Clear ROI, stakes justify thoroughness           |
| > $50,000        | **Arbitrium Framework + human expert review** | Highest stakes demand multiple validation layers |

### What Arbitrium Framework Is NOT For

**Don't use Arbitrium Framework for:**

- Routine queries or simple factual questions
- Task automation and workflow orchestration (use CrewAI, AutoGen)
- Code generation and tool use (use AutoGen, LangGraph)
- Time-sensitive decisions requiring answers in < 2 minutes
- Low-stakes decisions where a single model suffices

## Why Not Just Use Existing Frameworks?

### Arbitrium Framework vs. AutoGen & Microsoft Agent Framework

**AutoGen's Philosophy:** Flexible, conversation-driven multi-agent collaboration for task execution (code generation, tool use, automation).

**Arbitrium Framework's Philosophy:** Structured competitive synthesis for decision-making. Not designed for general tasks, but specialized for converging on the best possible answer to complex, ambiguous questions.

**Key Difference:** AutoGen agents collaborate conversationally. Arbitrium Framework agents compete in tournaments while preserving collective wisdom.

### Arbitrium Framework vs. CrewAI

**CrewAI's Philosophy:** Role-based team simulation for automating known business workflows (marketing, content creation, structured processes).

**Arbitrium Framework's Philosophy:** Model-agnostic competition where the best solution wins regardless of predefined roles. Used when the optimal approach is unknown.

**Key Difference:** CrewAI delegates tasks to specialists in a known workflow. Arbitrium Framework evaluates competing approaches to discover the best workflow.

### Arbitrium Framework vs. LangGraph

**LangGraph's Philosophy:** Low-level library for building custom, stateful, cyclical agent workflows with fine-grained control.

**Arbitrium Framework's Philosophy:** Opinionated, ready-to-use system for competitive decision synthesis. The tournament workflow is built-in.

**Key Difference:** LangGraph is a toolkit for building any custom agent system. Arbitrium Framework is a complete solution for one specific problem: high-stakes decision synthesis.

## Core Architecture: How It Works

Arbitrium Framework orchestrates a multi-phase tournament where competitive pressure drives quality while collaborative knowledge sharing prevents waste.

```mermaid
flowchart TD
    Start([Start Tournament])

    Start --> Phase1[Phase 1:<br/>Initial Answers]

    Phase1 --> M1[Model 1]
    Phase1 --> M2[Model 2]
    Phase1 --> M3[Model 3]
    Phase1 --> M4[Model 4]

    M1 --> Phase2[Phase 2:<br/>Improvement]
    M2 --> Phase2
    M3 --> Phase2
    M4 --> Phase2

    Phase2 -.Description.-> P2_Desc["`**Prompt-driven improvement**

Models exchange responses and provide feedback.
Behavior controlled via customizable prompts.
Knowledge Bank insights injected here.
`"]

    Phase2 --> M1_Imp[Model 1<br/>Improved]
    Phase2 --> M2_Imp[Model 2<br/>Improved]
    Phase2 --> M3_Imp[Model 3<br/>Improved]
    Phase2 --> M4_Imp[Model 4<br/>Improved]

    M1_Imp --> Phase3[Phase 3:<br/>Cross-Evaluation]
    M2_Imp --> Phase3
    M3_Imp --> Phase3
    M4_Imp --> Phase3

    Phase3 -.Description.-> P3_Desc["`**Evaluation & Elimination**

Judge or peer review scores responses.
Weakest model eliminated each round.
Leader extracts insights from eliminated model.
`"]

    Phase3 --> Elim[Eliminate<br/>Weakest]
    Elim --> KB[(Knowledge Bank)]
    KB --> Check{More than 1<br/>model remaining?}
    KB -."insights injected".-> Phase2

    Check -->|Yes| Phase2
    Check -->|No| Champion[Champion<br/>Declared]

    Champion --> End([End])
```

### Tournament Phases

**Phase 1: Initial Answers**
- All models independently answer the question
- No collaboration yet—pure diverse perspectives

**Phase 2: Improvement**
- Models exchange responses and improve their answers based on peer feedback
- Customizable prompts control improvement behavior (critique, praise, or both)
- Knowledge Bank insights (if enabled) are injected here

**Phase 3: Cross-Evaluation**
- Judge model or peer review scores all responses
- Identifies leader (highest score) and weakest model
- Weakest model is eliminated

**After each elimination:**
- Leader extracts unique insights from eliminated model's response
- Insights are vectorized, deduplicated, and stored in Knowledge Bank
- Process repeats (refinement → evaluation → elimination) until one champion remains

**Result:** Champion's final answer combines the strongest approach with preserved insights from all eliminated perspectives.

## Key Features

### 1. No Insights Are Wasted
Unlike simple elimination systems, Arbitrium Framework's Knowledge Bank ensures that even "losing" agents contribute valuable perspectives to the final solution.

### 2. Complete Traceability
Every tournament generates provenance reports showing:
- Which model proposed each idea
- How ideas were critiqued and refined over rounds
- Why certain approaches were eliminated
- What insights were preserved from eliminated models

### 3. Configurable Tournament Dynamics
- **Improvement prompts**: customize how models critique, praise, or build on each other's responses
- **Evaluation methods**: single judge, peer review, or custom judge models
- **Knowledge Bank configuration**: choose which model extracts insights, set similarity thresholds for deduplication

### 4. Quality Monitoring
Built-in tools detect:
- **Consensus convergence**: when responses become too similar (potential groupthink)
- **Diversity metrics**: ensure the tournament maintains varied perspectives
- **Evaluation bias**: monitor judge consistency across rounds

## Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/arbitrium-framework/arbitrium.git
    cd arbitrium
    ```

2.  **Set up the environment** (virtual environment recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```sh
    pip install -e .
    ```

    For development (includes testing and linting tools):

    ```sh
    pip install -e .[dev]
    ```

4.  **Configure the application:**
    - Copy the example configuration file:
      ```sh
      cp config.example.yml config.yml
      ```
    - Open `config.yml` and configure your models. For most models, only `model_name` and `provider` are required:
      ```yaml
      models:
        gpt:
          provider: openai
          model_name: gpt-4-turbo
          # max_tokens and context_window auto-detected from LiteLLM!
      ```
    - **Auto-detection**: Arbitrium Framework automatically detects `max_tokens` and `context_window` from LiteLLM's database for supported models.
    - Add your API keys as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or configure 1Password paths in the secrets section.

## Quick Start

Get your first AI tournament running in minutes.

1.  **Define your question:**
    Open the config file and write the complex question or problem you want the agents to solve.

2.  **Run the tournament:**

    ```sh
    arbitrium
    ```

    Or:

    ```sh
    python -m arbitrium.runners.cli
    ```

3.  **View the results:**
    Watch the terminal for real-time updates. The final "Champion Solution" will be displayed in the terminal.

### Example Output

```
🔍 Starting Arbitrium Framework Tournament with 3 models: gpt-4-turbo, claude-3-opus, gemini-1.5-pro

Phase 1: Initial Answers
├── GPT-4: Generating response...
├── Claude Opus: Generating response...
└── Gemini Pro: Generating response...

Phase 2: Improvement
├── Models exchanging responses and feedback...
└── Generating improved answers...

Phase 3: Cross-Evaluation
├── Evaluating all responses...
└── Scores: GPT-4 (8.2), Claude (9.1), Gemini (7.8)

🏆 Round 1 Leader: Claude Opus
📤 Eliminated: Gemini Pro (insights preserved in Knowledge Bank)

🔄 Refinement Round
└── Remaining models refine answers with KB insights...

🎯 Final Champion: Claude Opus
```

## Configuration

### Auto-Detection of Model Parameters

Arbitrium Framework automatically detects `max_tokens` and `context_window` from LiteLLM's model database, making configuration easier:

**Minimal configuration (auto-detection):**

```yaml
models:
  gpt:
    provider: openai
    model_name: gpt-4-turbo
    # max_tokens: auto-detected (4096)
    # context_window: auto-detected (128000)
```

**Full configuration (manual override):**

```yaml
models:
  gpt:
    provider: openai
    model_name: gpt-4-turbo
    max_tokens: 8192 # Override default
    context_window: 200000 # Override default
    temperature: 0.9 # Optional, defaults to 0.7
```

**Required fields:**

- `model_name`: The model identifier (e.g., `gpt-4-turbo`, `claude-3-opus-20240229`)
- `provider`: The provider name (e.g., `openai`, `anthropic`, `vertex_ai`)

**Optional fields:**

- `max_tokens`: Maximum output tokens (auto-detected if omitted)
- `context_window`: Maximum input tokens (auto-detected if omitted)
- `temperature`: Sampling temperature (defaults to 0.7)
- `display_name`: Human-readable name (defaults to `model_name`)
- `reasoning_effort`: Extended thinking mode - `"low"`, `"medium"`, or `"high"` (supported by GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, Grok 4)

**Advanced features:**

```yaml
models:
  gpt:
    provider: openai
    model_name: gpt-5
    reasoning_effort: "high" # Enable extended thinking for complex problems
```

**Note:** For models not in LiteLLM's database (e.g., Grok), you must specify `max_tokens` and `context_window` manually.

## FAQ

### Why not just use a single model with careful prompting?

For most queries, you should! Arbitrium Framework is designed for high-stakes decisions ($5,000+) where the cost and time investment are justified. The tournament structure provides:

- **Multiple perspectives** that surface blind spots
- **Structured critique** that improves quality through adversarial collaboration
- **Progressive refinement** where each round builds on previous insights
- **Knowledge preservation** ensuring no valuable insight is lost
- **Provenance tracking** showing how ideas evolved

For routine queries, use Claude or GPT-4 directly.

### What about multi-agent debate systems?

Multi-agent debate systems (like those built in AutoGen or LangGraph) aim to explore a problem space through discussion. Arbitrium Framework differs fundamentally:

- **Debate systems**: All agents participate throughout, working toward consensus through dialogue
- **Arbitrium Framework**: Agents compete for survival through elimination, while their best ideas are preserved in the Knowledge Bank

The tournament creates genuine competitive pressure while the Knowledge Bank ensures collaborative benefit.

### What about rate limits?

Arbitrium Framework makes 12 parallel API calls during peak phases. This requires:

- **Anthropic**: Tier 3+ (200 RPM) recommended
- **OpenAI**: Tier 1+ (500 RPM) usually sufficient
- **Google Vertex AI**: Default quotas typically handle it

For lower-tier API plans, consider adjusting `max_concurrent_requests` in the configuration file or using local models (Ollama) which have no rate limits.

### How do I know models aren't just agreeing with each other?

Arbitrium Framework includes multiple safeguards:

- **Consensus monitoring** using TF-IDF vectorization and cosine similarity
- **Diversity metrics** that flag when responses become too similar
- **Multiple evaluation strategies** (peer review, external judge, multiple judges)
- **Independent initial responses** before any collaboration begins

### Can I use this for research?

Absolutely! Arbitrium Framework is designed for both production use and research. The framework provides:

- **Configurable tournament structures** for studying competitive-collaborative dynamics
- **Detailed logging** of all tournament phases
- **Monitoring utilities** for detecting bias and measuring consensus
- **Reproducible experiments** with deterministic mode
- **Provenance tracking** for analyzing how ideas evolve

We welcome research collaborations and contributions.

## Development

### Setting up for development

1. Install development dependencies:

   ```sh
   pip install -e .[dev]
   ```

2. Install pre-commit hooks:

   ```sh
   pre-commit install
   ```

3. Run code formatting:
   ```sh
   black src/
   isort src/
   ruff check src/
   ```

## Contributing

We welcome contributions! Arbitrium Framework fills a unique niche in the multi-agent AI ecosystem, and there are many opportunities for improvement:

- **New tournament structures**: Swiss-system, double-elimination, round-robin
- **Advanced Knowledge Bank features**: semantic clustering, insight quality scoring
- **Additional evaluation strategies**: multi-judge consensus, domain-specific judges
- **Research applications**: studying competitive-collaborative dynamics, optimal elimination rates

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Disclaimer

> **Important:** This is a personal open-source project, not affiliated with or endorsed by any organization. All development work is performed outside working hours using personal equipment and resources.

## License

This project is licensed under the MIT License - see the [**LICENSE**](./LICENSE) file for details.

## Citation

If you use Arbitrium Framework in your research, please cite:

```bibtex
@software{arbitrium2025,
  author = {Eremeev, Nikolay},
  title = {Arbitrium: Tournament-Based Multi-Model Synthesis Framework},
  year = {2025},
  version = {0.0.1},
  url = {https://github.com/arbitrium-framework/arbitrium}
}
```

See [CITATION.cff](CITATION.cff) for the full citation metadata.
