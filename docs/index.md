# Arbitrium Framework

**A framework for collaborative-competitive LLM evaluation using multi-model tournaments.**

## Overview

Arbitrium Framework is a Python framework that helps you make better decisions by running collaborative-competitive tournaments between multiple LLM models. Instead of relying on a single model, Arbitrium Framework orchestrates multiple models to debate, refine, and improve answers through structured rounds.

## Key Features

- **Multi-Model Tournaments**: Run competitions between 3-5 LLM models simultaneously
- **Iterative Refinement**: Models improve their answers based on feedback from competitors
- **Flexible Configuration**: Support for OpenAI, Anthropic, Google, X.AI, and Ollama models
- **Rich Analytics**: Detailed reports with metrics, comparisons, and insights
- **Knowledge Bank**: Learn from past tournaments to improve future results
- **Production-Ready**: Async architecture with retry logic and comprehensive logging

## Quick Example

```python
import asyncio
from arbitrium.config import Config
from arbitrium.core.comparison import ModelComparison
from arbitrium.models.base import LiteLLMModel
from arbitrium.utils.display import Display

async def main():
    # Load configuration
    config = Config("config.yml")
    config.load()

    # Initialize models
    models = {
        key: LiteLLMModel.from_config(key, cfg)
        for key, cfg in config.config_data["models"].items()
    }

    # Run comparison
    comparison = ModelComparison(
        config=config.config_data,
        models=models,
        display=Display()
    )

    result = await comparison.run(
        "What are the trade-offs between monolithic and microservices?"
    )

    print(f"Champion Answer: {result}")

asyncio.run(main())
```

## When to Use Arbitrium Framework

Arbitrium Framework excels at:

- **High-stakes decisions** where accuracy matters more than speed
- **Complex problems** requiring multiple perspectives
- **Research and analysis** where depth of reasoning is critical
- **Content generation** where quality justifies the extra cost

Use our [ROI Calculator](calculator.html) to determine if Arbitrium Framework is right for your use case.

## Performance

| Models   | Time    | Cost  | Quality Improvement |
| -------- | ------- | ----- | ------------------- |
| 3 models | ~7 min  | $0.92 | 15-20%              |
| 4 models | ~9 min  | $1.53 | 18-25%              |
| 5 models | ~15 min | $2.62 | 20-30%              |

## Getting Started

1. [Install Arbitrium Framework](getting-started/installation.md)
2. [Configure your models](getting-started/configuration.md)
3. [Run your first comparison](getting-started/quick-start.md)

## Architecture

```
┌─────────────────────────────────────────┐
│         Question Input                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Initial Answers (All Models)         │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
│  │ GPT │  │Claude│  │Gemini│ │ ... │   │
│  └─────┘  └─────┘  └─────┘  └─────┘   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Iterative Improvement Rounds (3x)      │
│  • Cross-model feedback                 │
│  • Answer refinement                    │
│  • Quality evaluation                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Champion Selection & Report          │
│  • Best answer selection                │
│  • Detailed metrics                     │
│  • Performance analytics                │
└─────────────────────────────────────────┘
```

## License

MIT License - see [LICENSE](https://github.com/nikolay-e/arbitrium-framework/blob/main/LICENSE) for details.
