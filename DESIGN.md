# Arbitrium Core Architecture

## Naming & Structure Contract

This document defines the canonical naming conventions and directory structure for arbitrium-core.

### Directory Naming Conventions

| Layer | Convention | Examples |
|-------|------------|----------|
| Python packages under `src/` | `snake_case` | `arbitrium_core` |
| Python modules | `snake_case.py` | `tournament.py`, `base_node.py` |
| Workflow templates | `kebab-case.yml` | `tournament-elimination.yml` |
| YAML extension | `.yml` only | Never `.yaml` |
| Top-level directories | `lowercase` | `src/`, `tests/`, `scripts/` |

### Canonical Directory Locations

| Concept | Location |
|---------|----------|
| Business logic | `src/arbitrium_core/domain/` |
| Workflow nodes | `src/arbitrium_core/domain/workflow/nodes/` |
| Node registry | `src/arbitrium_core/domain/workflow/registry.py` |
| LLM adapters | `src/arbitrium_core/adapters/llm/` |
| CLI | `src/arbitrium_core/interfaces/cli/` |
| Application services | `src/arbitrium_core/application/` |
| Support utilities | `src/arbitrium_core/support/` |

### Re-export Pattern

Top-level modules (like `executor.py`, `tournament.py`) are **pure facades**:

```python
from arbitrium_core.application.execution.executor import GraphExecutor as GraphExecutor
__all__ = ["GraphExecutor"]
```

The `nodes/` directory re-exports from `domain/workflow/nodes/`.

### Node File Naming

Node files use simple names without `_node` suffix:

| Correct | Incorrect |
|---------|-----------|
| `evaluation.py` | `evaluation_node.py` |
| `flow.py` | `flow_node.py` |
| `base.py` | `base_node.py` |

### Facade Pattern for nodes/

The `src/arbitrium_core/nodes/` directory is a **compatibility layer** containing only re-exports:

```python
# nodes/llm.py - allowed (pure re-export)
from arbitrium_core.domain.workflow.nodes.llm import *  # noqa: F401, F403
```

Allowed files in `nodes/`: `__init__.py`, `registry.py`, and re-export stubs only.

### Forbidden Patterns

1. No `.yaml` extension - use `.yml` only
2. No SQLite WAL files in git (`.db-wal`, `.db-shm`, `.db-journal`)
3. No `*_node.py` files in domain/workflow/nodes/
4. No implementation code in `nodes/` compat layer

---

# Config-as-Workflow-Facade Architecture

## Problem Statement

Arbitrium currently has two sources of truth for defining AI pipelines:

1. **config.yml** - Simple declarative format that launches tournament mode directly
2. **examples/workflows/*.yml** - Graph-based workflow definitions with nodes and edges

This creates confusion:
- Users don't know which format to use
- Features get duplicated across both paths
- Tournament logic exists in two places: hardcoded `tournament.py` and workflow nodes

## Solution: Config as Workflow Facade

Config becomes a **simplified syntax** that **translates to workflow internally**. A single execution engine (WorkflowExecutor) handles all executions.

```
config.yml → ConfigResolver → Full Workflow → WorkflowExecutor
                    ↓
            Builtin Templates
            (tournament.yml, etc.)
```

### Design Principles

Inspired by proven patterns in infrastructure tooling:

| Tool | Pattern | Arbitrium Analogy |
|------|---------|-------------------|
| [Helm](https://helm.sh/docs/chart_template_guide/values_files/) | values.yaml overrides chart templates | config.yml overrides workflow templates |
| [Terraform](https://mikeball.info/blog/terraform-patterns-the-wrapper-module/) | Facade modules wrap complex resources | Config wraps workflow graph complexity |
| [Kustomize](https://notes.kodekloud.com/docs/Certified-Kubernetes-Application-Developer-CKAD/2025-Updates-Kustomize-Basics/Overlays) | Base + overlay composition | Builtin template + user config overlay |
| [GitHub Actions](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows) | Reusable workflows with inputs | Workflow templates with config inputs |

### Key Benefits

1. **Single Source of Truth** - All execution paths use WorkflowExecutor
2. **Simplified Interface** - Users write simple config, system handles graph complexity
3. **Full Power When Needed** - Power users can still write raw workflows
4. **Maintainability** - Tournament logic lives in one place (workflow nodes)

## Architecture

### Current State (Dual Path)

```
                    ┌──────────────────────┐
                    │   arbitrium CLI      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  --config       │               │  workflow exec  │
    │  config.yml     │               │  workflow.yml   │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
             ▼                                  ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  Bootstrap      │               │  YAML Loader    │
    │  + tournament.py│               │  (no bootstrap) │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
             ▼                                  ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  Hardcoded      │               │  WorkflowExec   │
    │  Tournament Loop│               │  (models={})    │ ← BUG: models empty
    └─────────────────┘               └─────────────────┘
```

### Target State (Unified Path)

```
                    ┌──────────────────────┐
                    │   arbitrium CLI      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │  --config       │               │  workflow exec  │
    │  config.yml     │               │  workflow.yml   │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
             ▼                                  │
    ┌─────────────────┐                         │
    │  ConfigResolver │                         │
    │  type: tournament───▶ load template       │
    │  models: → inject                         │
    │  tournament.* → map                       │
    └────────┬────────┘                         │
             │                                  │
             ▼                                  ▼
    ┌─────────────────────────────────────────────┐
    │           WorkflowExecutor                  │
    │  ┌─────────────────────────────────────┐    │
    │  │  ExecutionContext                   │    │
    │  │  - models: {gpt: Model, claude: ...}│    │
    │  │  - config: {...}                    │    │
    │  └─────────────────────────────────────┘    │
    └─────────────────────────────────────────────┘
```

## Config Syntax

### Simple Config (Current)

```yaml
# config.yml - это workflow, просто упрощённый
type: tournament  # selects base workflow template

models:
  gpt: { provider: openai, model_name: gpt-4o }
  claude: { provider: anthropic, model_name: claude-3.5 }

question: "What is the best approach to..."

tournament:
  judge_model: claude
  improvement_rounds: 2
  anonymize: true

knowledge_bank:
  enabled: true
  max_insights: 100
```

### How It Translates

The `type: tournament` field loads a builtin workflow template:

```yaml
# internal: templates/tournament.yml (simplified)
name: Tournament Template
nodes:
  - id: question
    type: simple/text
    properties:
      texts: ["{{ config.question }}"]

  - id: models
    type: tournament/models
    properties:
      selected_models: "{{ config.models | keys }}"

  - id: gate
    type: flow/gate
    properties:
      max_rounds: "{{ config.tournament.improvement_rounds * len(models) }}"

  - id: generate
    type: tournament/generate

  - id: peer_review
    type: tournament/peer_review
    properties:
      anonymize: "{{ config.tournament.anonymize }}"
      criteria: "{{ config.tournament.criteria | default(DEFAULT_CRITERIA) }}"

  - id: eliminate
    type: tournament/eliminate
    properties:
      count: 1

edges:
  # ... standard tournament graph
```

### Node Overrides (Power User Feature)

```yaml
type: tournament
models: { ... }

# Direct patches to specific nodes
node_overrides:
  peer_review:
    criteria: |
      Rate on: accuracy (1-10), clarity (1-10)

  generate:
    system_prompt: |
      You are competing in an AI tournament. Give your best answer.
```

## Implementation Plan

### Phase 1: ConfigResolver

Create `ConfigResolver` class that:
1. Reads config type field
2. Loads corresponding workflow template from `templates/`
3. Injects config values into template placeholders
4. Returns complete workflow definition

```python
class ConfigResolver:
    def __init__(self, config: dict):
        self.config = config

    def resolve(self) -> WorkflowDefinition:
        workflow_type = self.config.get("type", "tournament")
        template = self._load_template(workflow_type)
        return self._apply_config(template, self.config)

    def _load_template(self, name: str) -> dict:
        template_path = TEMPLATES_DIR / f"{name}.yml"
        return yaml.safe_load(template_path.read_text())

    def _apply_config(self, template: dict, config: dict) -> WorkflowDefinition:
        # Inject models into tournament/models node
        # Map tournament.* to node properties
        # Apply node_overrides if present
        ...
```

### Phase 2: Fix ExecutionContext.models

Current bug: `SyncExecutor._initialize_execution_state()` creates `models={}`.

Fix: Pass loaded models from Bootstrap/ConfigResolver into ExecutionContext.

```python
# Before
def _initialize_execution_state(self, workflow: Workflow) -> dict[str, Any]:
    return ExecutionContext(
        models={},  # BUG: always empty
        ...
    )

# After
def _initialize_execution_state(
    self,
    workflow: Workflow,
    models: dict[str, Model] | None = None
) -> dict[str, Any]:
    return ExecutionContext(
        models=models or {},
        ...
    )
```

### Phase 3: Create Builtin Templates

Extract tournament logic from `tournament.py` into `templates/tournament.yml`.

Directory structure:
```
src/arbitrium/
└── templates/
    ├── __init__.py
    ├── tournament.yml      # Standard tournament
    ├── comparison.yml      # Simple A/B comparison
    └── chain.yml           # Sequential chain
```

### Phase 4: Remove Hardcoded Tournament

1. Delete or deprecate `domain/tournament/tournament.py`
2. Update `Arbitrium.run_tournament()` to use ConfigResolver + WorkflowExecutor
3. Maintain API compatibility

### Phase 5: Migrate Examples

Convert `config.example.yml` to new format:
```yaml
type: tournament
question: "..."
models: { ... }
tournament: { ... }
```

Update `examples/workflows/tournament-elimination.yml` to use simplified syntax or keep as power-user example.

## Workflow Template Inheritance

Following [Helm's values hierarchy](https://helm.sh/docs/chart_template_guide/values_files/):

```
Template defaults
    ↓ (overridden by)
Config values
    ↓ (overridden by)
node_overrides
    ↓ (overridden by)
CLI --set flags
```

## Comparison with Raw Workflows

| Aspect | Simple Config | Raw Workflow |
|--------|---------------|--------------|
| Verbosity | ~30 lines | ~260 lines |
| Flexibility | Preset patterns | Full graph control |
| Learning curve | Minutes | Hours |
| Use case | Standard tournaments | Custom pipelines |

Users start with simple config, graduate to raw workflows when needed.

## Migration Strategy

1. **Backward Compatibility** - Old configs continue to work
2. **Deprecation Warnings** - Log when using legacy tournament path
3. **Documentation** - Update CLAUDE.md with new patterns
4. **Examples** - Both simple config and raw workflow examples

## Open Questions

1. **Template Format** - YAML with Jinja2 placeholders vs Python templating?
2. **Validation** - Validate config against template schema?
3. **Custom Templates** - Allow users to define their own templates?
4. **CLI Integration** - `arbitrium --config` vs `arbitrium workflow exec`?

## References

- [Helm Values Files](https://helm.sh/docs/chart_template_guide/values_files/)
- [Terraform Wrapper Module Pattern](https://mikeball.info/blog/terraform-patterns-the-wrapper-module/)
- [Kustomize Overlays](https://notes.kodekloud.com/docs/Certified-Kubernetes-Application-Developer-CKAD/2025-Updates-Kustomize-Basics/Overlays)
- [Facade Design Pattern](https://refactoring.guru/design-patterns/facade)
- [GitHub Actions Reusable Workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
- [DSL Guide by Martin Fowler](https://martinfowler.com/dsl.html)
