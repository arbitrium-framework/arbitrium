# Certamen Core

> Extends [../CLAUDE.md](../CLAUDE.md)

## Ultimate Goal

Extract the absolute maximum of knowledge from AI — every last
insight, perspective, and connection that exists within these models
but remains inaccessible through ordinary interaction.

AI models contain vastly more knowledge than any single prompt can
unlock. They hedge, simplify, omit, and hold back. Certamen exists
to close that gap: leave nothing on the table.

## Architecture

Layered DDD: interfaces → application → infrastructure → domain →
ports → shared, plus bundled YAML workflows in
`src/certamen/workflows/`. The boundaries are enforced, not
aspirational — import-linter contracts in `pyproject.toml` and the
local `lint-structure` hook fail on a violation, so read the
contracts there rather than a tree here. Concepts and design
rationale live in `DESIGN.md`; commands in the `Makefile`
(`make help`).

Tournaments run via the CLI against a slim `config.yml` that
references a packaged workflow (e.g. `workflow: diamond-tournament`;
`--workflow` overrides it per run). There is no
`Certamen.run_tournament()` — the Python `Certamen` class exposes
only `run_single_model` and `run_all_models` for ad-hoc probing;
full tournament execution goes through the workflow executor.
Ad-hoc pipelines are YAML workflows (`certamen workflow execute`,
examples in `examples/workflows/`).

## Configuration

`config.example.yml` is the reference, including the `secrets:`
block — each provider's API key resolves from an env var or a
1Password `op://` path. Ollama models additionally require
`OLLAMA_BASE_URL` (the config loader raises without it).

## Testing

Integration tests only (no mocks). Tests run against real LLM providers:

```bash
pytest tests/integration/ -v
```

Requires API keys in environment or `.env`.
