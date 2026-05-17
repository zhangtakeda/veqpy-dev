# Repository Guidelines

## Project Structure & Module Organization

`veqpy/` contains the package source. Keep domain snapshots in `veqpy/model/` (`Grid`, `Profile`, `Geometry`, `Equilibrium`), packed solver layouts and codecs in `veqpy/operator/`, NumPy/Numba kernels in `veqpy/engine/`, and nonlinear solve orchestration/results in `veqpy/solver/`. Cross-cutting route and stage coordination lives in `veqpy/orchestration.py`; math helpers live in `veqpy/math/`; shared serialization/registry utilities live in `veqpy/base/`. Tests, demos, GEQDSK fixtures, and benchmark artifacts are under `tests/`. Documentation is in `docs/`; generated paper/demo figures are in `figures/`.

## Build, Test, and Development Commands

Use the project `uv` environment rather than system Python.

- `uv sync --group dev`: install runtime and development dependencies.
- `uv run python -m pytest`: run the full regression suite.
- `uv run python -m pytest tests/test_name.py`: run a focused test file.
- `uv run python tests/demo.py` and `uv run python tests/demo_geqdsk_workflow.py`: exercise demo workflows.
- `uv run python tests/benchmark.py`: compare performance-sensitive paths.
- `uv run ruff check veqpy tests`: run lint and import-order checks.
- `uv run python -m compileall veqpy tests`: validate imports and bytecode compilation.

If `uv` is unavailable, use `.venv/bin/python -m pytest` with the checked-in virtual environment.

## Coding Style & Naming Conventions

Target Python 3.12+. Use 4-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Ruff selects `E`, `F`, `W`, and `I` rules with a 100-character line length. Prefer typed, explicit array data flow for packed solver state. Do not move grid construction into hot runtime kernels, and avoid reintroducing fixed low-order special cases unless benchmarks justify them.

## Testing Guidelines

Use `pytest`; name regression files `tests/test_*.py`. Add focused tests near the affected subsystem and keep demo/GEQDSK workflows working when changing model, operator, solver, or orchestration behavior. For performance-sensitive kernel changes, compare representative low-order and high-order cases and record benchmark notes in the PR.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `update`, `add regular`, and `del fixed-point`; prefer clearer Conventional Commit prefixes when possible, for example `fix: preserve solver reset state` or `test: cover packed layout offsets`. PRs should include a concise summary, linked issues when applicable, commands run, benchmark results for hot paths, and screenshots/figures only for visual documentation changes.

## Agent-Specific Instructions

Maintain the architecture boundaries documented in `README.md` and `docs/guardrails.md`: packed layout authority stays in `veqpy/operator/`, grid construction stays outside hot runtime code, and `Equilibrium` remains a materialized snapshot rather than live runtime state.
