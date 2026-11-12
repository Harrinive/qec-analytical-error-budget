# Code and Data for "Analytical error budgeting for quantum error correction: construction, accuracy, and a cat-code example"

This repository contains the `qec_aeb` Python package, simulation and plotting scripts, and JSON data files for the paper *"Analytical error budgeting for quantum error correction: construction, accuracy, and a cat-code example"*.

## Installation

Requires Python >= 3.10 and `chencrafts` >= 4.0.0. From the root of this repository, install `qec_aeb` in editable mode:

```bash
pip install -e .
```

Run the unit tests with:

```bash
pytest tests
```

## Usage

Each top-level script corresponds to a calculation or figure in the paper and reads/writes its `.json` data file in the repository root. Figure scripts write PDFs and PNGs to `figs/` and support `--plot-only` to regenerate figures directly from the stored JSON data without re-running simulations:

```bash
python error-budget.py --plot-only
python accuracy-sweep.py --plot-only
python cycle-mechanism-budget.py --plot-only
python optimization-figure.py
python idle-duration-sweep.py --plot-only
python formula-terms.py --plot-only
python magnitudes-figure.py
python ancilla-comparison.py --plot-only
```

## Repository Structure

- `qec_aeb/`: Python package implementing the evolution trees (actual evolution tree, logical evolution tree, and failure Markov chain), the cat-code QEC protocol model, and the analytical error budget (AEB).
- `tests/`: Unit and integration tests for `qec_aeb`.
- `*.py` and `*.json`: Simulation scripts, shared helper modules (`device_model.py`, `fluxonium_model.py`, `model_chain.py`, `step_magnitudes.py`, `budget_tools.py`, `gate_model.py`, `cycle_target.py`, `sweep_tools.py`, `search_ranges.py`, `plot_style.py`), and precomputed JSON data files.
