# Code and Data for "Analytical error budgeting for quantum error correction: construction, accuracy, and a cat-code example"

This repository contains the `qec_aeb` and `cat_code` Python packages, simulation and plotting scripts, and precomputed JSON data files for the paper *"Analytical error budgeting for quantum error correction: construction, accuracy, and a cat-code example"*.

## Installation

Requires Python >= 3.10 and `chencrafts` >= 4.0.0. From the root of this repository, install `qec_aeb` and `cat_code` in editable mode:

```bash
pip install -e .
```

Run the unit tests with:

```bash
pytest tests
```

## Usage

Each script under `scripts/` corresponds to a calculation or figure in the paper, reads/writes its `.json` data file in `data/`, and writes figure PDFs and PNGs to `figs/`. Figure scripts support `--plot-only` to regenerate figures directly from the stored JSON data without re-running simulations:

```bash
python scripts/1_calibration/ofek-parameter-fit.py
python scripts/2_accuracy_and_bounds/error-budget.py --plot-only
python scripts/2_accuracy_and_bounds/accuracy-sweep.py --plot-only
python scripts/2_accuracy_and_bounds/magnitudes.py --plot-only
python scripts/3_mechanism_budget/cycle-mechanism-budget.py --plot-only
python scripts/3_mechanism_budget/formula-terms.py --plot-only
python scripts/4_optimization/optimization-figure.py
python scripts/4_optimization/idle-duration-sweep.py --plot-only
python scripts/4_optimization/ancilla-comparison.py --plot-only
```

## Repository Structure

- `qec_aeb/`: General, code-agnostic QEC analytical error budgeting (AEB) framework (Sections II–IV and Appendices A–B):
  - `node.py`, `edge.py`, `graph.py`, `utils.py`, `settings.py`: Actual evolution tree, logical evolution tree, and failure Markov chain.
  - `reduction_chain.py`: Eight-model reduction chain between the actual QEC dynamics and the failure Markov chain, and breakdown by logical manifestation.
  - `magnitudes.py`: Step-by-step diamond-norm magnitudes of the seven neglected processes, discrepancy bounds of Appendix B, and square-root relations.
  - `budget.py`: General `ErrorContrib` and `ErrorBudget` containers.
- `cat_code/`: Cat-code QEC protocol, resonator–ancilla device models, and analytical error budget by mechanism (Sections V–VI and Appendices C–K):
  - `systems.py`, `sweeps.py`, `ideal.py`, `real.py`, `tree.py`: Resonator–transmon and resonator–fluxonium Hilbert spaces, batched dressed sweeps, ideal cat codewords and step operators, dressed jump operators and master-equation propagators, and `FullCatTreeBuilder`.
  - `formula.py`: Closed-form 31-term AEB by mechanism (`cat_code_budget`) and single-cycle evaluation (`evaluate_formula`, `mechanism_entries`).
  - `device_model.py`, `fluxonium_model.py`: Reference parameters of Ofek et al., zero-flux fluxonium model, dressed sweeps, actual evolution trees, and mechanism-selective sweeps.
  - `gate_model.py`, `budget_tools.py`: Rotating-wave ancilla pulse model and single-step/single-mechanism master-equation attribution helpers.
  - `optimization.py`: Four-parameter search ranges, L-BFGS-B optimization objective, dispersive-regime checks, and three-model sweep helpers.
  - `plot_style.py`: Shared REVTeX figure styling and color palette.
- `scripts/`: Calculation and figure scripts organized by topic:
  - `scripts/1_calibration/`: Calibration of the Ofek et al. circuit, bath, $1/f$ noise, and readout assignment error parameters (`ofek-parameter-fit.py`).
  - `scripts/2_accuracy_and_bounds/`: AEB by step, incoherent-error scale sweep, merged parity-mapping steps, error–leakage coherence attribution, step-by-step magnitudes and bounds, physical mechanisms of neglected processes, and three minimal $N$-step models.
  - `scripts/3_mechanism_budget/`: AEB by mechanism, term-by-term comparison against single-mechanism simulations, and ancilla pulse error breakdown.
  - `scripts/4_optimization/`: Multi-start parameter optimization for the transmon (`optimization.py`) and zero-flux fluxonium (`fluxonium-optimization.py`), 2D failure-rate landscape, dispersive-shift trade-off, EA idling duration sweep and attribution, coupling scan, evaluation time benchmark, range of validity, multi-cycle fidelity, and transmon vs. fluxonium comparison.
- `data/`: Precomputed `.json` data files.
- `figs/`: Generated figure PDFs, PNGs, and TikZ style files.
- `tests/`: Unit and integration test suite for `qec_aeb` and `cat_code`.
