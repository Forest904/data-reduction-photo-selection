PYTHON := uv run python
RESULTS ?= experiments/results
FIGURES ?= experiments/figures
HARDWARE_NOTES ?=
BATCH_ID ?=

EXPERIMENT_ARGS = --output "$(RESULTS)"
ifneq ($(strip $(HARDWARE_NOTES)),)
EXPERIMENT_ARGS += --hardware-notes "$(HARDWARE_NOTES)"
endif
ifneq ($(strip $(BATCH_ID)),)
EXPERIMENT_ARGS += --batch-id "$(BATCH_ID)"
endif

.PHONY: help sync validate-data test lint check run-method-d figures \
	exp-small exp-scalability exp-budget exp-ablations \
	exp-exact-infeasibility exp-synthetic exp-core exp-all clean-experiments

help:
	@$(PYTHON) -c "lines=['Available targets:', '  sync                    Install/sync dependencies with uv', '  validate-data           Validate data/raw/photos.csv and data/raw/queries.csv', '  test                    Run pytest', '  lint                    Run Ruff checks', '  check                   Run tests and lint', '  run-method-d            Run Method D with budget 3 on the local dataset', '  exp-small               Run small exact comparison experiments', '  exp-scalability         Run B/D scalability experiments', '  exp-budget              Run budget sensitivity experiments', '  exp-ablations           Run Method D ablation experiments', '  exp-exact-infeasibility Run full-data exact infeasibility documentation', '  exp-synthetic           Run synthetic sanity experiments', '  exp-core                Run small, exact infeasibility, and figures', '  exp-all                 Run all experiment configs and figures', '  figures                 Regenerate figures from saved results', '  clean-experiments       Remove generated result folders and figure PNGs', '', 'Variables:', '  RESULTS=$(RESULTS)', '  FIGURES=$(FIGURES)', '  HARDWARE_NOTES=$(HARDWARE_NOTES)', '  BATCH_ID=$(BATCH_ID)']; print('\n'.join(lines))"

sync:
	uv sync

validate-data:
	$(PYTHON) scripts/validate_data.py

test:
	uv run pytest

lint:
	uv run ruff check .

check: test lint

run-method-d:
	$(PYTHON) scripts/run_method.py --method D --budget 3

exp-small:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/small.yaml $(EXPERIMENT_ARGS)

exp-scalability:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/scalability.yaml $(EXPERIMENT_ARGS)

exp-budget:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/budget_sensitivity.yaml $(EXPERIMENT_ARGS)

exp-ablations:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/d_ablations.yaml $(EXPERIMENT_ARGS)

exp-exact-infeasibility:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/exact_infeasibility.yaml $(EXPERIMENT_ARGS)

exp-synthetic:
	$(PYTHON) scripts/run_experiments.py --config experiments/configs/synthetic.yaml $(EXPERIMENT_ARGS)

exp-core: exp-small exp-exact-infeasibility figures

exp-all: exp-synthetic exp-small exp-scalability exp-budget exp-ablations exp-exact-infeasibility figures

figures:
	$(PYTHON) scripts/generate_figures.py --results "$(RESULTS)" --output "$(FIGURES)"

clean-experiments:
	$(PYTHON) -c "from pathlib import Path; import shutil; results=Path('$(RESULTS)'); figures=Path('$(FIGURES)'); [shutil.rmtree(path) for path in results.iterdir() if path.name != '.gitkeep'] if results.exists() else None; [path.unlink() for path in figures.glob('*.png')] if figures.exists() else None"
