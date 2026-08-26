# Nuclear Waste Glass Corrosion Prediction (100,000-Year Horizon)

**Physics-Informed Multi-Scale ML for Vitrified Nuclear Waste Long-Term Performance Assessment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## About the Project

Safe disposal of high-level nuclear waste is one of the most pressing environmental challenges of our time. The international consensus is to immobilize this waste by vitrifying it into borosilicate glass and burying it in deep geological repositories. For these repositories to be approved, regulatory bodies require robust safety cases proving that the glass will contain the radioactive elements for over **100,000 years**.

Because we cannot run laboratory experiments for 100 millennia, traditional methods rely on accelerated testing and complex, computationally expensive reactive transport models that struggle to scale over long time horizons.

**This project bridges this gap by developing physics-informed machine learning (ML) surrogates that predict the long-term corrosion behavior of nuclear waste glass.**

By embedding known physical laws (such as Transition-State Theory rate laws and mass conservation) directly into the neural network's loss function, our models can extrapolate far beyond the temporal limits of training data without violating thermodynamic constraints.

## How It Works

The repository implements a complete multi-scale framework that bridges atomic-level interactions to repository-scale predictions:

1. **Multi-Scale Physics Coupling**
   - **Atomistic Scale**: Data derived from Reactive Molecular Dynamics (ReaxFF) determines fundamental parameters like bond-breaking rates and activation energies.
   - **Mesoscale**: We model the formation of the passivating "gel layer" on the glass surface and track its porosity and diffusion characteristics over time.
   - **Continuum Scale**: The atomic and mesoscale parameters are homogenized and fed into 1D reactive transport models to predict large-scale radionuclide release.

2. **Physics-Informed ML Surrogates (PINO / DeepONet)**
   - Instead of running slow numerical PDE solvers for every scenario, we train neural operators (like FNOs and DeepONets) to learn the underlying solution operator.
   - The networks are trained on experimental datasets (like SRL 165 and ISG) and penalized if their predictions violate the governing partial differential equations (PDEs) of reactive transport.
   - **Result**: A model that runs 1000× faster than numerical simulators while remaining physically consistent.

3. **Uncertainty Quantification (UQ)**
   - For regulatory safety cases, predicting a single number isn't enough; we must rigorously quantify confidence bounds.
   - The framework uses Bayesian Neural Networks to separate epistemic (model) uncertainty from aleatoric (data) uncertainty.
   - We also apply split Conformal Prediction to guarantee strict coverage probabilities for our 100,000-year extrapolations.

## Tools & Technologies

This project is built using a modern scientific Python stack tailored for high-performance ML and physics simulations:

- **Core ML Framework**: [PyTorch](https://pytorch.org/) (used for PINO, DeepONet, and Bayesian NNs)
- **Data & Scientific Computing**: `numpy`, `pandas`, `scipy`
- **Configuration Management**: `pydantic` and `omegaconf` for structured, type-safe experiment configs
- **Experiment Tracking**: [Weights & Biases (WandB)](https://wandb.ai/) for logging metrics, losses, and hyperparameters
- **Visualization**: `matplotlib` for domain-specific Arrhenius plots and uncertainty bounds
- **Environment & CI/CD**: Docker for reproducible training environments, GitHub Actions for continuous integration, and `pre-commit` hooks for code formatting (using `ruff` and `black`)

---

## Repository Structure

```
nuclear-glass-corrosion/
├── data/                          # Experimental & simulation datasets
│   ├── raw/                       # Original data (DO NOT COMMIT LARGE FILES)
│   ├── processed/                 # Cleaned, normalized datasets
│   └── benchmarks/                # Standard test cases (SRL, ISG, etc.)
├── src/
│   ├── nuclear_glass/             # Core package
│   │   ├── physics/               # Physics models & constraints
│   │   │   ├── thermodynamics.py  # Glass dissolution thermodynamics
│   │   │   ├── kinetics.py        # Rate laws (TST, affinity-based)
│   │   │   ├── transport.py       # Diffusion-reaction equations
│   │   │   └── radiation.py       # Radiation-enhanced corrosion
│   │   ├── models/                # ML architectures
│   │   │   ├── pino.py            # Physics-Informed Neural Operator
│   │   │   ├── deeponet.py        # DeepONet for operator learning
│   │   │   ├── bayesian_nn.py     # Bayesian neural networks
│   │   │   └── conformal.py       # Conformal prediction
│   │   ├── multiscale/            # Multi-scale coupling
│   │   │   ├── coupling.py        # Scale bridging
│   │   │   └── homogenization.py  # Upscaling
│   │   ├── data/                  # Data loading & preprocessing
│   │   │   ├── loaders.py         # Experimental data loaders
│   │   │   ├── simulation_io.py   # PFLOTRAN/TOUGHREACT parsers
│   │   │   └── augmentation.py    # Physics-aware augmentation
│   │   ├── training/              # Training pipelines
│   │   │   ├── losses.py          # Physics-informed losses
│   │   │   ├── trainers.py        # Custom trainers
│   │   │   └── callbacks.py       # Logging, checkpointing
│   │   ├── evaluation/            # Evaluation & UQ
│   │   │   ├── metrics.py         # Domain-specific metrics
│   │   │   ├── uncertainty.py     # UQ evaluation
│   │   │   └── benchmarks.py      # Standard benchmarks
│   │   └── utils/                 # Utilities
│   │       ├── config.py          # Configuration management
│   │       ├── logging.py         # Structured logging
│   │       └── visualization.py   # Domain-specific plots
│   └── cli/                       # Command-line interfaces
├── experiments/                   # Experiment configs & scripts
│   ├── configs/                   # YAML configs
│   ├── scripts/                   # Training/eval scripts
│   └── notebooks/                 # Exploratory notebooks
├── tests/                         # Unit & integration tests
├── docs/                          # Documentation
├── papers/                        # Paper drafts & figures
├── docker/                        # Dockerfiles for reproducibility
├── .github/workflows/             # CI/CD
├── pyproject.toml                 # Package config
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment
└── README.md
```

---

## Quick Start

### Installation

```bash
# Using conda (recommended for scientific stack)
conda env create -f environment.yml
conda activate nuclear-glass

# Or using pip
pip install -e .
```

### Run Training

```bash
# Train PINO on SRL glass dataset
python -m src.cli.train --config experiments/configs/pino_srl.yaml

# Train DeepONet with physics constraints
python -m src.cli.train --config experiments/configs/deeponet_affinity.yaml

# Run multi-fidelity training (MD + continuum)
python -m src.cli.train --config experiments/configs/multifidelity.yaml
```

### Run Evaluation

```bash
# Evaluate on held-out test set with UQ
python -m src.cli.evaluate --checkpoint checkpoints/pino_srl_best.pt --uq

# Generate benchmark comparison table
python -m src.cli.benchmark --config experiments/configs/benchmark.yaml
```

---

## Key Datasets

| Dataset | Description | Source | Timescale |
|---------|-------------|--------|-----------|
| **SRL 165** | Savannah River Lab glass, PCT-A tests | DOE | 1-10 years |
| **ISG** | International Simple Glass (ISG) | CEA/ANDRA | 1-5 years |
| **LAW/LAWA** | Low-Activity Waste glasses | PNNL | 1-2 years |
| **MCC-1** | MCC-1 static leach tests | MCC | 1-10 years |
| **VHT** | Vapor Hydration Tests (accelerated) | PNNL | 100-1000°C days |
| **Natural Analogues** | Roman glass, basaltic glass | Literature | 1000-2M years |

---

## Physics Constraints Implemented

1. **Thermodynamic Consistency**: Affinity-based rate law (TST)
   ```
   r = k₀ exp(-Ea/RT) · (1 - exp(-A/RT))^η
   ```

2. **Mass Conservation**: ∂C/∂t = ∇·(D∇C) + R(C)

3. **Gel Layer Growth**: Parabolic rate law with porosity evolution

4. **Radiation Effects**: α-recoil damage + radiolysis enhancement

5. **Uncertainty Quantification**: Aleatoric + Epistemic separation

---

## Target Publications (Q1 Journals)

| Paper | Target Journal | Timeline |
|-------|----------------|----------|
| **Physics-Informed Neural Operators for 100 kyr Glass Corrosion** | *Nature Materials* | Month 6-9 |
| **Multi-Fidelity ML Coupling Atomistic MD to Continuum Transport** | *Journal of Nuclear Materials* | Month 9-12 |
| **Conformal Prediction for Nuclear Waste Form Performance** | *Environmental Science & Technology* | Month 12-15 |
| **Radiation-Enhanced Corrosion: ML Surrogate for Alpha-Dose Effects** | *Acta Materialia* | Month 15-18 |

