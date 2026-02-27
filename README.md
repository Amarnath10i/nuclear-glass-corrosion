# Nuclear Waste Glass Corrosion Prediction (100,000-Year Horizon)

**Physics-Informed Multi-Scale ML for Vitrified Nuclear Waste Long-Term Performance Assessment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Project Overview

This project develops **physics-informed machine learning models** to predict the corrosion behavior of borosilicate nuclear waste glass over **100,000-year timescales** — a critical requirement for geological repository safety cases (e.g., Yucca Mountain, WIPP, Cigéo, ONKALO).

### The Challenge
- **Timescale gap**: Lab experiments run for years; repositories must prove safety for 100,000+ years
- **Multi-physics**: Coupled chemical dissolution, diffusion, precipitation, radiation effects
- **Multi-scale**: Ångstrom (bond breaking) → micron (gel layer) → meter (repository near-field)
- **Data scarcity**: Limited long-term experimental data, heavy reliance on accelerated tests

### Our Approach
| Scale | Method | Output |
|-------|--------|--------|
| **Atomistic** | Reactive MD (ReaxFF) + DFT | Bond breaking rates, activation energies |
| **Mesoscale** | Phase-field / KMC | Gel layer growth, porosity evolution |
| **Continuum** | Reactive transport (PFLOTRAN/TOUGHREACT) | pH, saturation indices, radionuclide release |
| **ML Surrogate** | Physics-Informed Neural Operators (PINO/DeepONet) | 1000× speedup with UQ |
| **UQ** | Bayesian NN + Conformal Prediction | Validated uncertainty bounds |

---

## 📁 Repository Structure

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

## 🚀 Quick Start

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

## 📊 Key Datasets

| Dataset | Description | Source | Timescale |
|---------|-------------|--------|-----------|
| **SRL 165** | Savannah River Lab glass, PCT-A tests | DOE | 1-10 years |
| **ISG** | International Simple Glass (ISG) | CEA/ANDRA | 1-5 years |
| **LAW/LAWA** | Low-Activity Waste glasses | PNNL | 1-2 years |
| **MCC-1** | MCC-1 static leach tests | MCC | 1-10 years |
| **VHT** | Vapor Hydration Tests (accelerated) | PNNL | 100-1000°C days |
| **Natural Analogues** | Roman glass, basaltic glass | Literature | 1000-2M years |

---

## 🧪 Physics Constraints Implemented

1. **Thermodynamic Consistency**: Affinity-based rate law (TST)
   ```
   r = k₀ exp(-Ea/RT) · (1 - exp(-A/RT))^η
   ```

2. **Mass Conservation**: ∂C/∂t = ∇·(D∇C) + R(C)

3. **Gel Layer Growth**: Parabolic rate law with porosity evolution

4. **Radiation Effects**: α-recoil damage + radiolysis enhancement

5. **Uncertainty Quantification**: Aleatoric + Epistemic separation

---

## 📈 Target Publications (Q1 Journals)

| Paper | Target Journal | Timeline |
|-------|----------------|----------|
| **Physics-Informed Neural Operators for 100 kyr Glass Corrosion** | *Nature Materials* | Month 6-9 |
| **Multi-Fidelity ML Coupling Atomistic MD to Continuum Transport** | *Journal of Nuclear Materials* | Month 9-12 |
| **Conformal Prediction for Nuclear Waste Form Performance** | *Environmental Science & Technology* | Month 12-15 |
| **Radiation-Enhanced Corrosion: ML Surrogate for Alpha-Dose Effects** | *Acta Materialia* | Month 15-18 |

---

## 🏗️ Startup Potential

**Company**: *Vitreous AI* (working name)

**Business Model**: SaaS platform for nuclear waste organizations
- **Customers**: DOE (EM), ANDRA (France), NDA (UK), NWMO (Canada), POSIVA (Finland), Rosatom
- **Product**: Regulatory-grade corrosion prediction with UQ for safety cases
- **Moat**: Only physics-informed ML with validated UQ for 100 kyr predictions
- **Revenue**: $500K-$2M/year per license (regulatory requirement)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📞 Contact

**Amarnath** — [@Amarnath10i](https://github.com/Amarnath10i)
- Research: Multi-scale ML for nuclear materials
- Open to: PhD positions, research collaborations, startup co-founders