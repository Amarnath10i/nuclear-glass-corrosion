# Changelog

## [0.2.0] - 2026-08-20

### Added
- evaluation/benchmarks.py: standardised benchmark suite
- docker/: Dockerfile and docker-compose for reproducible training
- docs/: documentation skeleton
- CLI entry-points in pyproject.toml

### Fixed
- models/bayesian_nn.py: KL divergence closure capture bug
- training/losses.py: variable name typo in mass_conservation_residual

### Changed
- Version bumped 0.1.0 to 0.2.0

## [0.1.0] - 2026-03-27

### Added
- Initial project scaffold
- Physics modules: thermodynamics, kinetics, transport, radiation
- ML models: PINO (FNO), DeepONet, Bayesian NN, Conformal prediction
- Data loaders: SRL165, ISG, VHT, synthetic generator
- Training: losses, PINO trainer, callbacks
- Evaluation: metrics, UQ decomposition
- Multi-scale coupling and homogenisation
- Unit tests (pytest)
- CI/CD (GitHub Actions)
