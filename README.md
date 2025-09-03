## Ab-initio MD and Quantum VQE for Water (H2O)

Reproducible workflows to simulate liquid water with DFT-based AIMD (CP2K) and to compute molecular H2O properties with Qiskit Nature VQE.

### Contents
- `aimd/`: CP2K inputs and utilities
  - `cp2k/`: input templates for NPT/NVT, SCAN-rVV10 and PBE0-D3
  - `utils/`: system builders and helpers
- `analysis/`: Python scripts for RDFs, MSD/diffusion, IR spectrum, dielectric
- `vqe/`: Qiskit Nature VQE for H2O (STO-3G, active space, UCCSD)
- `data/`: initial coordinates, trajectories, outputs
- `scripts/`: convenience launchers

### Quick start
1) Create a water box (64 H2O, ~12.4 Å edge):
```bash
python -m aimd.utils.build_water_box --num 64 --density 0.997 --out /workspace/data/water64.pdb
```

2) Run CP2K NPT equilibration then NVT production (edit pseudopotential/basis paths inside input):
```bash
scripts/run_cp2k_npt.sh /workspace/data/water64.pdb
scripts/run_cp2k_nvt.sh npt/restart
```

3) Analyze trajectory:
```bash
python analysis/rdf.py --traj data/production.dcd --top data/system.pdb
python analysis/msd.py --traj data/production.dcd --top data/system.pdb
python analysis/ir.py --dipoles data/dipoles.npy --dt 0.5
```

4) VQE for molecular H2O:
```bash
python vqe/h2o_vqe.py --basis STO-3G --active 4,4 --ansatz UCCSD
```

### Targets
- Density ≈ 0.997 g/cm^3 at 298 K; RDFs g_OO(r), g_OH(r), g_HH(r)
- Self-diffusion D ≈ 2.3e-9 m^2/s; dielectric ≈ 78
- IR/Raman features: bend ~1645 cm^-1; OH stretch ~3200–3600 cm^-1

### Notes
- Prefer SCAN-rVV10 or PBE0-D3; enable dispersion consistently.
- Use Γ-point only; test cutoffs (400–600 Ry wavefunction; ≥1000 Ry density for CP2K).
- Timestep 0.5–1.0 fs; NPT → NVT; consider PIMD (16 beads) for NQEs.

### Reproducibility
Record code versions, seeds, functional/dispersion, pseudopotential and basis files, cutoffs, thermostat/barostat parameters, and all inputs/outputs.

# Quantum-Pathway
An exploration of Quantum computing
