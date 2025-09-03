#!/usr/bin/env python3
import argparse
import math
import random
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write
from ase.units import kB


def compute_box_length_angstrom(num_molecules: int, density_g_per_cm3: float) -> float:
    molar_mass_kg_per_mol = 0.01801528  # kg/mol for H2O
    na = 6.02214076e23
    mass_per_molecule_kg = molar_mass_kg_per_mol / na
    total_mass_kg = mass_per_molecule_kg * num_molecules
    density_kg_per_m3 = density_g_per_cm3 * 1000.0
    volume_m3 = total_mass_kg / density_kg_per_m3
    length_m = volume_m3 ** (1.0 / 3.0)
    length_ang = length_m * 1.0e10
    return length_ang


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    q = np.array([q1, q2, q3, q4])
    q1, q2, q3, q4 = q
    rot = np.array([
        [1 - 2*(q3*q3 + q4*q4), 2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [2*(q2*q3 + q1*q4),     1 - 2*(q2*q2 + q4*q4), 2*(q3*q4 - q1*q2)],
        [2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2),     1 - 2*(q2*q2 + q3*q3)],
    ])
    return rot


def minimum_image_distance(vec: np.ndarray, box_len: float) -> float:
    delta = vec - box_len * np.round(vec / box_len)
    return float(np.linalg.norm(delta))


def place_molecules(num_molecules: int, box_len: float, min_com_dist: float, rng: np.random.Generator) -> np.ndarray:
    positions = []
    attempts = 0
    max_attempts = num_molecules * 10000
    while len(positions) < num_molecules and attempts < max_attempts:
        attempts += 1
        trial = rng.random(3) * box_len
        ok = True
        for p in positions:
            if minimum_image_distance(trial - p, box_len) < min_com_dist:
                ok = False
                break
        if ok:
            positions.append(trial)
    if len(positions) != num_molecules:
        raise RuntimeError("Failed to place molecules without overlap. Try smaller min distance or use pre-equilibrated snapshot.")
    return np.array(positions)


def build_water_box(num_molecules: int, density: float, temperature_K: float, seed: int, min_com_distance: float) -> Atoms:
    rng = np.random.default_rng(seed)
    # Start from an isolated water molecule (approx geometry)
    water = molecule('H2O')
    # Center molecule at origin
    water.translate(-water.get_center_of_mass())

    box_len = compute_box_length_angstrom(num_molecules, density)
    cell = np.eye(3) * box_len

    com_positions = place_molecules(num_molecules, box_len, min_com_distance, rng)
    all_symbols = []
    all_positions = []
    for com in com_positions:
        rot = random_rotation_matrix(rng)
        rotated = (water.get_positions() @ rot.T)
        translated = rotated + com
        all_symbols.extend(water.get_chemical_symbols())
        all_positions.extend(translated)

    system = Atoms(symbols=all_symbols, positions=np.array(all_positions), cell=cell, pbc=(True, True, True))

    # Maxwell-Boltzmann velocities
    masses = system.get_masses() * 1.66053906660e-27  # kg
    sigma = np.sqrt(kB * temperature_K / masses)  # ASE kB in eV/K; but velocities in sqrt(eV/kg)?
    # Use ASE's built-in MaxwellBoltzmannDistribution for consistency
    try:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        MaxwellBoltzmannDistribution(system, temperature_K * 8.617333262145e-5)  # ASE expects eV units for kT
        Stationary(system)
        ZeroRotation(system)
    except Exception:
        # Fallback: zero velocities
        system.set_velocities(np.zeros_like(system.get_positions()))

    return system


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a periodic water box at target density.")
    p.add_argument("--num", type=int, default=64, help="Number of H2O molecules")
    p.add_argument("--density", type=float, default=0.997, help="Density in g/cm^3")
    p.add_argument("--temperature", type=float, default=298.0, help="Temperature in K for velocities")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--min_com", type=float, default=2.3, help="Minimum COM distance in Å for placement")
    p.add_argument("--out", type=str, required=True, help="Output PDB path")
    return p.parse_args()


def main():
    args = parse_args()
    system = build_water_box(
        num_molecules=args.num,
        density=args.density,
        temperature_K=args.temperature,
        seed=args.seed,
        min_com_distance=args.min_com,
    )
    write(args.out, system)
    # Also write a simple topology PDB with CONECTs omitted (CP2K does not need bonds)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

