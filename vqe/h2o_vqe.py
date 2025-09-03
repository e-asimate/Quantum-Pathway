#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA, SPSA
from qiskit.primitives import Estimator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.drivers import Molecule
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


def build_problem(basis: str, active: Tuple[int, int], r_oh: float, angle_deg: float):
    # Build H2O geometry from internal coordinates (approx)
    import numpy as np
    angle = np.deg2rad(angle_deg)
    x = r_oh * np.sin(angle / 2.0)
    z = r_oh * np.cos(angle / 2.0)
    geom = [["O", [0.0, 0.0, 0.0]], ["H", [x, 0.0, z]], ["H", [-x, 0.0, z]]]
    mol = Molecule(geometry=geom, charge=0, multiplicity=1, mass=None, units=DistanceUnit.ANGSTROM)
    driver = PySCFDriver.from_molecule(mol, basis=basis)
    problem = driver.run()

    # Freeze O 1s core, then select active space
    transformers = [FreezeCoreTransformer(), ActiveSpaceTransformer(*active)]
    for t in transformers:
        problem = t.transform(problem)
    return problem


def choose_mapper(mapper: str):
    return JordanWignerMapper() if mapper.lower() in ["jw", "jordan-wigner"] else BravyiKitaevMapper()


def choose_optimizer(name: str):
    name = name.lower()
    if name == "cobyla":
        return COBYLA(maxiter=1000)
    if name == "spsa":
        return SPSA(maxiter=2000)
    return L_BFGS_B(maxiter=500)


def run_vqe(basis: str, active_e: int, active_o: int, mapper: str, ansatz_name: str, optimizer_name: str,
            r_oh: float, angle_deg: float):
    problem = build_problem(basis, (active_e, active_o), r_oh, angle_deg)
    mapper_obj = choose_mapper(mapper)
    qubit_converter = mapper_obj

    num_spatial = problem.num_spatial_orbitals
    num_particles = problem.num_particles

    ansatz = UCCSD(num_spatial_orbitals=num_spatial, num_particles=num_particles, qubit_mapper=qubit_converter)
    estimator = Estimator()
    optimizer = choose_optimizer(optimizer_name)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

    gse = GroundStateEigensolver(qubit_converter, vqe)
    res = gse.solve(problem)
    energy = res.total_energies[0].real
    print(f"Basis={basis} Active=({active_e},{active_o}) Qubits={ansatz.num_qubits} Energy={energy:.8f} Ha")
    return energy


def main():
    p = argparse.ArgumentParser(description="VQE for molecular H2O ground-state energy")
    p.add_argument("--basis", default="STO-3G")
    p.add_argument("--active", default="4,4", help="active electrons,orbitals")
    p.add_argument("--mapper", default="jw", choices=["jw", "bk"]) 
    p.add_argument("--ansatz", default="UCCSD")
    p.add_argument("--optimizer", default="L_BFGS_B")
    p.add_argument("--roh", type=float, default=0.958, help="O-H bond length in Å")
    p.add_argument("--angle", type=float, default=104.5, help="HOH angle in degrees")
    args = p.parse_args()

    e, o = map(int, args.active.split(','))
    run_vqe(args.basis, e, o, args.mapper, args.ansatz, args.optimizer, args.roh, args.angle)


if __name__ == "__main__":
    main()

