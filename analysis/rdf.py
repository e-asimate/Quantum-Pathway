#!/usr/bin/env python3
import argparse
import numpy as np
import MDAnalysis as mda


def compute_rdf(u: mda.Universe, sel_i: str, sel_j: str, r_max: float, nbins: int):
    ag_i = u.select_atoms(sel_i)
    ag_j = u.select_atoms(sel_j)
    edges = np.linspace(0.0, r_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins)
    vol = 4.0 / 3.0 * np.pi * (r_max ** 3)

    n_frames = 0
    for ts in u.trajectory:
        box = ts.dimensions[:3]
        pos_i = ag_i.positions
        pos_j = ag_j.positions
        # Pair distances with PBC via broadcasting
        diff = pos_i[:, None, :] - pos_j[None, :, :]
        diff -= box * np.round(diff / box)
        dist = np.linalg.norm(diff, axis=-1)
        # Exclude self when i==j
        if ag_i is ag_j:
            dist = dist[np.triu_indices_from(dist, k=1)]
        else:
            dist = dist.ravel()
        counts, _ = np.histogram(dist, bins=edges)
        hist += counts
        n_frames += 1

    rho = (len(ag_j) / (u.dimensions[0] * u.dimensions[1] * u.dimensions[2]))
    shell_vol = 4.0 * np.pi * centers ** 2 * (edges[1] - edges[0])
    norm = n_frames * len(ag_i) * rho * shell_vol
    g_r = hist / np.maximum(norm, 1e-12)
    return centers, g_r


def main():
    p = argparse.ArgumentParser(description="Compute RDFs g_OO, g_OH, g_HH from a trajectory")
    p.add_argument("--top", required=True, help="Topology file (e.g., PDB)")
    p.add_argument("--traj", required=True, help="Trajectory file (e.g., DCD)")
    p.add_argument("--rmax", type=float, default=6.0, help="Max radius in Å")
    p.add_argument("--nbins", type=int, default=300, help="Number of bins")
    p.add_argument("--out", type=str, default="/workspace/data/rdf.npz", help="Output npz path")
    args = p.parse_args()

    u = mda.Universe(args.top, args.traj)

    r, g_OO = compute_rdf(u, "name O", "name O", args.rmax, args.nbins)
    r, g_OH = compute_rdf(u, "name O", "name H", args.rmax, args.nbins)
    r, g_HH = compute_rdf(u, "name H", "name H", args.rmax, args.nbins)

    np.savez(args.out, r=r, g_OO=g_OO, g_OH=g_OH, g_HH=g_HH)
    print(f"Saved RDFs to {args.out}")


if __name__ == "__main__":
    main()

