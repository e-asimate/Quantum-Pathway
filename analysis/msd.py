#!/usr/bin/env python3
import argparse
import numpy as np
import MDAnalysis as mda


def compute_msd(u: mda.Universe, sel: str, dt_fs: float, unwrap: bool = True):
    ag = u.select_atoms(sel)
    n = ag.n_atoms
    positions = []
    box = None
    for ts in u.trajectory:
        box = ts.dimensions[:3]
        positions.append(ag.positions.copy())
    pos = np.asarray(positions)  # (T, N, 3)
    if unwrap and box is not None:
        # unwrap via cumulative nearest image
        disp = pos[1:] - pos[:-1]
        disp -= np.round(disp / box) * box
        unwrapped = np.concatenate([pos[:1], pos[0:1] + np.cumsum(disp, axis=0)], axis=0)
        pos = unwrapped
    r0 = pos[0]
    dr = pos - r0
    msd = np.mean(np.sum(dr * dr, axis=-1), axis=1)
    times_ps = np.arange(len(msd)) * dt_fs * 1e-3
    return times_ps, msd


def estimate_diffusion(times_ps: np.ndarray, msd: np.ndarray, fit_start_ps: float, fit_end_ps: float):
    mask = (times_ps >= fit_start_ps) & (times_ps <= fit_end_ps)
    x = times_ps[mask] * 1e-12  # convert ps to s
    y = msd[mask] * 1e-20       # Å^2 to m^2
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    D = slope / 6.0
    return D, slope, intercept


def main():
    p = argparse.ArgumentParser(description="Compute MSD and self-diffusion from trajectory")
    p.add_argument("--top", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--dt", type=float, default=0.5, help="Time step in fs between saved frames")
    p.add_argument("--sel", type=str, default="name O", help="Atom selection for MSD (e.g., 'name O')")
    p.add_argument("--fit", type=str, default="5,40", help="Fit window in ps, e.g., '5,40'")
    p.add_argument("--out", type=str, default="/workspace/data/msd.npz")
    args = p.parse_args()

    u = mda.Universe(args.top, args.traj)
    t_ps, msd = compute_msd(u, args.sel, args.dt)
    start_ps, end_ps = map(float, args.fit.split(','))
    D, slope, intercept = estimate_diffusion(t_ps, msd, start_ps, end_ps)
    print(f"Estimated D = {D:.3e} m^2/s over {start_ps}-{end_ps} ps")
    np.savez(args.out, t_ps=t_ps, msd=msd, D=D)
    print(f"Saved MSD to {args.out}")


if __name__ == "__main__":
    main()

