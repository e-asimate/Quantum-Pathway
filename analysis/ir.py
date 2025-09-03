#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.fft import rfft, rfftfreq


def ir_spectrum_from_dipoles(mu_t: np.ndarray, dt_fs: float, window: str = "hann"):
    # mu_t: (T, 3) total dipole vector time series
    mu = mu_t - mu_t.mean(axis=0, keepdims=True)
    corr = np.sum(mu * mu, axis=1)  # autocorrelation approx of |mu|^2
    # Windowing
    if window == "hann":
        w = np.hanning(len(corr))
        corr = corr * w
    dt_ps = dt_fs * 1e-3
    spec = np.abs(rfft(corr))
    freq_thz = rfftfreq(len(corr), d=dt_ps)  # THz if d in ps
    # Convert THz to cm^-1: 1 THz ~ 33.35641 cm^-1
    freq_cm = freq_thz * 33.35641
    return freq_cm, spec


def main():
    p = argparse.ArgumentParser(description="Compute IR spectrum from dipole time series")
    p.add_argument("--dipoles", required=True, help="Path to npy/npz containing dipoles (T,3)")
    p.add_argument("--dt", type=float, default=0.5, help="Time step in fs between frames")
    p.add_argument("--out", type=str, default="/workspace/data/ir.npz")
    args = p.parse_args()

    if args.dipoles.endswith('.npz'):
        data = np.load(args.dipoles)
        mu = data['dipoles']
    else:
        mu = np.load(args.dipoles)

    freq_cm, spec = ir_spectrum_from_dipoles(mu, args.dt)
    np.savez(args.out, freq_cm=freq_cm, intensity=spec)
    print(f"Saved IR spectrum to {args.out}")


if __name__ == "__main__":
    main()

