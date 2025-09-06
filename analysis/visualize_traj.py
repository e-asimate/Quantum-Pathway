#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import numpy as np
import imageio
import MDAnalysis as mda
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_size(size_str: str) -> Tuple[int, int]:
    if 'x' in size_str:
        w, h = size_str.split('x')
        return int(w), int(h)
    return int(size_str), int(size_str)


def draw_box(ax, box):
    a, b, c = box
    edges = [
        ((0,0,0),(a,0,0)), ((0,0,0),(0,b,0)), ((0,0,0),(0,0,c)),
        ((a,b,0),(0,b,0)), ((a,b,0),(a,0,0)), ((a,b,0),(a,b,c)),
        ((a,0,c),(0,0,c)), ((a,0,c),(a,0,0)), ((a,0,c),(a,b,c)),
        ((0,b,c),(0,0,c)), ((0,b,c),(0,b,0)), ((0,b,c),(a,b,c)),
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2], color='black', linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, a)
    ax.set_ylim(0, b)
    ax.set_zlim(0, c)


def element_colors(symbols: np.ndarray) -> np.ndarray:
    color_map = {
        'H': np.array([0.8, 0.8, 0.8]),
        'O': np.array([0.8, 0.1, 0.1]),
    }
    return np.array([color_map.get(sym, np.array([0.2, 0.2, 0.8])) for sym in symbols])


def render(traj_top: str, traj_path: str, selection: str, stride: int, fps: int, size: str, out_mp4: str, elev: float, azim: float, atom_size: float):
    u = mda.Universe(traj_top, traj_path)
    ag = u.select_atoms(selection)
    writer = imageio.get_writer(out_mp4, fps=fps, codec='libx264', quality=8)

    width, height = parse_size(size)
    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # symbol list (try MDAnalysis elements first, fall back to atom names)
    try:
        symbols = np.array([atom.element.symbol for atom in ag.atoms])
    except Exception:
        symbols = np.array([atom.name[0] for atom in ag.atoms])
    colors = element_colors(symbols)

    frame_count = 0
    for ts in u.trajectory[::stride]:
        ax.cla()
        box = ts.dimensions[:3]
        draw_box(ax, box)
        pos = ag.positions.copy()
        # Wrap positions to primary cell
        pos = pos - np.floor(pos / box) * box
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=atom_size, c=colors, depthshade=False)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img)
        frame_count += 1
    writer.close()
    plt.close(fig)
    print(f"Wrote {out_mp4} with {frame_count} frames @ {fps} fps")


def main():
    ap = argparse.ArgumentParser(description="Render MD trajectory to MP4 video")
    ap.add_argument("--top", required=True, help="Topology file (e.g., PDB)")
    ap.add_argument("--traj", required=True, help="Trajectory file (e.g., DCD)")
    ap.add_argument("--sel", default="all", help="Atom selection (e.g., 'name O or name H')")
    ap.add_argument("--stride", type=int, default=5, help="Frame stride for rendering")
    ap.add_argument("--fps", type=int, default=24, help="Output frames per second")
    ap.add_argument("--size", default="800x800", help="Video size WxH in pixels")
    ap.add_argument("--out", default="/workspace/data/trajectory.mp4", help="Output MP4 path")
    ap.add_argument("--elev", type=float, default=20.0, help="Camera elevation angle")
    ap.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth angle")
    ap.add_argument("--atom_size", type=float, default=6.0, help="Matplotlib scatter size")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    render(args.top, args.traj, args.sel, args.stride, args.fps, args.size, args.out, args.elev, args.azim, args.atom_size)


if __name__ == "__main__":
    main()
Optional helper script scripts/render_video.sh:

#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 TOPOLOGY_PDB TRAJECTORY_DCD [OUTPUT_MP4]" >&2
  exit 1
fi
TOP=$1
TRJ=$2
OUT=${3:-/workspace/data/trajectory.mp4}
python /workspace/analysis/visualize_traj.py --top "$TOP" --traj "$TRJ" --out "$OUT"
echo "Video written to $OUT"
