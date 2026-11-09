"""Generate reproducible physical training, validation, and test data.

The revised default preserves the sampled continuous initial time.  The
provided checkpoints used the historical rounded-time distribution; pass
--legacy-rounded-time to reproduce its coordinates with the corrected solver.
No dataset generation takes place on import.  Existing checkpoints are not
trained or modified by this script.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from artifact_io import save_npz_atomic
import ttc as T


SNAPS = (0.0, 0.025, 0.05, 0.1, 0.2)
FINE = tuple(np.round(np.arange(0, 0.0501, 0.005), 4))


def generate(n, rng, snaps, *, rounded_time=False, batch=100000, dt=T.DT):
    """Sample all coordinates first, then compute labels in bounded batches."""
    s, y = T.sample_box(n, rng)
    if rounded_time:
        s = np.rint(s / T.DT) * T.DT
    fields = ("s", "V", "hit", "lam_imp", "y_imp", "y_final", "initial_gap")
    chunks = {k: [] for k in fields}
    snapshots = {float(k): [] for k in snaps}
    for first in range(0, n, batch):
        r = T.ttc(s[first:first + batch], y[first:first + batch], snap_times=snaps, dt=dt)
        for k in fields:
            chunks[k].append(r[k])
        for k in snaps:
            snapshots[float(k)].append(r["snaps"][float(k)])
    r = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    out = {**r, "y": y, "inD": T.in_D(r), "inDD": T.in_DD(r)}
    for k, parts in snapshots.items():
        # Retain the original key spelling used by train.py, e.g. snap_0.05.
        out[f"snap_{k}"] = np.concatenate(parts, axis=0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=1000000)
    p.add_argument("--n-val", type=int, default=100000)
    p.add_argument("--n-test", type=int, default=300000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch", type=int, default=100000)
    p.add_argument("--dt", type=float, default=T.DT)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--legacy-rounded-time", action="store_true",
                   help="round sampled start times to 0.001 as for the supplied checkpoints")
    args = p.parse_args()
    if min(args.n_train, args.n_val, args.n_test, args.batch) < 1:
        p.error("dataset sizes and batch size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    protocol = dict(seed=args.seed, rng="numpy.default_rng/PCG64",
                    initial_times="rounded_to_0.001" if args.legacy_rounded_time else "continuous/unrounded",
                    dt=args.dt, nbis=40, solver="event-aware piecewise-constant RK4",
                    sampling="independent uniforms in (s,Xi,vF,zeta_L); Lambda=zeta_L-vF",
                    sampling_box={k: list(v) for k, v in T.BOX.items()},
                    draw_order="train, val, test; for each split, arrays s,Xi,vF,zeta_L",
                    numerical_labels_are_certified=False,
                    splits={})
    specifications = [("train", args.n_train, SNAPS), ("val", args.n_val, SNAPS),
                      ("test", args.n_test, tuple(sorted(set(SNAPS) | set(FINE))))]
    for name, n, snaps in specifications:
        t0 = time.time()
        out = generate(n, rng, snaps, rounded_time=args.legacy_rounded_time,
                       batch=args.batch, dt=args.dt)
        save_npz_atomic(args.out_dir / f"data_{name}.npz", **out)
        protocol["splits"][name] = dict(n=n, snapshots=list(snaps),
                                       n_hit=int(out["hit"].sum()), n_D=int(out["inD"].sum()),
                                       elapsed_seconds=time.time() - t0)
        # Write after each completed split so a partial long run remains auditable.
        (args.out_dir / "data_protocol.json").write_text(json.dumps(protocol, indent=2))
        print(name, n, "hit %.3f D %.3f  %.0fs" %
              (out["hit"].mean(), out["inD"].mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
