"""
Compare the frames of the run_horeka_fv_validate.sh correctness runs.

Arguments are four run tags: PLM unsharded, PLM sharded, PPM unsharded, PPM
sharded. Prints, at the last common frame, the relative difference of the
frame fields sharded vs unsharded (should be rounding-level, growing slowly
with time) next to PPM vs PLM (a genuine change of scheme, for scale).

    python cgols_fv_validate_compare.py _fvval_v128plm_s11 _fvval_v128plm_s22 \\
        _fvval_v128ppm_s11 _fvval_v128ppm_s22
"""

# general
import os
import sys

# numerics
import numpy as np

FIELDS = ("n_xz", "T_xz", "n_xy", "vr_xz", "mdot_z")


def last_frames(tags):
    """Last frame index present in every run, and its file per run."""
    names = [set(f for f in os.listdir(f"cgols_snapshots{tag}") if f.startswith("frame_")) for tag in tags]
    last = sorted(set.intersection(*names))[-1]
    return last, [np.load(os.path.join(f"cgols_snapshots{tag}", last)) for tag in tags]


def rel(a, b, field):
    """median / 99th pct / max of |a - b| / scale, scale = |b| (or max|b| for signed fields)."""
    a, b = a[field].astype(np.float64), b[field].astype(np.float64)
    scale = np.abs(b) if field in ("n_xz", "T_xz", "n_xy") else np.max(np.abs(b))
    d = np.abs(a - b) / np.maximum(scale, 1e-300)
    return np.median(d), np.percentile(d, 99), d.max(), bool(np.all(np.isfinite(a)))


def main():
    tags = sys.argv[1:5]
    last, (plm1, plm4, ppm1, ppm4) = last_frames(tags)
    print(f"frame {last}, t = {float(plm1['time_myr']):.2f} Myr")
    print(f"{'pair':22s} {'field':7s} {'median':>9s} {'p99':>9s} {'max':>9s} finite")
    for label, a, b in (("PLM sharded/unsharded", plm4, plm1), ("PPM sharded/unsharded", ppm4, ppm1),
                        ("PPM/PLM (scale)", ppm1, plm1)):
        for field in FIELDS:
            med, p99, mx, finite = rel(a, b, field)
            print(f"{label:22s} {field:7s} {med:9.1e} {p99:9.1e} {mx:9.1e} {finite}")


if __name__ == "__main__":
    main()
