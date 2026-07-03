import os
os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")
import matplotlib
matplotlib.use("Agg")

from cgols import build_config, plot_paper_slices

config, _ = build_config()
plot_paper_slices(
    config,
    target_times_myr=(10.0, 25.0, 50.0, 60.0),
    out="cgols_paper_slices_512_inj300_noramp.png"
)