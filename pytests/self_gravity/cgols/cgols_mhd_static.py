"""
Static (no-wind) equilibrium check of the CGOLS initial conditions, MHD or hydro.

Run cgols.py with CGOLS_WIND=0 first (e.g. tags _A128mhdstatic / _A128static),
then this script with the same CGOLS_* knobs. It rebuilds the IC in-process,
loads the final state, and prints

  * the rotation added by the toroidal field's hoop stress (MHD only):
    max and disk-mean of v_phi - sqrt(v_phi^2 - B_phi^2 / rho);
  * the drift of the disk: the density-weighted rms height of the disk gas
    (R < 4 kpc, |z| < 2 kpc, n > 0.1 cm^-3) from every frame and from the
    initial / final states, and the disk rms v_z at the end;
  * the field: E_B and the vertical field fraction in the disk (MHD).

A buoyant (Parker) instability of the magnetised disk would show up as growing
disk v_z, a growing scale height and a rising vertical field fraction.
CPU only:

    JAX_PLATFORMS=cpu CGOLS_DIM=128 CGOLS_MHD=1 CGOLS_WIND=0 \
        CGOLS_END_FRACTION=0.26666667 CGOLS_RUN_TAG=_A128mhdstatic \
        python cgols_mhd_static.py
"""

# general
import os

os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# cgols driver
from cgols import (
    MHD,
    RUN_TAG,
    SNAPSHOTS_DIR,
    _code_field_to_microgauss,
    _data_final,
    _snapshot_grid,
    build_config,
    build_initial_conditions,
    code_length,
    code_mass,
    code_velocity,
    load_snapshots,
    m_p,
    mu,
    u,
)

N_DISK_MIN = 0.1  # cm^-3, the disk-gas threshold of the frame-based height


def slice_scale_height(n_xz, x, z):
    """Density-weighted rms |z| of the disk gas in one edge-on slice (pc)."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    disk = (np.abs(X) < 4.0) & (np.abs(Z) < 2.0) & (n_xz > N_DISK_MIN)
    weight = np.where(disk, n_xz, 0.0)
    return 1e3 * float(np.sqrt((weight * Z**2).sum() / weight.sum()))


def state_disk_statistics(state, registered_variables, coordinates, n_factor):
    """(rms height [pc], rms v_z [km/s]) of the disk gas in a 3D state."""
    X, Y, Z = coordinates
    rho = np.asarray(state[registered_variables.density_index])
    v_z = np.asarray(state[registered_variables.velocity_index.z])
    disk = (np.sqrt(X**2 + Y**2) < 4.0) & (np.abs(Z) < 2.0) & (rho * n_factor > N_DISK_MIN)
    weight = np.where(disk, rho, 0.0)
    height = 1e3 * float(np.sqrt((weight * Z**2).sum() / weight.sum()))
    rms_v_z = float(np.sqrt((weight * v_z**2).sum() / weight.sum()))
    return height, rms_v_z * code_velocity.to(u.km / u.s).value


def main():
    """Rebuild the IC, load the final state and the frames, print the checks."""
    config, registered_variables = build_config()
    initial_state, config, _, registered_variables = build_initial_conditions(
        save_potential=False
    )
    final_state = jnp.asarray(np.load(_data_final(f"cgols_final_state{RUN_TAG}.npy")))
    x, y, z, _, _ = _snapshot_grid(config)
    coordinates = np.meshgrid(x, y, z, indexing="ij")
    n_factor = float(
        (code_mass / code_length**3).to(u.g / u.cm**3).value / (mu * m_p.to(u.g).value)
    )
    km_s = code_velocity.to(u.km / u.s).value

    print(f"Static check {RUN_TAG} (MHD={MHD})")
    if MHD:
        rho = np.asarray(initial_state[registered_variables.density_index])
        v_x = np.asarray(initial_state[registered_variables.velocity_index.x])
        v_y = np.asarray(initial_state[registered_variables.velocity_index.y])
        B_x = np.asarray(initial_state[registered_variables.magnetic_index.x])
        B_y = np.asarray(initial_state[registered_variables.magnetic_index.y])
        v_phi = np.sqrt(v_x**2 + v_y**2)
        hoop_free = np.sqrt(np.maximum(v_phi**2 - (B_x**2 + B_y**2) / rho, 0.0))
        delta = (v_phi - hoop_free) * km_s
        disk = rho * n_factor > 1.0
        print(f"  hoop-stress rotation: max dv_phi = {delta.max():.2f} km/s, "
              f"mean over n > 1 cm^-3 = {delta[disk].mean():.2f} km/s "
              f"(max v_phi {v_phi.max() * km_s:.1f} km/s)")

    for label, state in (("initial", initial_state), ("final", final_state)):
        height, rms_v_z = state_disk_statistics(state, registered_variables, coordinates, n_factor)
        line = f"  {label:7s}: disk rms height {height:6.1f} pc, disk rms v_z {rms_v_z:6.2f} km/s"
        if MHD:
            fields = [np.asarray(state[registered_variables.magnetic_index[i]]) for i in range(3)]
            energy = 0.5 * sum(f**2 for f in fields)
            disk = np.abs(coordinates[2]) < 0.3
            vertical = float((fields[2] ** 2)[disk].sum() / (2 * energy[disk]).sum())
            line += (f", E_B {energy.sum() * config.grid_spacing**3:.4e}, "
                     f"max|B| {np.sqrt(2 * energy.max()) * _code_field_to_microgauss():.1f} uG, "
                     f"disk Bz^2/B^2 {vertical:.3f}")
        print(line)

    frames = load_snapshots(SNAPSHOTS_DIR)
    heights = [slice_scale_height(n, x, z) for n in frames["n_xz"]]
    print("  frame rms height [pc] (y=0 slice): "
          + ", ".join(f"{t:.0f} Myr {h:.0f}" for t, h in zip(frames["time_myr"], heights)))


if __name__ == "__main__":
    main()
