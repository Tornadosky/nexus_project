"""Is `base_height` the right quantity for the go1 rules on ROUGH terrain?

`tools/audit_semantics.py` verifies that `base_height` is wired to the trunk's world z, and it
passes on Go1JoystickRoughTerrain. That is a DOF-level check: right joint, right sign. It
cannot see a frame-level error, and the go1 rules are frame-sensitive:

    go1_joystick.py:97   recover admissible iff height < 0.28 | |roll| > 0.25 | |pitch| > 0.25
    go1_joystick.py:117  not_fallen        iff height > 0.22 & |roll| < 0.6  & |pitch| < 0.6
    go1_joystick.py:77   height_reward     = 1 - |height - 0.32|

On flat terrain the ground is z=0, so trunk world z == height above ground and the thresholds
mean what they say. On rough terrain the trunk's world z is (terrain elevation under the robot)
+ (clearance above it), so an upright robot on a bump reads "tall" and an upright robot in a
trough reads "fallen". The nominal standing clearance is ~0.30 against a 0.28 threshold — a
2 cm margin. This script measures the terrain relief against that margin.
"""

from __future__ import annotations

import numpy as np

from mujoco_playground import registry


def main() -> int:
    for env_name in ("Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"):
        cfg = registry.get_default_config(env_name)
        cfg.impl = "jax"
        env = registry.load(env_name, cfg)
        model = env.mj_model
        print(f"\n=== {env_name} ===")
        nhf = int(model.nhfield)
        print(f"  heightfields: {nhf}")
        for i in range(nhf):
            # hfield_size = (radius_x, radius_y, elevation_z, base_z); hfield_data in [0,1]
            size = np.asarray(model.hfield_size[i])
            nrow = int(model.hfield_nrow[i])
            ncol = int(model.hfield_ncol[i])
            adr = int(model.hfield_adr[i])
            data = np.asarray(model.hfield_data[adr : adr + nrow * ncol])
            relief = float(size[2]) * (data.max() - data.min())
            print(f"  hfield[{i}] {nrow}x{ncol} size={size}")
            print(
                f"    normalized data range [{data.min():.3f}, {data.max():.3f}] "
                f"-> vertical relief {relief:.4f} m  (z-scale {float(size[2]):.4f} m)"
            )
            print(f"    std of surface {float(size[2]) * data.std():.4f} m")
    print(
        "\nRule margins for reference: recover fires below 0.28 m, not_fallen requires "
        "> 0.22 m,\nheight_reward peaks at 0.32 m. Nominal standing trunk z is ~0.30 m."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
