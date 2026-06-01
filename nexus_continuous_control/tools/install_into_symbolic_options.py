"""Install this extension into a checked-out `remunds/symbolic_options` repo.

Usage:
  python tools/install_into_symbolic_options.py /path/to/symbolic_options

The script copies:
  * the `nexus_continuous` package into `src/nexus_continuous`
  * a thin train wrapper to `src/symbolic_options/hierarchical_ac_pqn_playground.py`
  * config files to `src/symbolic_options/config/alg/`
  * re-export policy modules to `src/symbolic_options/reward_functions/playground/`
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    repo = Path(sys.argv[1]).resolve()
    if not (repo / "src" / "symbolic_options").exists():
        raise SystemExit(f"Not a symbolic_options checkout: {repo}")

    pkg_dst = repo / "src" / "nexus_continuous"
    copytree(ROOT / "nexus_continuous", pkg_dst)

    wrapper = repo / "src" / "symbolic_options" / "hierarchical_ac_pqn_playground.py"
    wrapper.write_text(
        '"""Thin wrapper for the continuous-control NEXUS extension."""\n\n'
        'from __future__ import annotations\n\n'
        'import sys\n'
        'from pathlib import Path\n\n'
        'SRC_DIR = Path(__file__).resolve().parents[1]\n'
        'if str(SRC_DIR) not in sys.path:\n'
        '    sys.path.insert(0, str(SRC_DIR))\n\n'
        'from nexus_continuous.scripts.train_nexus_playground import main\n\n'
        "if __name__ == '__main__':\n"
        '    main()\n',
        encoding="utf-8",
    )

    cfg_dst = repo / "src" / "symbolic_options" / "config" / "alg"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for cfg in (ROOT / "configs").glob("*.yaml"):
        shutil.copy2(cfg, cfg_dst / f"nexus_playground_{cfg.name}")

    reward_dst = repo / "src" / "symbolic_options" / "reward_functions" / "playground"
    reward_dst.mkdir(parents=True, exist_ok=True)
    (reward_dst / "__init__.py").write_text("", encoding="utf-8")
    for name in [
        "cartpole_balance",
        "cheetah_run",
        "walker_walk",
        "hopper_hop",
        "panda_pick_cube",
        "go1_joystick",
    ]:
        (reward_dst / f"{name}.py").write_text(
            f"from nexus_continuous.policies.{name} import *  # noqa: F401,F403\n",
            encoding="utf-8",
        )

    print(f"Installed continuous-control NEXUS extension into {repo}")
    print("Example:")
    print("  cd", repo)
    print(
        "  uv run python src/symbolic_options/hierarchical_ac_pqn_playground.py "
        "--config src/symbolic_options/config/alg/nexus_playground_cartpole_balance_nesy.yaml"
    )


if __name__ == "__main__":
    main()
