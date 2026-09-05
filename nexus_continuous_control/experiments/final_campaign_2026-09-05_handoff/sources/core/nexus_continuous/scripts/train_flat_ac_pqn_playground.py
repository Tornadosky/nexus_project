"""CLI for the flat continuous-control AC-PQN baseline."""

from __future__ import annotations

from nexus_continuous.scripts.train_nexus_playground import main as nexus_main


def main(argv: list[str] | None = None) -> None:
    nexus_main(argv)


if __name__ == "__main__":
    main()
