#!/usr/bin/env python3
"""NeuroDrift - 2D autonomous driving RL simulation entry point."""

from src.simulation import Simulation


def main() -> None:
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
