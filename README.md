# NeuroDrift

A custom 2D driving environment built with Raylib, where a population of small PyTorch MLPs evolves via a genetic algorithm (neuroevolution) to navigate the track and avoid collisions.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Controls

| Key | Action |
|-----|--------|
| **S** | Toggle sensor rays |
| **C** | Toggle checkpoint debug lines |
| **← / →** | Previous / next track shape (manual) |
| **Space** | Toggle camera follow (zoom on the current leader) |

The left sidebar shows generation, population, track mode, fitness, and a **Brain Cam** (leader network) plus a short legend on the bottom right.
