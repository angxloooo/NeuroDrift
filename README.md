# NeuroDrift

A 2D top-down driving environment built with **Raylib**, where a population of small **PyTorch** MLPs evolves through **neuroevolution** (a genetic algorithm) to navigate the track and pass checkpoints. There is no hand-written driver: policies improve generation by generation.

## What you’ll see

- **Many cars** driving at once, each controlled by its own neural network.
- A **HUD** with generation count, population size, track shape index, best fitness from the **previous** generation, and steps remaining in the current generation.
- **Body colors** that reflect race progress (gates plus distance); **white** cars are **elites** carried over unchanged; the current **leader** is highlighted in **magenta**.
- **Brain Cam**: a live diagram of the leading car’s network (inputs, hidden layer, argmax action).
- Optional **sensor rays**, **checkpoint lines**, and **camera follow** zoomed on the leader.

## Stack

| Piece | Role |
|-------|------|
| Python 3 | Runtime |
| [`raylib`](https://www.raylib.com/) (Python bindings) | Window, input, 2D drawing, line intersection for sensors and checkpoints |
| PyTorch | MLP weights and forward passes |

Dependencies are listed in `requirements.txt` (`raylib`, `torch`).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use a recent Python 3 (3.10+ recommended). A GPU is optional; inference runs on the CPU by default.

## Run

```bash
python3 main.py
```

Entry point: `main.py` → `Simulation.run()` in `src/simulation.py`.

## Physics timestep (reproducible trajectories)

Car movement uses a **fixed integration step** of **1/60 second** (`FIXED_PHYSICS_DT` in `src/simulation.py`). It does **not** use Raylib’s variable frame delta (`GetFrameTime`) for physics, so each **simulation tick** is identical in length. With the same network weights and the same sequence of actions (argmax is deterministic on CPU), an elite car follows the **same trajectory** tick-for-tick across runs. Display FPS can still vary; the sim advances **one physics step per frame** at that fixed dt.

## Controls

| Key | Action |
|-----|--------|
| **S** | Toggle **sensor** rays (5 directions, distance to track boundary) |
| **C** | Toggle **checkpoint** segments (radial gates around the track) |
| **← / →** | **Previous / next track shape** (three procedural layouts). Changing track re-runs evolution from the current spawn. |
| **Space** | **Camera follow**: on = zoomed view centered on the fitness **leader**; off = full track view |

Raylib also draws **FPS** in the corner.

## How the agent works

### Observation (6 numbers)

Each brain receives:

1. Five **normalized** ray distances (0–1 relative to max sensor range).
2. **Normalized speed** (current speed vs the car’s max).

So the input dimension is **6** (`STATE_DIM` in `src/ga_agent.py`).

### Actions (6 discrete)

The network outputs **logits** for **6** actions: combine **accelerate or brake** with **turn left, straight, or right**. Steering is scaled by forward speed (you get full turn only when moving fast enough). See `Car.apply_action` in `src/car.py`.

### Network

Default architecture: **6 → 16 → 16 → 6** with ReLUs (`Brain` in `src/ga_agent.py`). Hidden width is `HIDDEN_DIM` (16).

### Evolution

Each **generation**:

1. Cars are scored by **fitness** while they drive (see below).
2. A generation ends after exactly **2500** simulation steps (`MAX_GENERATION_STEPS` in `src/simulation.py`).
3. Individuals are sorted by fitness. The top **10%** are **elites**: weights unchanged, cars respawn at the track spawn.
4. Everyone else gets a new brain by **copying a random elite parent** and applying **masked Gaussian mutation**. Roughly **20%** of offspring use a stronger “wildcard” mutation; the rest use a tighter mutation. Non-elites also get a small random **heading jitter** on reset.

Population size defaults to **100** (`POPULATION_SIZE`).

## Fitness and checkpoints (summary)

Fitness is **not** only “go fast”: it is a **strict economy** tuned in `src/simulation.py`:

- **Per-frame cost** so simply existing in the sim always drains fitness.
- **Starvation**: if the car goes too long without clearing the **next** checkpoint in order, extra drain kicks in after a short grace period (tiered).
- **Wall contact**: per-frame fitness drain for each frame the car stays touching the boundary (`FITNESS_WALL_DRAIN_PER_FRAME`).
- **Rewards** come from advancing along **checkpoints in CCW order**. You must cross the next gate with enough **forward motion aligned** with the track direction; multiple gate hits per frame are capped. Checkpoint **score** is granted in **batches** every **20** gate crossings (`CHECKPOINT_FITNESS_INTERVAL`); completing a full lap adds a **lap bonus**.

Exact numbers (costs, rewards, grace frames, etc.) live next to the `FITNESS_*` and `CHECKPOINT_*` constants at the top of `src/simulation.py`.

## Track

The track is a **ring** between an outer and inner boundary (ellipses approximated by segments). **Checkpoints** are radial segments at regular angular steps. Three **shape presets** warp the boundary (sinusoidal radial offsets) for variety; index **0–2** is shown in the HUD as “Track Mode”.

Collision uses the car as a **rotated box** against boundary segments (`Track.check_car_collision`). On impact the sim **reverts** the car to its pre-step position, **halves velocity** (× 0.5), and **fitness** takes the configured **per-frame wall drain** for as long as the car stays in contact (`FITNESS_WALL_DRAIN_PER_FRAME` in `src/simulation.py`).

## Project layout

```
NeuroDrift/
├── main.py              # Entry point
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py      # Package exports
    ├── simulation.py    # Main loop, fitness, UI, Brain Cam, camera
    ├── car.py             # Physics, sensors, rendering
    ├── track.py           # Geometry, checkpoints, collision, drawing
    └── ga_agent.py        # Brain MLP, population, mutation, evolve
```

## Tuning

To experiment, edit the constants at the top of `src/simulation.py` (fitness, steps per generation, population size) and `src/ga_agent.py` (network size, elite fraction, mutation rates). Car handling defaults are constructor arguments on `Car` and are passed from `Simulation` via `car_kwargs`.
