# NeuroDrift

A 2D top-down driving environment built with **Raylib**, where a population of small **PyTorch** MLPs evolves through **neuroevolution** (a genetic algorithm) to navigate the track and pass checkpoints. There is no hand-written driver: policies improve generation by generation.

## What you’ll see

- **Many cars** driving at once, each controlled by its own neural network.
- A **HUD** with generation count, population size, track shape index and name, best fitness from the **previous** generation, and steps remaining in the current generation.
- **Body colors** that reflect race progress (gates plus distance); **white** cars are **elites** carried over unchanged; the current **leader** is highlighted in **magenta**; **gray** cars have retired (crashed or stalled).
- **Brain Cam**: a live diagram of the leading car’s network (inputs, hidden layer, and its steer and pedal outputs).
- Optional **sensor rays**, **checkpoint lines**, **camera follow** zoomed on the leader, and the track's **racing line** colored by achievable speed.

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

Car movement uses a **fixed integration step** of **1/60 second** (`FIXED_PHYSICS_DT` in `src/simulation.py`). It does **not** use Raylib’s variable frame delta (`GetFrameTime`) for physics, so each **simulation tick** is identical in length. With the same network weights (inference is deterministic on CPU), an elite car follows the **same trajectory** tick-for-tick across runs. Display FPS can still vary; the sim advances **one physics step per frame** at that fixed dt.

## Controls

| Key | Action |
|-----|--------|
| **S** | Toggle **sensor** rays (9 directions, distance to track boundary) |
| **C** | Toggle **checkpoint** gates (evenly spaced across the track) |
| **← / →** | **Previous / next track** (five circuits, easiest first). Changing track ends the generation and respawns on the new start line; evolved brains carry over, so you can step up through the tracks like a curriculum. |
| **Space** | **Camera follow**: on = zoomed view centered on the fitness **leader**; off = full track view |
| **L** | Toggle the **racing line** (see below) and show the ideal lap time for this car |

Raylib also draws **FPS** in the corner.

## How the agent works

### Observation (10 numbers)

Each brain receives:

1. Nine **normalized** ray distances (0–1 relative to max sensor range), every 22.5° from 90° left to 90° right (`SENSOR_ANGLES` in `src/car.py`). The in-between rays show the shape of a corner, such as a chicane's second apex, before the car is in it.
2. **Normalized speed** (current speed vs the car’s max).

So the input dimension is **10** (`STATE_DIM` in `src/ga_agent.py`, derived from the sensor list).

### Controls (2 analog outputs)

The network outputs two values in **−1…+1** through `tanh` (`CONTROL_LABELS` in `src/car.py`):

- **STEER**: −1 full left, +1 full right. Steering is scaled by forward speed (full turn only when moving fast enough), and turning hard costs a little acceleration and top speed (`TURN_SPEED_PENALTY`).
- **PEDAL**: +1 full throttle, 0 coast, −1 full brake.

The car has **drag proportional to speed**, sized so full throttle still tops out at max speed. Throttle therefore settles at a steady speed (pedal 0.5 holds about half speed). See `Car.apply_action` in `src/car.py`.

Why analog with drag: with discrete argmax actions, and with no drag, any throttle ran the car to top speed, and a small mutation almost never changed which pedal won. Instrumented runs showed the best cars on the technical tracks always hitting the wall at full speed and never braking. With analog outputs and drag, "ease off for this corner" is a small, smooth change that evolution can find.

### Network

Default architecture: **10 → 16 → 16 → 2** with ReLUs and a `tanh` output (`Brain` in `src/ga_agent.py`). Hidden width is `HIDDEN_DIM` (16).

### Evolution

Each **generation**:

1. Cars are scored by **fitness** while they drive (see below).
2. A generation ends after **2500** simulation steps (`MAX_GENERATION_STEPS` in `src/simulation.py`), or earlier once every car has retired (crashed or stalled).
3. Individuals are sorted by fitness. The top **10%** are **elites**: weights unchanged, cars respawn at the track spawn.
4. Everyone else gets a new brain by **copying a random elite parent**. With probability 30% (`crossover_rate`) it is first **crossed** with a second elite: each neuron (weight row plus bias) comes whole from one parent or the other. Then **masked Gaussian mutation** is applied. Roughly **20%** of offspring use a stronger “wildcard” mutation; the rest use a tighter mutation. Non-elites also get a small random **heading jitter** on reset.

Population size defaults to **100** (`POPULATION_SIZE`).

## Fitness and checkpoints (summary)

Fitness is **not** only “go fast”: it is a **strict economy** tuned in `src/simulation.py`:

- **Per-frame cost** so simply existing in the sim always drains fitness.
- **Starvation**: if the car goes too long without clearing the **next** checkpoint in order, extra drain kicks in after a short grace period; past that band the car **retires**.
- **Wall contact**: hitting the boundary costs a one-off `FITNESS_WALL_DRAIN_PER_FRAME` and **retires** the car.
- **Partial gate credit**: when a run ends, the car also gets the checkpoint reward times the fraction of the way it made to its next gate, so selection can tell a car that got deep into a corner from one that crashed at the entry.
- **Retirement**: a retired car stops where it is (drawn gray) and keeps the fitness it had, so its score reflects how far it got. Without this, a crashed car kept draining every frame and parking at the start beat attempting a corner, so populations learned not to move. When every car has retired the generation ends early.
- **Rewards** come from advancing along **checkpoints in race order**. You must cross the next gate with enough **forward motion aligned** with the track direction; multiple gate hits per frame are capped. Checkpoint **score** is granted on **every** gate crossing (`CHECKPOINT_FITNESS_INTERVAL` = 1; larger values batch it, which stalls learning when the first payout sits past a hard corner); completing a full lap adds a **lap bonus**.

Exact numbers (costs, rewards, grace frames, etc.) live next to the `FITNESS_*` and `CHECKPOINT_*` constants at the top of `src/simulation.py`.

## Track

Five circuits are available, easiest first; the index and name are shown in the HUD as “Track Mode”. The sim starts on **0 (Speedway)**.

| # | Track | Character |
|---|-------|-----------|
| 0 | **Speedway** | Four flat-out bankings: learn to steer before learning to brake. |
| 1 | **Grand Prix** | Every corner type: heavy braking into T1, esses, a hairpin, a chicane. |
| 2 | **High Speed** | Long full-throttle runs broken by two chicanes, fast sweepers between. |
| 3 | **Technical** | Esses, a double hairpin, then a long double-apex sweeper. |
| 4 | **Street Circuit** | Narrow and walled; almost every corner is a slow 90. |

Each circuit is laid out the way real tracks are designed: a polygon whose corners are rounded with circular arcs (`LAYOUTS` in `src/track.py`; each corner is `x, y, corner radius, track width`). Corner radii are sized against the car's handling (about 111 px turning radius flat out, 40 px at a crawl), so R ≥ ~110 is flat and R ≈ 45–75 needs a braking zone. Track width blends between corners and runs wider into heavy braking zones. Construction rejects corners whose radius doesn't fit the edge or is tighter than half the track width.

Checkpoints are gates perpendicular to the centerline, evenly spaced along it, starting at the start/finish line on the main straight (the straight after the last corner). The surface is drawn with asphalt, red/white kerbs, and a checkered start line; walls are simplified from the offset curves for collision and sensors. Sensors and collision only test boundary segments in nearby cells of a uniform grid (`Track.segments_near`).

### Racing line

Press **L** to draw each track's **minimum-curvature racing line**: the smoothest path that stays inside the walls with room for the car's width, found by relaxing offsets along the centerline (`Track.racing_line`). This is the classic geometric optimum: wide on entry, clip the apex, wide on exit. It is colored by the fastest speed this car can carry there (cornering limit from its turn rate, then acceleration and braking passes), red where it must slow down and green where it can stay flat out, and the HUD shows the resulting ideal lap time. It is drawn for reference only; the cars never see it.

Wall and gate tests use double-precision segment intersection (`segment_hit` in `src/track.py`). raylib's float32 `CheckCollisionLines` missed crossings when a car drove almost exactly along an axis, which silently dropped gates and blinded sensors.

Collision uses the car as a **rotated box** against boundary segments (`Track.check_car_collision`). On impact the sim **reverts** the car to its pre-step position, stops it, charges the wall cost (`FITNESS_WALL_DRAIN_PER_FRAME` in `src/simulation.py`) and retires it for the rest of the generation.

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
    ├── track.py           # Circuit layouts, checkpoints, collision, drawing
    └── ga_agent.py        # Brain MLP, population, mutation, evolve
```

## Tuning

To experiment, edit the constants at the top of `src/simulation.py` (fitness, steps per generation, population size) and `src/ga_agent.py` (network size, elite fraction, mutation rates). Car handling defaults are constructor arguments on `Car` and are passed from `Simulation` via `car_kwargs`.
