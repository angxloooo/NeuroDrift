"""Simulation module - main game loop and neuroevolution environment glue."""

import colorsys
import math

import torch
from raylib import ffi, rl, colors

from .car import CONTROL_LABELS, SENSOR_ANGLES, SENSOR_LABELS, Car
from .ga_agent import ACTION_DIM, HIDDEN_DIM, PopulationManager
from .track import Track, segment_hit


SCREEN_WIDTH = 1450
SCREEN_HEIGHT = 800
FPS = 60
# Car physics always integrates with this fixed step (never frame delta) so trajectories
# match a fixed number of simulation ticks—same weights → same path across runs.
FIXED_PHYSICS_DT = 1.0 / 60.0
SENSOR_RANGE = 200.0
MAX_SPEED = 250.0
MAX_GENERATION_STEPS = 2500
POPULATION_SIZE = 100

# --- Strict fitness economy (only checkpoint/lap gains; heavy drains elsewhere) ---
# A car retires (frozen, fitness kept) when it hits a wall or stalls past the starvation band.
# Without this, a crashed car keeps draining every frame, so parking at the start beats trying
# a corner and the population converges on not moving.
FITNESS_CHECKPOINT_REWARD = 130.0
FITNESS_LAP_REWARD = 380.0
# Small tax every frame so “existing” in the sim is never free.
FITNESS_BASE_COST_PER_FRAME = 1.25
# No checkpoint for this long → extra drain per frame (on top of base cost); past the band → retire.
STARVATION_GRACE_FRAMES = 30
STARVATION_DRAIN_TIER1 = 38.0  # frames in (grace, grace + 60]
STARVATION_TIER1_WIDTH = 60
STALL_RETIRE_FRAMES = STARVATION_GRACE_FRAMES + STARVATION_TIER1_WIDTH
# One-off cost for the frame a car hits the wall (and retires).
FITNESS_WALL_DRAIN_PER_FRAME = 125.0
# When a run ends, credit the fraction of the way to the next gate (x checkpoint reward) so
# selection can tell a car that got deep into a corner from one that crashed at its entry.

# Checkpoint gates: require motion aligned with the race direction; cap multi-hits / frame.
CHECKPOINT_MIN_FORWARD_FRAC = 0.28  # dot(move, tangent) >= frac * |move|
CHECKPOINT_MAX_AWARDS_PER_FRAME = 2
# Fitness only every N sequential gate crossings (still advance target every crossing).
# Keep this small: if the first payout lies past a hard corner (e.g. Grand Prix T1 at gate ~6),
# cars that make the corner score the same as cars that crash and evolution stalls.
CHECKPOINT_FITNESS_INTERVAL = 1

# Body color: gates dominate ordering; path length updates every frame for smooth hue changes.
COLOR_PROGRESS_GATE_WEIGHT = 10000.0

_ELITE_COLOR = getattr(colors, "WHITE", (255, 255, 255, 255))
_RETIRED_COLOR = (110, 110, 110, 255)
FITNESS_HUE_MAX = 0.65


def _starvation_extra_per_frame(frames_since_checkpoint: int) -> float:
    """Penalty for not hitting the next checkpoint; 0 during grace (car retires after the band)."""
    if frames_since_checkpoint <= STARVATION_GRACE_FRAMES:
        return 0.0
    return STARVATION_DRAIN_TIER1


def _car_color_progress_score(car: Car) -> float:
    """Monotonic race-ish signal: discrete gates + continuous odometer (updates each frame)."""
    return float(car.total_gates_crossed) * COLOR_PROGRESS_GATE_WEIGHT + car.total_distance_traveled


def _credit_partial_gate(car: Car, track: Track) -> None:
    car.fitness += FITNESS_CHECKPOINT_REWARD * track.gate_progress(car.position, car.target_checkpoint)


def _apply_strict_fitness_penalties(car: Car, *, touching_wall: bool) -> None:
    car.fitness -= FITNESS_BASE_COST_PER_FRAME
    car.fitness -= _starvation_extra_per_frame(car.frames_since_checkpoint)
    if touching_wall:
        car.fitness -= FITNESS_WALL_DRAIN_PER_FRAME


def get_fitness_color(
    value: float, min_value: float, max_value: float
) -> tuple[int, int, int, int]:
    """Map a scalar (e.g. fitness or track progress) to HSV red (low) -> blue ~0.65."""
    if max_value == min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)
    t = max(0.0, min(1.0, t))

    hue = t * FITNESS_HUE_MAX
    rf, gf, bf = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    r = int(rf * 255)
    g = int(gf * 255)
    b = int(bf * 255)
    # raylib-py has no rl.Color(); (r, g, b, a) is accepted as Color by DrawRectanglePro.
    return (r, g, b, 255)


def _draw_sidebar_separator(x: int, y: int, width: int) -> None:
    rl.DrawLine(x, y, x + width, y, colors.GRAY)


def _draw_fitness_gradient_block(x: int, y: int, bar_w: int = 280) -> int:
    """Draw HSV gradient legend for car body colors (track progress, not net fitness)."""
    rl.DrawText(
        b"Car colors: behind->ahead (gates + path length; updates every frame)",
        x,
        y,
        12,
        colors.LIGHTGRAY,
    )
    bar_y = y + 22
    bar_h = 14
    for i in range(bar_w):
        t = i / max(1, bar_w - 1)
        hue = t * FITNESS_HUE_MAX
        rf, gf, bf = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        seg = (int(rf * 255), int(gf * 255), int(bf * 255), 255)
        rl.DrawRectangle(x + i, bar_y, 1, bar_h, seg)
    rl.DrawRectangleLines(x, bar_y, bar_w, bar_h, colors.RAYWHITE)
    sub_y = bar_y + bar_h + 6
    rl.DrawText(b"Fewer gates", x, sub_y, 11, colors.WHITE)
    rl.DrawText(
        b"More gates",
        x + bar_w - 65,
        sub_y,
        11,
        colors.WHITE,
    )
    return sub_y + 28


def _draw_ui_sidebar(
    *,
    generation: int,
    population_size: int,
    track_shape: int,
    track_name: str,
    best_fitness: float,
    steps_left: int,
    max_steps: int,
    show_sensors: bool,
    show_checkpoints: bool,
    camera_follow: bool,
    ideal_lap: float | None,
) -> None:
    """Left column UI: title, stats, controls, color legend."""
    x = 30
    sep_w = 320
    y = 40

    rl.DrawText(b"NEURODRIFT AI", x, y, 30, colors.SKYBLUE)
    y += 45
    _draw_sidebar_separator(x, y, sep_w)
    y += 35

    rl.DrawText(f"Generation: {generation}".encode(), x, y, 18, colors.RAYWHITE)
    y += 35
    rl.DrawText(
        f"Population: {population_size}".encode(),
        x,
        y,
        18,
        colors.RAYWHITE,
    )
    y += 35
    rl.DrawText(
        f"Track Mode: {track_shape} ({track_name})".encode(),
        x,
        y,
        18,
        colors.RAYWHITE,
    )
    y += 35
    rl.DrawText(
        f"Best Fitness (Last Gen): {best_fitness:.0f}".encode(),
        x,
        y,
        18,
        colors.RAYWHITE,
    )
    y += 35
    rl.DrawText(
        f"Steps Left: {steps_left} / {max_steps}".encode(),
        x,
        y,
        18,
        colors.RAYWHITE,
    )
    y += 35
    _draw_sidebar_separator(x, y, sep_w)
    y += 35

    rl.DrawText(b"Controls:", x, y, 18, colors.SKYBLUE)
    y += 35
    s_on = "ON" if show_sensors else "OFF"
    rl.DrawText(f"[S] - Toggle Sensors ({s_on})".encode(), x, y, 16, colors.RAYWHITE)
    y += 35
    c_on = "ON" if show_checkpoints else "OFF"
    rl.DrawText(f"[C] - Toggle Checkpoints ({c_on})".encode(), x, y, 16, colors.RAYWHITE)
    y += 35
    rl.DrawText(
        "Left / Right Arrow - Prev/Next Track".encode(),
        x,
        y,
        16,
        colors.RAYWHITE,
    )
    y += 35
    cf_on = "ON" if camera_follow else "OFF"
    rl.DrawText(f"[Space] - Camera Follow ({cf_on})".encode(), x, y, 16, colors.RAYWHITE)
    y += 35
    rl_on = f"ON, ideal lap {ideal_lap:.1f}s" if ideal_lap is not None else "OFF"
    rl.DrawText(f"[L] - Racing Line ({rl_on})".encode(), x, y, 16, colors.RAYWHITE)
    y += 35
    _draw_sidebar_separator(x, y, sep_w)
    y += 35

    rl.DrawText(b"Colors:", x, y, 18, colors.SKYBLUE)
    y += 35
    rl.DrawText(b"White - Elite Carryover", x, y, 16, _ELITE_COLOR)
    y += 35
    y = _draw_fitness_gradient_block(x, y)
    rl.DrawFPS(x, SCREEN_HEIGHT - 30)


def _draw_brain_cam_key() -> None:
    """Bottom-right Brain Cam legend; matches left sidebar typography (header + body)."""
    margin = 20
    line_step = 32
    entries: list[tuple[bytes, int, object]] = [
        (b"Brain Cam Key", 18, colors.SKYBLUE),
        (b"Yellow: Inputs (Sensors + Speed)", 16, colors.RAYWHITE),
        (f"  L90 ... R90 = {len(SENSOR_ANGLES)} Sensors (deg off nose)".encode(), 16, colors.RAYWHITE),
        (b"  SPD = Speed (Normalized)", 16, colors.RAYWHITE),
        (f"Blue: Hidden Layer ({HIDDEN_DIM} Neurons)".encode(), 16, colors.RAYWHITE),
        (b"Outputs: STEER (- left / + right)", 16, colors.RAYWHITE),
        (b"  PEDAL (- brake / 0 coast / + gas)", 16, colors.RAYWHITE),
        (b"Green = positive, Red = negative", 16, colors.RAYWHITE),
    ]
    total_h = line_step * len(entries)
    y = SCREEN_HEIGHT - margin - total_h
    for text, font, color in entries:
        w = rl.MeasureText(text, font)
        x = SCREEN_WIDTH - margin - w
        rl.DrawText(text, x, y, font, color)
        y += line_step


def _process_checkpoint_crossing(
    car: Car,
    track: Track,
    old_x: float,
    old_y: float,
) -> None:
    """Award when movement crosses the next gate in race order (not grazing / wrong-way / hub chaining)."""
    n_ck = len(track.checkpoints)
    if n_ck == 0:
        return
    dx = float(car.position[0]) - old_x
    dy = float(car.position[1]) - old_y
    move_len_sq = dx * dx + dy * dy
    if move_len_sq < 1e-12:
        return
    move_len = math.sqrt(move_len_sq)
    min_forward = CHECKPOINT_MIN_FORWARD_FRAC * move_len

    move_start = (old_x, old_y)
    move_end = (float(car.position[0]), float(car.position[1]))
    awards = 0
    while awards < CHECKPOINT_MAX_AWARDS_PER_FRAME:
        p1, p2 = track.checkpoints[car.target_checkpoint]
        if segment_hit(move_start, move_end, p1, p2) < 0.0:
            break
        tx, ty = track.checkpoint_tangents[car.target_checkpoint]
        if (dx * tx + dy * ty) < min_forward:
            break
        car.frames_since_checkpoint = 0
        car.target_checkpoint += 1
        awards += 1
        car.checkpoints_since_fitness_reward += 1
        car.total_gates_crossed += 1
        if car.checkpoints_since_fitness_reward >= CHECKPOINT_FITNESS_INTERVAL:
            car.fitness += FITNESS_CHECKPOINT_REWARD * CHECKPOINT_FITNESS_INTERVAL
            car.checkpoints_since_fitness_reward = 0
        if car.target_checkpoint >= n_ck:
            car.target_checkpoint = 0
            car.fitness += FITNESS_LAP_REWARD


class Simulation:
    """Main simulation loop: multi-car population and genetic evolution."""

    def __init__(self):
        self.track = Track()
        spawn_pos, spawn_angle = self.track.get_spawn_info(self.track.spawn_checkpoint)
        car_kwargs = {
            "max_speed": MAX_SPEED,
            "sensor_range": SENSOR_RANGE,
            "spawn_target_checkpoint": self.track.spawn_checkpoint,
        }
        self.population = PopulationManager(
            population_size=POPULATION_SIZE,
            elite_fraction=0.1,
            spawn_position=spawn_pos,
            spawn_angle=spawn_angle,
            car_kwargs=car_kwargs,
        )
        self.generation = 1
        self.last_gen_best_fitness = 0.0
        self.show_sensors = True
        self.generation_step = 0
        self.camera_follow = False
        self.show_racing_line = False

    def _state_from_distances(
        self, sensor_distances: list[float], car: Car
    ) -> list[float]:
        normalized_sensors = [d / SENSOR_RANGE for d in sensor_distances]
        denom = max(car.base_max_speed, 1e-6)
        normalized_speed = car.velocity / denom
        return normalized_sensors + [normalized_speed]

    def _draw_brain_cam(self, car: Car, brain, start_x: int, start_y: int) -> None:
        """Mini network diagram: inputs, hidden layer, outputs for one car/brain pair."""

        def _fade(c, alpha: float):
            a = max(0.0, min(1.0, alpha))
            if hasattr(rl, "Fade"):
                return rl.Fade(c, float(a))
            return rl.ColorAlpha(c, int(255 * a))

        activations = [1.0 - (d / car.sensor_range) for d in car.last_sensor_distances]
        activations.append(car.velocity / max(car.base_max_speed, 1e-6))

        state = self._state_from_distances(car.last_sensor_distances, car)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            controls = brain(state_tensor)[0].tolist()

        layer_x = [start_x, start_x + 90, start_x + 180]
        hidden_step = 250 // max(1, HIDDEN_DIM - 1)
        edge_color = _fade(colors.WHITE, 0.28)
        n_in = len(activations)
        # All three columns share the hidden column's top and height.
        column_top = start_y - 45
        column_h = hidden_step * (HIDDEN_DIM - 1)
        input_step = column_h / max(1, n_in - 1)
        # Few analog outputs: spread them evenly inside the column rather than pinned to its ends.
        output_step = column_h / (ACTION_DIM + 1)

        for i in range(n_in):
            iy = column_top + i * input_step
            for h in range(HIDDEN_DIM):
                hy = column_top + h * hidden_step
                rl.DrawLine(
                    int(layer_x[0]),
                    int(iy),
                    int(layer_x[1]),
                    int(hy),
                    edge_color,
                )

        for h in range(HIDDEN_DIM):
            hy = column_top + h * hidden_step
            for o in range(ACTION_DIM):
                oy = column_top + (o + 1) * output_step
                rl.DrawLine(
                    int(layer_x[1]),
                    int(hy),
                    int(layer_x[2]),
                    int(oy),
                    edge_color,
                )

        input_labels = SENSOR_LABELS + [b"SPD"]
        output_labels = CONTROL_LABELS

        for i in range(n_in):
            iy = column_top + i * input_step
            act = max(0.0, min(1.0, activations[i]))
            color = _fade(colors.YELLOW, act) if act > 0.1 else colors.DARKGRAY
            rl.DrawCircle(int(layer_x[0]), int(iy), 9, color)
            rl.DrawCircleLines(int(layer_x[0]), int(iy), 9, colors.LIGHTGRAY)
            rl.DrawText(input_labels[i], int(layer_x[0]) - 35, int(iy) - 5, 10, colors.LIGHTGRAY)

        avg_act = sum(activations) / len(activations)
        for h in range(HIDDEN_DIM):
            hy = column_top + h * hidden_step
            rl.DrawCircle(
                int(layer_x[1]),
                int(hy),
                6,
                _fade(colors.SKYBLUE, avg_act * 0.8 + 0.2),
            )

        for o in range(ACTION_DIM):
            oy = column_top + (o + 1) * output_step
            value = controls[o]
            color = _fade(colors.GREEN if value >= 0 else colors.RED, abs(value) * 0.85 + 0.15)
            rl.DrawCircle(int(layer_x[2]), int(oy), 12, color)
            rl.DrawCircleLines(int(layer_x[2]), int(oy), 12, colors.LIGHTGRAY)
            rl.DrawText(
                output_labels[o] + f" {value:+.2f}".encode(),
                int(layer_x[2]) + 18,
                int(oy) - 5,
                10,
                colors.LIGHTGRAY,
            )

    def run(self) -> None:
        rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"NeuroDrift")
        rl.SetTargetFPS(FPS)

        self.show_checkpoints = False

        while not rl.WindowShouldClose():
            if rl.IsKeyPressed(rl.KEY_S):
                self.show_sensors = not self.show_sensors
            if rl.IsKeyPressed(rl.KEY_C):
                self.show_checkpoints = not self.show_checkpoints
            if rl.IsKeyPressed(rl.KEY_SPACE):
                self.camera_follow = not self.camera_follow
            if rl.IsKeyPressed(rl.KEY_L):
                self.show_racing_line = not self.show_racing_line

            manual_shift = 0
            if rl.IsKeyPressed(rl.KEY_RIGHT):
                manual_shift = 1
            elif rl.IsKeyPressed(rl.KEY_LEFT):
                manual_shift = -1

            if manual_shift != 0:
                self.track.set_shape(self.track.current_shape + manual_shift)
                for car in self.population.cars:
                    car.spawn_target_checkpoint = self.track.spawn_checkpoint
                spawn_pos, spawn_angle = self.track.get_spawn_info(self.track.spawn_checkpoint)
                self.last_gen_best_fitness = self.population.evolve_and_reset(
                    spawn_pos, spawn_angle
                )
                self.generation += 1
                self.generation_step = 0

            for idx, car in enumerate(self.population.cars):
                if car.retired:
                    continue
                sensor_distances = car.get_sensor_distances(self.track)
                state = self._state_from_distances(sensor_distances, car)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                brain = self.population.brains[idx]
                with torch.no_grad():
                    steer, pedal = brain(state_tensor)[0].tolist()

                old_x = float(car.position[0])
                old_y = float(car.position[1])
                safe_pos = list(car.position)
                car.apply_action(steer, pedal, dt=FIXED_PHYSICS_DT)

                _process_checkpoint_crossing(car, self.track, old_x, old_y)

                car.frames_since_checkpoint += 1

                hit = self.track.check_car_collision(car)
                if hit:
                    car.position = safe_pos
                    car.velocity = 0.0
                    car.is_touching_wall = True
                else:
                    car.is_touching_wall = False

                _apply_strict_fitness_penalties(car, touching_wall=hit)
                if hit or car.frames_since_checkpoint > STALL_RETIRE_FRAMES:
                    car.retired = True
                    _credit_partial_gate(car, self.track)

                car.total_distance_traveled += math.hypot(
                    float(car.position[0]) - old_x,
                    float(car.position[1]) - old_y,
                )

            self.generation_step += 1
            all_retired = all(c.retired for c in self.population.cars)
            if self.generation_step >= MAX_GENERATION_STEPS or all_retired:
                for car in self.population.cars:
                    if not car.retired:
                        _credit_partial_gate(car, self.track)
                spawn_pos, spawn_angle = self.track.get_spawn_info(self.track.spawn_checkpoint)
                self.last_gen_best_fitness = self.population.evolve_and_reset(
                    spawn_pos, spawn_angle
                )
                self.generation += 1
                self.generation_step = 0
                print(
                    f"Generation {self.generation} | "
                    f"Best fitness (prev): {self.last_gen_best_fitness:.1f}"
                )

            cars = self.population.cars
            prog_vals = [_car_color_progress_score(c) for c in cars]
            min_prog = min(prog_vals) if prog_vals else 0.0
            max_prog = max(prog_vals) if prog_vals else 0.0

            best_car = None
            best_brain = None
            max_fit = -float("inf")
            for i, car in enumerate(self.population.cars):
                if car.fitness > max_fit:
                    max_fit = car.fitness
                    best_car = car
                    best_brain = self.population.brains[i]

            camera = ffi.new("struct Camera2D *")
            if self.camera_follow and best_car is not None:
                camera.offset.x = SCREEN_WIDTH / 2.0
                camera.offset.y = SCREEN_HEIGHT / 2.0
                camera.target.x = float(best_car.position[0])
                camera.target.y = float(best_car.position[1])
                camera.rotation = 0.0
                camera.zoom = 2.5
            else:
                camera.offset.x = 0.0
                camera.offset.y = 0.0
                camera.target.x = 0.0
                camera.target.y = 0.0
                camera.rotation = 0.0
                camera.zoom = 1.0

            rl.BeginDrawing()
            rl.ClearBackground(colors.DARKGRAY)
            rl.BeginMode2D(camera[0])
            self.track.render()
            car0 = self.population.cars[0]
            line_physics = (
                car0.base_max_speed,
                car0.turn_rate,
                car0.base_acceleration,
                car0.brake_deceleration,
            )
            if self.show_racing_line:
                self.track.render_racing_line(*line_physics)
            if self.show_checkpoints:
                self.track.render_checkpoints()
            for idx, car in enumerate(self.population.cars):
                if car is best_car:
                    body = colors.MAGENTA
                elif car.retired:
                    body = _RETIRED_COLOR
                elif idx in self.population.elite_indices:
                    body = _ELITE_COLOR
                else:
                    body = get_fitness_color(
                        _car_color_progress_score(car), min_prog, max_prog
                    )
                car.render(
                    body_color=body,
                    show_sensors=self.show_sensors and not car.retired,
                )
            rl.EndMode2D()
            steps_left = max(0, MAX_GENERATION_STEPS - self.generation_step)
            _draw_ui_sidebar(
                generation=self.generation,
                population_size=len(self.population.cars),
                track_shape=self.track.current_shape,
                track_name=self.track.shape_name,
                best_fitness=self.last_gen_best_fitness,
                steps_left=steps_left,
                max_steps=MAX_GENERATION_STEPS,
                show_sensors=self.show_sensors,
                show_checkpoints=self.show_checkpoints,
                camera_follow=self.camera_follow,
                ideal_lap=(
                    self.track.racing_line(*line_physics)[2] if self.show_racing_line else None
                ),
            )

            if best_car and best_brain:
                right_panel_x = SCREEN_WIDTH - 350
                brain_start_x = right_panel_x + 40
                hidden_col_x = brain_start_x + 90
                title_text = b"LIVE BRAIN CAM (LEADER)"
                title_font = 24
                title_w = rl.MeasureText(title_text, title_font)
                title_x = int(hidden_col_x - title_w / 2)
                rl.DrawText(title_text, title_x, 26, title_font, colors.SKYBLUE)
                self._draw_brain_cam(best_car, best_brain, brain_start_x, 142)

            _draw_brain_cam_key()

            rl.EndDrawing()

        rl.CloseWindow()
