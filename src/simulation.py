"""Simulation module - main game loop and neuroevolution environment glue."""

import colorsys

import torch
from raylib import ffi, rl, colors

from .car import Car
from .ga_agent import HIDDEN_DIM, PopulationManager
from .track import Track


SCREEN_WIDTH = 1450
SCREEN_HEIGHT = 800
FPS = 60
DT = 1.0 / FPS
SENSOR_RANGE = 200.0
MAX_SPEED = 250.0
MAX_GENERATION_STEPS = 2500
POPULATION_SIZE = 100

_ELITE_COLOR = getattr(colors, "WHITE", (255, 255, 255, 255))
FITNESS_HUE_MAX = 0.65


def get_fitness_color(
    fitness: float, min_fitness: float, max_fitness: float
) -> tuple[int, int, int, int]:
    """Map fitness to a smooth HSV spectrum (red hue 0 -> blue ~0.65) for raylib."""
    if max_fitness == min_fitness:
        t = 0.0
    else:
        t = (fitness - min_fitness) / (max_fitness - min_fitness)
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
    """Draw HSV fitness gradient + labels; returns y after block."""
    rl.DrawText(
        b"HSV gradient: red (low) -> blue (high) vs gen min/max",
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
    rl.DrawText(b"Worse (low fitness)", x, sub_y, 11, colors.WHITE)
    rl.DrawText(
        b"Better (high fitness)",
        x + bar_w - 130,
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
    best_fitness: float,
    steps_left: int,
    max_steps: int,
    show_sensors: bool,
    show_checkpoints: bool,
    camera_follow: bool,
) -> None:
    """Left column UI: title, stats, controls, color legend."""
    x = 30
    sep_w = 320
    y = 40

    rl.DrawText(b"NEURODRIFT AI", x, y, 30, colors.ORANGE)
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
        f"Track Mode: {track_shape}".encode(),
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
        (b"  L / DL / F / DR / R = Five Sensors", 16, colors.RAYWHITE),
        (b"  SPD = Speed (Normalized)", 16, colors.RAYWHITE),
        (f"Blue: Hidden Layer ({HIDDEN_DIM} Neurons)".encode(), 16, colors.RAYWHITE),
        (b"Right: Outputs (A = Accel, B = Brake)", 16, colors.RAYWHITE),
        (b"  L / S / R = Left / Straight / Right", 16, colors.RAYWHITE),
        (b"Green Dot = Selected Action", 16, colors.RAYWHITE),
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
    collision_point,
) -> None:
    """Award fitness when movement segment crosses the car's next checkpoint (sequential)."""
    n_ck = len(track.checkpoints)
    if n_ck == 0 or not car.is_alive:
        return
    move_start = [old_x, old_y]
    move_end = [float(car.position[0]), float(car.position[1])]
    while True:
        p1, p2 = track.checkpoints[car.target_checkpoint]
        seg_start = [float(p1[0]), float(p1[1])]
        seg_end = [float(p2[0]), float(p2[1])]
        if not rl.CheckCollisionLines(
            move_start, move_end, seg_start, seg_end, collision_point
        ):
            break
        car.fitness += 1000.0
        car.target_checkpoint += 1
        if car.target_checkpoint >= n_ck:
            car.target_checkpoint = 0
            car.laps += 1
            car.fitness += 5000.0


class Simulation:
    """Main simulation loop: multi-car population and genetic evolution."""

    def __init__(self):
        self.track = Track()
        self.start_ck = (3 * self.track.num_segments) // 4
        spawn_pos, spawn_angle = self.track.get_spawn_info(self.start_ck)
        car_kwargs = {
            "max_speed": MAX_SPEED,
            "sensor_range": SENSOR_RANGE,
            "spawn_target_checkpoint": self.start_ck,
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
            logits = brain(state_tensor)
        action = int(torch.argmax(logits, dim=-1).item())

        layer_x = [start_x, start_x + 90, start_x + 180]
        hidden_step = 250 // max(1, HIDDEN_DIM - 1)
        edge_color = _fade(colors.WHITE, 0.28)

        for i in range(6):
            iy = start_y + i * 35
            for h in range(HIDDEN_DIM):
                hy = start_y - 45 + h * hidden_step
                rl.DrawLine(
                    int(layer_x[0]),
                    int(iy),
                    int(layer_x[1]),
                    int(hy),
                    edge_color,
                )

        for h in range(HIDDEN_DIM):
            hy = start_y - 45 + h * hidden_step
            for o in range(6):
                oy = start_y + o * 35
                rl.DrawLine(
                    int(layer_x[1]),
                    int(hy),
                    int(layer_x[2]),
                    int(oy),
                    edge_color,
                )

        input_labels = [b"L", b"DL", b"F", b"DR", b"R", b"SPD"]
        output_labels = [
            b"A+L",
            b"A+S",
            b"A+R",
            b"B+L",
            b"B+S",
            b"B+R",
        ]

        for i in range(6):
            iy = start_y + i * 35
            act = max(0.0, min(1.0, activations[i]))
            color = _fade(colors.YELLOW, act) if act > 0.1 else colors.DARKGRAY
            rl.DrawCircle(int(layer_x[0]), int(iy), 10, color)
            rl.DrawCircleLines(int(layer_x[0]), int(iy), 10, colors.LIGHTGRAY)
            rl.DrawText(input_labels[i], int(layer_x[0]) - 35, int(iy) - 5, 10, colors.LIGHTGRAY)

        avg_act = sum(activations) / len(activations)
        for h in range(HIDDEN_DIM):
            hy = start_y - 45 + h * hidden_step
            rl.DrawCircle(
                int(layer_x[1]),
                int(hy),
                6,
                _fade(colors.SKYBLUE, avg_act * 0.8 + 0.2),
            )

        for o in range(6):
            oy = start_y + o * 35
            color = colors.GREEN if o == action else colors.DARKGRAY
            rl.DrawCircle(int(layer_x[2]), int(oy), 10, color)
            rl.DrawCircleLines(int(layer_x[2]), int(oy), 10, colors.LIGHTGRAY)
            rl.DrawText(
                output_labels[o],
                int(layer_x[2]) + 15,
                int(oy) - 5,
                10,
                colors.LIGHTGRAY,
            )

    def run(self) -> None:
        rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"NeuroDrift")
        rl.SetTargetFPS(60)

        self.show_checkpoints = False

        while not rl.WindowShouldClose():
            if rl.IsKeyPressed(rl.KEY_S):
                self.show_sensors = not self.show_sensors
            if rl.IsKeyPressed(rl.KEY_C):
                self.show_checkpoints = not self.show_checkpoints
            if rl.IsKeyPressed(rl.KEY_SPACE):
                self.camera_follow = not self.camera_follow

            manual_shift = 0
            if rl.IsKeyPressed(rl.KEY_RIGHT):
                manual_shift = 1
            elif rl.IsKeyPressed(rl.KEY_LEFT):
                manual_shift = -1

            if manual_shift != 0:
                self.track.set_shape(self.track.current_shape + manual_shift)
                spawn_pos, spawn_angle = self.track.get_spawn_info(self.start_ck)
                self.last_gen_best_fitness = self.population.evolve_and_reset(
                    spawn_pos, spawn_angle
                )
                self.generation += 1
                self.generation_step = 0

            collision_pt = ffi.new("struct Vector2 *")
            all_dead = True
            for idx, car in enumerate(self.population.cars):
                if not car.is_alive:
                    continue
                all_dead = False

                sensor_distances = car.get_sensor_distances(self.track)
                state = self._state_from_distances(sensor_distances, car)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                brain = self.population.brains[idx]
                with torch.no_grad():
                    logits = brain(state_tensor)
                    action = int(logits.argmax(dim=-1).item())

                safe_position = list(car.position)
                old_x = float(car.position[0])
                old_y = float(car.position[1])
                car.apply_action(action, dt=DT)
                car.fitness += (car.velocity * DT) * 0.5

                _process_checkpoint_crossing(
                    car, self.track, old_x, old_y, collision_pt
                )

                hit = self.track.check_car_collision(car)
                if hit:
                    car.position = list(safe_position)
                    car.velocity *= 0.5
                    car.is_touching_wall = True
                    car.active_max_speed = car.base_max_speed * 0.5
                    car.active_acceleration = car.base_acceleration * 0.5
                    if car.velocity > car.active_max_speed:
                        car.velocity = car.active_max_speed
                    car.fitness -= 2.0
                else:
                    car.is_touching_wall = False
                    car.active_max_speed = car.base_max_speed
                    car.active_acceleration = car.base_acceleration

            self.generation_step += 1
            if self.generation_step >= MAX_GENERATION_STEPS:
                for c in self.population.cars:
                    if c.is_alive:
                        c.is_alive = False
                all_dead = True

            if all_dead:
                if self.generation % 10 == 0:
                    self.track.cycle_shape()
                spawn_pos, spawn_angle = self.track.get_spawn_info(self.start_ck)
                self.last_gen_best_fitness = self.population.evolve_and_reset(
                    spawn_pos, spawn_angle
                )
                self.generation += 1
                self.generation_step = 0
                print(
                    f"Generation {self.generation} | "
                    f"Best fitness (prev): {self.last_gen_best_fitness:.1f}"
                )

            fit_vals = [c.fitness for c in self.population.cars]
            min_fitness = min(fit_vals) if fit_vals else 0.0
            max_fitness = max(fit_vals) if fit_vals else 0.0

            best_car = None
            best_brain = None
            max_fit = -float("inf")
            for i, car in enumerate(self.population.cars):
                if car.is_alive and car.fitness > max_fit:
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
            if self.show_checkpoints:
                self.track.render_checkpoints()
            for idx, car in enumerate(self.population.cars):
                if car.is_alive:
                    if car is best_car:
                        body = colors.MAGENTA
                    elif idx in self.population.elite_indices:
                        body = _ELITE_COLOR
                    else:
                        body = get_fitness_color(
                            car.fitness, min_fitness, max_fitness
                        )
                    car.render(
                        body_color=body,
                        show_sensors=self.show_sensors,
                    )
            rl.EndMode2D()
            steps_left = max(0, MAX_GENERATION_STEPS - self.generation_step)
            _draw_ui_sidebar(
                generation=self.generation,
                population_size=len(self.population.cars),
                track_shape=self.track.current_shape,
                best_fitness=self.last_gen_best_fitness,
                steps_left=steps_left,
                max_steps=MAX_GENERATION_STEPS,
                show_sensors=self.show_sensors,
                show_checkpoints=self.show_checkpoints,
                camera_follow=self.camera_follow,
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
