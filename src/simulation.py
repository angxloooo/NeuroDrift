"""Simulation module - main game loop and neuroevolution environment glue."""

import colorsys

import torch
from raylib import rl, colors

from .car import Car
from .ga_agent import PopulationManager
from .track import Track


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
DT = 1.0 / FPS
SENSOR_RANGE = 200.0
MAX_SPEED = 250.0
SPAWN_POS = (400, 80)
MAX_GENERATION_STEPS = 2500
POPULATION_SIZE = 100

_ELITE_COLOR = getattr(colors, "GOLD", (255, 215, 0, 255))
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


def _draw_fitness_color_key(x: int, y: int) -> None:
    """HUD legend: HSV fitness scale (red=low, blue=high) matches get_fitness_color."""
    title = b"Fitness color (this generation)"
    rl.DrawText(title, x, y, 12, colors.LIGHTGRAY)
    bar_x = x
    bar_y = y + 16
    bar_w = 200
    bar_h = 14
    for i in range(bar_w):
        t = i / max(1, bar_w - 1)
        hue = t * FITNESS_HUE_MAX
        rf, gf, bf = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        seg = (int(rf * 255), int(gf * 255), int(bf * 255), 255)
        rl.DrawRectangle(bar_x + i, bar_y, 1, bar_h, seg)
    rl.DrawRectangleLines(bar_x, bar_y, bar_w, bar_h, colors.RAYWHITE)
    sub_y = bar_y + bar_h + 4
    rl.DrawText(b"Worse (low fitness)", bar_x, sub_y, 10, colors.WHITE)
    rl.DrawText(b"Better (high fitness)", bar_x + bar_w - 118, sub_y, 10, colors.WHITE)
    rl.DrawText(
        b"Gold cars = elite from previous gen",
        bar_x,
        sub_y + 14,
        10,
        _ELITE_COLOR,
    )


class Simulation:
    """Main simulation loop: multi-car population and genetic evolution."""

    def __init__(self):
        self.track = Track()
        car_kwargs = {
            "max_speed": MAX_SPEED,
            "sensor_range": SENSOR_RANGE,
        }
        self.population = PopulationManager(
            population_size=POPULATION_SIZE,
            elite_fraction=0.1,
            spawn_position=SPAWN_POS,
            spawn_angle=0.0,
            car_kwargs=car_kwargs,
        )
        self.generation = 1
        self.last_gen_best_fitness = 0.0
        self.show_sensors = True
        self.generation_step = 0

    def _state_from_distances(
        self, sensor_distances: list[float], car: Car
    ) -> list[float]:
        normalized_sensors = [d / SENSOR_RANGE for d in sensor_distances]
        normalized_speed = car.velocity / MAX_SPEED
        return normalized_sensors + [normalized_speed]

    def run(self) -> None:
        rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"NeuroDrift")
        rl.SetTargetFPS(FPS)

        while not rl.WindowShouldClose():
            if rl.IsKeyPressed(rl.KEY_S):
                self.show_sensors = not self.show_sensors

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

                car.apply_action(action, dt=DT)
                car.fitness += car.velocity * DT

                hit = self.track.check_car_collision(
                    (car.position[0], car.position[1]),
                    car.radius,
                )
                if hit:
                    car.is_alive = False

            self.generation_step += 1
            if self.generation_step >= MAX_GENERATION_STEPS:
                for c in self.population.cars:
                    if c.is_alive:
                        c.is_alive = False
                all_dead = True

            if all_dead:
                self.last_gen_best_fitness = self.population.evolve_and_reset()
                self.generation += 1
                self.generation_step = 0
                print(
                    f"Generation {self.generation} | "
                    f"Best fitness (prev): {self.last_gen_best_fitness:.1f}"
                )

            alive = sum(1 for c in self.population.cars if c.is_alive)
            fit_vals = [c.fitness for c in self.population.cars]
            min_fitness = min(fit_vals) if fit_vals else 0.0
            max_fitness = max(fit_vals) if fit_vals else 0.0

            rl.BeginDrawing()
            rl.ClearBackground(colors.DARKGRAY)
            self.track.render()
            for idx, car in enumerate(self.population.cars):
                if car.is_alive:
                    render_distances = (
                        car.get_sensor_distances(self.track)
                        if self.show_sensors
                        else None
                    )
                    if idx in self.population.elite_indices:
                        body = _ELITE_COLOR
                    else:
                        body = get_fitness_color(
                            car.fitness, min_fitness, max_fitness
                        )
                    car.render(
                        body_color=body,
                        show_sensors=self.show_sensors,
                        sensor_distances=render_distances,
                    )
            rl.DrawFPS(10, 10)
            rl.DrawText(f"Gen: {self.generation}".encode(), 10, 35, 16, colors.WHITE)
            rl.DrawText(f"Alive: {alive}/{len(self.population.cars)}".encode(), 10, 55, 16, colors.WHITE)
            rl.DrawText(
                f"Best (last): {self.last_gen_best_fitness:.0f}".encode(),
                10,
                75,
                16,
                colors.WHITE,
            )
            steps_left = max(0, MAX_GENERATION_STEPS - self.generation_step)
            rl.DrawText(
                f"Steps left: {steps_left}".encode(),
                10,
                95,
                16,
                colors.WHITE,
            )
            sensors_txt = "Sensors: ON" if self.show_sensors else "Sensors: OFF"
            rl.DrawText(sensors_txt.encode(), 10, 115, 16, colors.WHITE)
            _draw_fitness_color_key(SCREEN_WIDTH - 220, 10)
            rl.EndDrawing()

        rl.CloseWindow()
