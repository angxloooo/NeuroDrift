"""Simulation module - main game loop and neuroevolution environment glue."""

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
MAX_SPEED = 8.0
SPAWN_POS = (400, 150)

_ELITE_COLOR = getattr(colors, "GOLD", (255, 215, 0, 255))


class Simulation:
    """Main simulation loop: multi-car population and genetic evolution."""

    def __init__(self):
        self.track = Track()
        car_kwargs = {
            "max_speed": MAX_SPEED,
            "sensor_range": SENSOR_RANGE,
        }
        self.population = PopulationManager(
            population_size=50,
            elite_fraction=0.1,
            spawn_position=SPAWN_POS,
            spawn_angle=0.0,
            car_kwargs=car_kwargs,
            mutation_rate=0.05,
            noise_scale=0.1,
        )
        self.generation = 1
        self.last_gen_best_fitness = 0.0

    def _state_for_car(self, car: Car) -> list[float]:
        sensor_distances = car.get_sensor_distances(self.track)
        normalized_sensors = [d / SENSOR_RANGE for d in sensor_distances]
        normalized_speed = car.velocity / MAX_SPEED
        return normalized_sensors + [normalized_speed]

    def run(self) -> None:
        rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"NeuroDrift")
        rl.SetTargetFPS(FPS)

        while not rl.WindowShouldClose():
            all_dead = True
            for idx, car in enumerate(self.population.cars):
                if not car.is_alive:
                    continue
                all_dead = False

                state = self._state_for_car(car)
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

            if all_dead:
                self.last_gen_best_fitness = self.population.evolve_and_reset()
                self.generation += 1
                print(
                    f"Generation {self.generation} | "
                    f"Best fitness (prev): {self.last_gen_best_fitness:.1f}"
                )

            alive = sum(1 for c in self.population.cars if c.is_alive)

            rl.BeginDrawing()
            rl.ClearBackground(colors.DARKGRAY)
            self.track.render()
            for idx, car in enumerate(self.population.cars):
                if car.is_alive:
                    body = _ELITE_COLOR if idx in self.population.elite_indices else colors.RED
                    car.render(body_color=body)
            rl.DrawFPS(10, 10)
            rl.DrawText(f"Gen: {self.generation}".encode(), 10, 35, 16, colors.WHITE)
            rl.DrawText(f"Alive: {alive}/50".encode(), 10, 55, 16, colors.WHITE)
            rl.DrawText(
                f"Best (last): {self.last_gen_best_fitness:.0f}".encode(),
                10,
                75,
                16,
                colors.WHITE,
            )
            rl.EndDrawing()

        rl.CloseWindow()
