"""Genetic algorithm / neuroevolution: MLP brains and population management."""

import random
from typing import Any

import torch
import torch.nn as nn

from .car import Car


STATE_DIM = 6
ACTION_DIM = 6
HIDDEN_DIM = 16


def mutate_state_dict(
    state_dict: dict[str, torch.Tensor],
    mutation_rate: float = 0.15,
    noise_scale: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Clone state dict and add Gaussian noise to a random subset of parameters."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        t = tensor.clone()
        mask = torch.rand_like(t) < mutation_rate
        noise = torch.randn_like(t) * noise_scale
        t = t + noise * mask.to(t.dtype)
        out[key] = t
    return out


class Brain(nn.Module):
    """MLP policy: 6 inputs -> 16 -> 16 -> 6 logits."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PopulationManager:
    """Manages a fixed population of cars and one Brain per car."""

    def __init__(
        self,
        population_size: int = 100,
        elite_fraction: float = 0.1,
        spawn_position: tuple[float, float] = (400, 150),
        spawn_angle: float = 0.0,
        car_kwargs: dict[str, Any] | None = None,
        mutation_rate: float = 0.15,
        noise_scale: float = 0.2,
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.spawn_position = spawn_position
        self.spawn_angle = spawn_angle
        self.car_kwargs = car_kwargs or {}
        self.mutation_rate = mutation_rate
        self.noise_scale = noise_scale

        self.elite_count = max(1, int(population_size * elite_fraction))
        self.elite_indices: set[int] = set()

        self.cars: list[Car] = []
        self.brains: list[Brain] = []
        for _ in range(population_size):
            self.cars.append(
                Car(
                    position=spawn_position,
                    angle=spawn_angle,
                    **self.car_kwargs,
                )
            )
            self.brains.append(Brain())

    def evolve_and_reset(self) -> float:
        """Sort by fitness, elitism + mutation for the rest, reset all cars. Returns best fitness."""
        indexed = [(i, self.cars[i].fitness) for i in range(self.population_size)]
        indexed.sort(key=lambda t: (-t[1], t[0]))
        ranked_indices = [i for i, _ in indexed]
        best_fitness = indexed[0][1]

        elite = ranked_indices[: self.elite_count]
        self.elite_indices = set(elite)

        elite_brains = {i: self.brains[i] for i in elite}

        for i in range(self.population_size):
            if i in self.elite_indices:
                continue
            parent_idx = random.choice(elite)
            parent_sd = elite_brains[parent_idx].state_dict()
            mutated = mutate_state_dict(
                parent_sd,
                mutation_rate=self.mutation_rate,
                noise_scale=self.noise_scale,
            )
            self.brains[i].load_state_dict(mutated)

        for i in range(self.population_size):
            if i in self.elite_indices:
                self.cars[i].reset(self.spawn_position, self.spawn_angle)
            else:
                jitter = random.uniform(-0.2, 0.2)
                self.cars[i].reset(self.spawn_position, self.spawn_angle + jitter)

        return best_fitness
