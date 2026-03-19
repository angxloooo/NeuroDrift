"""Simulation module - main game loop and RL environment glue."""

import torch
from raylib import rl, colors

from .car import Car
from .track import Track
from .dqn_agent import DQNAgent


# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
DT = 1.0 / FPS
SENSOR_RANGE = 200.0
MAX_SPEED = 8.0
SPEED_BONUS_THRESHOLD = 0.5 * MAX_SPEED  # 50% of max for +5 bonus
BATCH_SIZE = 64
SPAWN_POS = (400, 150)  # Center of top drivable bar (between outer/inner boundaries)


class Simulation:
    """Main simulation loop: environment, agent, and training."""

    def __init__(self):
        self.track = Track()
        self.car = Car(
            position=SPAWN_POS,
            angle=0.0,
            max_speed=MAX_SPEED,
            sensor_range=SENSOR_RANGE,
        )
        self.agent = DQNAgent(
            state_dim=6,
            action_dim=3,
            buffer_size=10000,
            batch_size=BATCH_SIZE,
            gamma=0.99,
            lr=1e-3,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.05,
            target_update_freq=100,
        )
        self.episode_reward = 0.0
        self.episode_count = 0

    def _get_state(self) -> list[float]:
        """Build state vector: 5 normalized sensor distances + normalized speed."""
        sensor_distances = self.car.get_sensor_distances(self.track)
        normalized_sensors = [d / SENSOR_RANGE for d in sensor_distances]
        normalized_speed = self.car.velocity / MAX_SPEED
        return normalized_sensors + [normalized_speed]

    def _compute_reward(self, done: bool) -> float:
        """+1 per frame, +5 for speed, -100 on crash."""
        if done:
            return -100.0
        reward = 1.0
        if self.car.velocity > SPEED_BONUS_THRESHOLD:
            reward += 5.0
        return reward

    def run(self) -> None:
        """Run the main simulation loop."""
        rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"NeuroDrift")
        rl.SetTargetFPS(FPS)

        state = self._get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while not rl.WindowShouldClose():
            # 1. Select action
            action = self.agent.select_action(state_tensor)

            # 2. Apply action (physics step)
            self.car.apply_action(action, dt=DT)

            # 3. Check collision and compute reward
            done = self.track.check_car_collision(
                (self.car.position[0], self.car.position[1]),
                self.car.radius,
            )
            reward = self._compute_reward(done)
            self.episode_reward += reward

            # 4. Get next state
            next_state = self._get_state()
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # 5. Store in replay buffer (use raw lists for storage)
            self.agent.buffer.add((state, action, reward, next_state, done))

            # 6. Training step
            if len(self.agent.buffer) >= BATCH_SIZE:
                self.agent.train_step()

            # 7. Reset if done
            if done:
                self.car.reset(SPAWN_POS, 0.0)
                self.agent.epsilon = max(
                    self.agent.epsilon_min,
                    self.agent.epsilon * self.agent.epsilon_decay,
                )
                self.episode_count += 1
                if self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count} | Reward: {self.episode_reward:.0f} | Epsilon: {self.agent.epsilon:.3f}")
                self.episode_reward = 0.0
                state = self._get_state()
            else:
                state = next_state

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # 8. Render
            rl.BeginDrawing()
            rl.ClearBackground(colors.DARKGRAY)
            self.track.render()
            self.car.render()
            rl.DrawFPS(10, 10)
            rl.DrawText(f"Episode: {self.episode_count}".encode(), 10, 35, 16, colors.WHITE)
            rl.DrawText(f"Reward: {self.episode_reward:.0f}".encode(), 10, 55, 16, colors.WHITE)
            rl.EndDrawing()

        rl.CloseWindow()
