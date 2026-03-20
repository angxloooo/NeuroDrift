"""Car module - physics, sensors, and rendering for the autonomous vehicle."""

import math
from raylib import rl, ffi, colors

from .track import Track


# Sensor angles in degrees relative to car heading: left, diag-left, forward, diag-right, right
SENSOR_ANGLES = [-90, -45, 0, 45, 90]


class Car:
    """A top-down car with 5 raycast sensors and simple physics."""

    def __init__(
        self,
        position: tuple[float, float] = (400, 300),
        angle: float = 0.0,
        max_speed: float = 250.0,
        acceleration: float = 600.0,
        brake_deceleration: float = 800.0,
        turn_rate: float = 1.8,
        sensor_range: float = 200.0,
        radius: float = 12.0,
        length: float = 24.0,
        spawn_target_checkpoint: int = 0,
    ):
        self.position = list(position)
        self.angle = angle
        self.velocity = 0.0
        self.base_max_speed = max_speed
        self.base_acceleration = acceleration
        self.brake_deceleration = brake_deceleration
        self.active_max_speed = max_speed
        self.active_acceleration = acceleration
        self.is_touching_wall = False
        self.turn_rate = turn_rate
        self.sensor_range = sensor_range
        self.radius = radius
        self.length = length
        self.fitness = 0.0
        self.is_alive = True
        self._default_target_checkpoint = spawn_target_checkpoint
        self.target_checkpoint = spawn_target_checkpoint
        self.laps = 0

    def get_sensor_distances(self, track: Track) -> list[float]:
        """Cast 5 rays and return distance to nearest boundary for each."""
        cx, cy = self.position
        ray_start_x = cx
        ray_start_y = cy

        distances = []
        for deg in SENSOR_ANGLES:
            rad = math.radians(deg) + self.angle
            ray_end_x = ray_start_x + math.cos(rad) * self.sensor_range
            ray_end_y = ray_start_y + math.sin(rad) * self.sensor_range

            ray_start = [ray_start_x, ray_start_y]
            ray_end = [ray_end_x, ray_end_y]

            min_dist = self.sensor_range
            collision_point = ffi.new("struct Vector2 *")

            for (p1, p2) in track.get_boundary_segments():
                seg_p1 = [float(p1[0]), float(p1[1])]
                seg_p2 = [float(p2[0]), float(p2[1])]
                if rl.CheckCollisionLines(ray_start, ray_end, seg_p1, seg_p2, collision_point):
                    dx = collision_point.x - ray_start_x
                    dy = collision_point.y - ray_start_y
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist

            distances.append(min_dist)

        return distances

    def apply_action(self, action: int, dt: float = 1.0 / 60.0) -> None:
        """Apply discrete action: accel/brake combined with left, straight, or right."""
        if not self.is_alive:
            return

        if action in (0, 3):
            self.angle -= self.turn_rate * dt
        elif action in (2, 5):
            self.angle += self.turn_rate * dt

        if action <= 2:
            self.velocity = min(
                self.velocity + self.active_acceleration * dt,
                self.active_max_speed,
            )
        else:
            self.velocity = max(
                self.velocity - self.brake_deceleration * dt,
                0.0,
            )

        self.position[0] += self.velocity * math.cos(self.angle) * dt
        self.position[1] += self.velocity * math.sin(self.angle) * dt

    def reset(self, position: tuple[float, float], angle: float = 0.0) -> None:
        """Reset car state for new episode."""
        self.position = list(position)
        self.angle = angle
        self.velocity = 0.0
        self.fitness = 0.0
        self.is_alive = True
        self.is_touching_wall = False
        self.active_max_speed = self.base_max_speed
        self.active_acceleration = self.base_acceleration
        self.target_checkpoint = self._default_target_checkpoint
        self.laps = 0

    def render(
        self,
        body_color=colors.RED,
        *,
        show_sensors: bool = True,
        sensor_distances: list[float] | None = None,
    ) -> None:
        """Draw the car and optional sensor rays to actual hit distances."""
        if not self.is_alive:
            return

        cx, cy = self.position

        if show_sensors and sensor_distances is not None:
            for i, deg in enumerate(SENSOR_ANGLES):
                dist = sensor_distances[i]
                rad = math.radians(deg) + self.angle
                ray_start_x = cx
                ray_start_y = cy
                ray_end_x = ray_start_x + math.cos(rad) * dist
                ray_end_y = ray_start_y + math.sin(rad) * dist
                rl.DrawLineV(
                    [ray_start_x, ray_start_y],
                    [ray_end_x, ray_end_y],
                    colors.YELLOW,
                )

        width = 20.0
        height = 10.0
        rect = ffi.new(
            "struct Rectangle *",
            {
                "x": float(cx - width / 2),
                "y": float(cy - height / 2),
                "width": float(width),
                "height": float(height),
            },
        )
        origin = ffi.new(
            "struct Vector2 *",
            {"x": float(width / 2), "y": float(height / 2)},
        )
        rotation_deg = float(self.angle * 180.0 / math.pi)
        rl.DrawRectanglePro(rect[0], origin[0], rotation_deg, body_color)
