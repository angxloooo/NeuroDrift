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
        max_speed: float = 8.0,
        acceleration: float = 0.3,
        turn_rate: float = 0.08,
        sensor_range: float = 200.0,
        radius: float = 12.0,
        length: float = 24.0,
    ):
        self.position = list(position)
        self.angle = angle
        self.velocity = 0.0
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.turn_rate = turn_rate
        self.sensor_range = sensor_range
        self.radius = radius
        self.length = length

    def get_sensor_distances(self, track: Track) -> list[float]:
        """Cast 5 rays and return distance to nearest boundary for each."""
        cx, cy = self.position
        front_offset = self.length / 2
        ray_start_x = cx + math.cos(self.angle) * front_offset
        ray_start_y = cy + math.sin(self.angle) * front_offset

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
        """Apply discrete action: 0=left+accel, 1=straight+accel, 2=right+accel."""
        if action == 0:
            self.angle -= self.turn_rate
        elif action == 2:
            self.angle += self.turn_rate
        # action 1: straight, no steering change

        self.velocity = min(self.velocity + self.acceleration, self.max_speed)
        self.position[0] += self.velocity * math.cos(self.angle) * dt
        self.position[1] += self.velocity * math.sin(self.angle) * dt

    def reset(self, position: tuple[float, float], angle: float = 0.0) -> None:
        """Reset car state for new episode."""
        self.position = list(position)
        self.angle = angle
        self.velocity = 0.0

    def render(self) -> None:
        """Draw the car and sensor rays."""
        cx, cy = self.position
        front_offset = self.length / 2

        # Draw sensor rays
        for i, deg in enumerate(SENSOR_ANGLES):
            rad = math.radians(deg) + self.angle
            ray_start_x = cx + math.cos(self.angle) * front_offset
            ray_start_y = cy + math.sin(self.angle) * front_offset
            ray_end_x = ray_start_x + math.cos(rad) * self.sensor_range
            ray_end_y = ray_start_y + math.sin(rad) * self.sensor_range
            rl.DrawLineV(
                [ray_start_x, ray_start_y],
                [ray_end_x, ray_end_y],
                colors.YELLOW,
            )

        # Draw car as rotated rectangle
        half_len = self.length / 2
        half_w = self.radius
        corners = [
            (half_len, -half_w),
            (half_len, half_w),
            (-half_len, half_w),
            (-half_len, -half_w),
        ]
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        world_corners = []
        for dx, dy in corners:
            rx = dx * cos_a - dy * sin_a + cx
            ry = dx * sin_a + dy * cos_a + cy
            world_corners.append([rx, ry])

        # Draw as filled polygon (triangle strip for rect)
        rl.DrawTriangle(
            world_corners[0],
            world_corners[1],
            world_corners[2],
            colors.RED,
        )
        rl.DrawTriangle(
            world_corners[0],
            world_corners[2],
            world_corners[3],
            colors.RED,
        )
