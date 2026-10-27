"""Car module - physics, sensors, and rendering for the autonomous vehicle."""

import math
from raylib import rl, ffi, colors

from .track import Track, segment_hit


# Sensor angles in degrees relative to car heading, left (-90) to right (+90) every 22.5 degrees.
# The in-between rays let the brain see a corner's shape (e.g. a chicane's second apex) early.
SENSOR_ANGLES = [-90.0, -67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5, 90.0]
SENSOR_LABELS = [b"L90", b"L68", b"L45", b"L23", b"F", b"R23", b"R45", b"R68", b"R90"]

# Steering only when moving forward; scales from 0 at ~MIN to full at ~FULL (sim units ≈ px/s).
MIN_FORWARD_FOR_STEER = 10.0
STEER_FULL_AT_SPEED = 72.0
# Off the throttle (and below the brake) the car always bleeds a little speed.
COAST_DECELERATION = 30.0
# Turning at full lock costs this fraction of acceleration and top speed.
TURN_SPEED_PENALTY = 0.2

# Analog controls from the brain, each in [-1, 1]: steer (left < 0 < right) and
# pedal (brake < 0 < accelerate; 0 = coast).
CONTROL_LABELS = [b"STEER", b"PEDAL"]


class Car:
    """A top-down car with raycast sensors (SENSOR_ANGLES) and simple physics."""

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
        # Set when the car crashes or stalls; it stops moving and scoring until reset.
        self.retired = False
        self.turn_rate = turn_rate
        self.sensor_range = sensor_range
        self.radius = radius
        self.length = length
        self.fitness = 0.0
        self.spawn_target_checkpoint = spawn_target_checkpoint
        self.target_checkpoint = spawn_target_checkpoint
        self.frames_since_checkpoint = 0
        self.checkpoints_since_fitness_reward = 0
        self.total_gates_crossed = 0
        self.total_distance_traveled = 0.0
        self.last_sensor_distances = [self.sensor_range] * len(SENSOR_ANGLES)

    def get_sensor_distances(self, track: Track) -> list[float]:
        """Cast one ray per SENSOR_ANGLES entry; return distance to the nearest boundary for each."""
        cx, cy = self.position
        ray_start_x = cx
        ray_start_y = cy

        nearby = track.segments_near(cx, cy, self.sensor_range)
        distances = []
        for deg in SENSOR_ANGLES:
            rad = math.radians(deg) + self.angle
            ray_end_x = ray_start_x + math.cos(rad) * self.sensor_range
            ray_end_y = ray_start_y + math.sin(rad) * self.sensor_range

            ray_start = (ray_start_x, ray_start_y)
            ray_end = (ray_end_x, ray_end_y)

            min_t = 1.0
            for (p1, p2) in nearby:
                t = segment_hit(ray_start, ray_end, p1, p2)
                if 0.0 <= t < min_t:
                    min_t = t

            distances.append(min_t * self.sensor_range)

        self.last_sensor_distances = distances
        return distances

    def apply_action(self, steer: float, pedal: float, dt: float = 1.0 / 60.0) -> None:
        """Apply analog controls: steer and pedal in [-1, 1] (see CONTROL_LABELS)."""
        steer = max(-1.0, min(1.0, steer))
        pedal = max(-1.0, min(1.0, pedal))
        turn_mult = 1.0 - TURN_SPEED_PENALTY * abs(steer)
        # Drag proportional to speed, sized so full throttle tops out at max speed: throttle then
        # settles at a steady speed (~pedal * max), so "ease off for this corner" is a small change.
        # Without it any positive throttle ran to top speed and populations never learned to slow.
        drag = self.active_acceleration / max(1e-6, self.active_max_speed) * self.velocity
        if pedal >= 0.0:
            dv = self.active_acceleration * turn_mult * pedal - drag
        else:
            dv = -(self.brake_deceleration * -pedal) - drag
        dv -= COAST_DECELERATION
        self.velocity = max(0.0, min(self.velocity + dv * dt, self.active_max_speed * turn_mult))

        span = max(1e-6, STEER_FULL_AT_SPEED - MIN_FORWARD_FOR_STEER)
        steer_scale = min(
            1.0,
            max(0.0, self.velocity - MIN_FORWARD_FOR_STEER) / span,
        )
        self.angle += self.turn_rate * dt * steer_scale * steer

        self.position[0] += self.velocity * math.cos(self.angle) * dt
        self.position[1] += self.velocity * math.sin(self.angle) * dt

    def reset(self, position: tuple[float, float], angle: float = 0.0) -> None:
        """Reset car state for new episode."""
        self.position = list(position)
        self.angle = angle
        self.velocity = 0.0
        self.fitness = 0.0
        self.is_touching_wall = False
        self.retired = False
        self.active_max_speed = self.base_max_speed
        self.active_acceleration = self.base_acceleration
        self.target_checkpoint = self.spawn_target_checkpoint
        self.frames_since_checkpoint = 0
        self.checkpoints_since_fitness_reward = 0
        self.total_gates_crossed = 0
        self.total_distance_traveled = 0.0

    def render(
        self,
        body_color=colors.RED,
        *,
        show_sensors: bool = True,
    ) -> None:
        """Draw the car and optional sensor rays using last raycast from get_sensor_distances."""
        cx, cy = self.position

        if show_sensors:
            for i, deg in enumerate(SENSOR_ANGLES):
                dist = self.last_sensor_distances[i]
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
