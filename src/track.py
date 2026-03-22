"""Track module - oval boundaries for the driving environment."""

import math
from raylib import rl, colors


class Track:
    """Donut-shaped track: outer and inner ovals approximated by line segments."""

    def __init__(
        self,
        center: tuple[float, float] = (750, 400),
        outer_rx: float = 350.0,
        outer_ry: float = 250.0,
        inner_rx: float = 250.0,
        inner_ry: float = 150.0,
        num_segments: int = 36,
    ):
        self.center = center
        self.outer_rx = outer_rx
        self.outer_ry = outer_ry
        self.inner_rx = inner_rx
        self.inner_ry = inner_ry
        self.num_segments = num_segments
        self.current_shape = 0
        self._boundary_segments = self._build_boundary_segments()
        self.checkpoints = self._build_checkpoints()

    def _distortion_at(self, a: float) -> float:
        """Radial offset added to ellipse radii at parameter angle a (radians)."""
        if self.current_shape == 0:
            return 0.0
        if self.current_shape == 1:
            return 60.0 * math.sin(2.0 * a)
        if self.current_shape == 2:
            return 40.0 * math.sin(3.0 * a)
        return 0.0

    def cycle_shape(self) -> None:
        """Advance procedural layout (oval -> peanut -> clover -> ...) and rebuild geometry."""
        self.current_shape = (self.current_shape + 1) % 3
        self._boundary_segments = self._build_boundary_segments()
        self.checkpoints = self._build_checkpoints()

    def set_shape(self, shape_index: int) -> None:
        self.current_shape = shape_index % 3
        self._boundary_segments = self._build_boundary_segments()
        self.checkpoints = self._build_checkpoints()

    def get_spawn_info(
        self, checkpoint_idx: int
    ) -> tuple[tuple[float, float], float]:
        """Midpoint of checkpoint radial and heading toward the next checkpoint centerline."""
        n = self.num_segments
        if n == 0:
            cx, cy = self.center
            return ((float(cx), float(cy)), 0.0)
        idx = checkpoint_idx % n
        (i1, i2) = self.checkpoints[idx]
        ix, iy = float(i1[0]), float(i1[1])
        ox, oy = float(i2[0]), float(i2[1])
        spawn_x = (ix + ox) * 0.5
        spawn_y = (iy + oy) * 0.5
        next_idx = (idx + 1) % n
        (n1, n2) = self.checkpoints[next_idx]
        nx = (float(n1[0]) + float(n2[0])) * 0.5
        ny = (float(n1[1]) + float(n2[1])) * 0.5
        spawn_angle = math.atan2(ny - spawn_y, nx - spawn_x)
        return ((spawn_x, spawn_y), spawn_angle)

    def _build_checkpoints(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Radial segments from inner to outer ellipse at each angle step (race order)."""
        cx, cy = self.center
        n = self.num_segments
        cks: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            d = self._distortion_at(a)
            ix = cx + (self.inner_rx + d) * math.cos(a)
            iy = cy + (self.inner_ry + d) * math.sin(a)
            ox = cx + (self.outer_rx + d) * math.cos(a)
            oy = cy + (self.outer_ry + d) * math.sin(a)
            cks.append(((ix, iy), (ox, oy)))
        return cks

    def _ellipse_segments(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        n = self.num_segments
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for i in range(n):
            a1 = 2.0 * math.pi * i / n
            a2 = 2.0 * math.pi * (i + 1) / n
            d1 = self._distortion_at(a1)
            d2 = self._distortion_at(a2)
            x1 = cx + (rx + d1) * math.cos(a1)
            y1 = cy + (ry + d1) * math.sin(a1)
            x2 = cx + (rx + d2) * math.cos(a2)
            y2 = cy + (ry + d2) * math.sin(a2)
            segs.append(((x1, y1), (x2, y2)))
        return segs

    def _build_boundary_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        cx, cy = self.center
        outer = self._ellipse_segments(cx, cy, self.outer_rx, self.outer_ry)
        inner = self._ellipse_segments(cx, cy, self.inner_rx, self.inner_ry)
        return outer + inner

    def get_boundary_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return list of (start, end) line segments for raycast and collision."""
        return self._boundary_segments

    def check_car_collision(self, car) -> bool:
        """OBB vs track segments: segment–segment intersection (expanded hull vs line centers)."""
        hw = 11.5
        hh = 6.5

        cx, cy = car.position[0], car.position[1]
        cos_a = math.cos(car.angle)
        sin_a = math.sin(car.angle)

        p1 = (cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a)
        p2 = (cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a)
        p3 = (cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a)
        p4 = (cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a)

        car_edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        for bp1, bp2 in self.get_boundary_segments():
            C = (bp1.x, bp1.y) if hasattr(bp1, "x") else (bp1[0], bp1[1])
            D = (bp2.x, bp2.y) if hasattr(bp2, "x") else (bp2[0], bp2[1])

            for A, B in car_edges:
                if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                    return True

        return False

    def render(self) -> None:
        """Draw the track boundaries."""
        line_thick = 3.0
        line_color = colors.WHITE

        for (p1, p2) in self._boundary_segments:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineEx(start, end, line_thick, line_color)

    def render_checkpoints(self) -> None:
        """Draw radial checkpoint segments (toggle in sim)."""
        ck_color = getattr(colors, "SKYBLUE", getattr(colors, "BLUE", (102, 191, 255, 255)))
        for (p1, p2) in self.checkpoints:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineV(start, end, ck_color)
