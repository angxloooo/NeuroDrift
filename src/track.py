"""Track module - oval boundaries for the driving environment."""

import math
from raylib import rl, colors


class Track:
    """Donut-shaped track: outer and inner ovals approximated by line segments."""

    def __init__(
        self,
        center: tuple[float, float] = (400, 300),
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
        self._boundary_segments = self._build_boundary_segments()

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
            x1 = cx + rx * math.cos(a1)
            y1 = cy + ry * math.sin(a1)
            x2 = cx + rx * math.cos(a2)
            y2 = cy + ry * math.sin(a2)
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

    def check_car_collision(self, center: tuple[float, float], radius: float) -> bool:
        """Check if car (circle) intersects any track boundary. Returns True on collision."""
        cx, cy = center
        center_vec = [cx, cy]

        for (p1, p2) in self._boundary_segments:
            seg_p1 = [p1[0], p1[1]]
            seg_p2 = [p2[0], p2[1]]
            if rl.CheckCollisionCircleLine(center_vec, radius, seg_p1, seg_p2):
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
