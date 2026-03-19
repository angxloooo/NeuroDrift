"""Track module - defines outer and inner boundaries for the driving environment."""

from raylib import rl, ffi, colors


class Track:
    """A simple donut-shaped track defined by outer and inner rectangular boundaries."""

    def __init__(
        self,
        outer_tl: tuple[float, float] = (100, 100),
        outer_br: tuple[float, float] = (700, 500),
        inner_tl: tuple[float, float] = (200, 200),
        inner_br: tuple[float, float] = (600, 400),
    ):
        """
        Create track with outer and inner rectangles.
        Drivable area is the space between them.
        """
        self.outer_tl = outer_tl
        self.outer_br = outer_br
        self.inner_tl = inner_tl
        self.inner_br = inner_br
        self._boundary_segments = self._build_boundary_segments()

    def _build_boundary_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Build list of (start, end) line segments for all boundaries."""
        ox1, oy1 = self.outer_tl
        ox2, oy2 = self.outer_br
        ix1, iy1 = self.inner_tl
        ix2, iy2 = self.inner_br

        outer = [
            ((ox1, oy1), (ox2, oy1)),  # top
            ((ox2, oy1), (ox2, oy2)),  # right
            ((ox2, oy2), (ox1, oy2)),  # bottom
            ((ox1, oy2), (ox1, oy1)),  # left
        ]
        inner = [
            ((ix1, iy1), (ix2, iy1)),  # top
            ((ix2, iy1), (ix2, iy2)),  # right
            ((ix2, iy2), (ix1, iy2)),  # bottom
            ((ix1, iy2), (ix1, iy1)),  # left
        ]
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
