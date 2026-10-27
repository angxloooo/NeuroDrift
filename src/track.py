"""Track module - ring layouts and a Grand Prix circuit for the driving environment."""

import math
from raylib import rl, ffi, colors


SHAPE_NAMES = ["Oval", "Peanut", "Clover", "Grand Prix"]
GRAND_PRIX_SHAPE = 3

# Grand Prix circuit, clockwise on screen. Each entry is a polygon vertex rounded off with a
# circular arc: (x, y, corner radius, track width at that corner). Radii are sized against car
# handling (turn radius ~111 px flat out, ~40 px at crawl): R >= ~110 is flat, R ~45-75 needs a
# braking zone. Width blends smoothly between corners; braking zones and the hairpin run wider
# so there is room for more than one line.
GP_CORNERS = [
    (505, 715, 65, 110),  # T1: heavy braking 90 right at the end of the main straight
    (505, 545, 140, 90),  # T2-T4: medium-speed esses
    (600, 425, 120, 85),
    (510, 295, 140, 85),
    (525, 110, 70, 95),  # T5: onto the back straight
    (975, 110, 65, 105),  # T6: hairpin (two 90s)
    (975, 240, 65, 105),
    (830, 240, 70, 95),  # T7-T8: infield lefts
    (830, 420, 70, 95),
    (1085, 420, 75, 90),  # T9: right onto the run to the chicane
    (1085, 525, 50, 80),  # T10-T11: chicane
    (975, 595, 50, 80),
    (975, 715, 90, 95),  # T12: final corner onto the main straight
]
GP_SAMPLE_STEP = 4.0
GP_GATE_SPACING = 45.0
# Start/finish gate distance past the final corner exit.
GP_START_LINE_OFFSET = 120.0
# Max deviation (px) when reducing dense boundary samples to collision/raycast segments.
GP_SIMPLIFY_TOLERANCE = 0.75
GP_KERB_DEPTH = 7.0
GP_KERB_BLOCK_SAMPLES = 2
# Outside (exit) kerbs cover this many samples before each corner arc ends.
GP_EXIT_KERB_SAMPLES = 12

ASPHALT_COLOR = (44, 44, 48, 255)
KERB_RED = (210, 40, 45, 255)
KERB_WHITE = (235, 235, 235, 255)

# Uniform grid over boundary segments so sensors/collision only test nearby walls.
SEGMENT_GRID_CELL = 80.0

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _fillet_centerline(
    corners: list[tuple[float, float, float, float]], step: float
) -> tuple[list[Point], list[float], list[int], list[float], float]:
    """Sample a closed polygon whose vertices are rounded with tangent circular arcs.

    Returns (points, widths, turn_signs, corner_mid_s, main_straight_s): turn_signs is +1/-1
    on corner arcs (right/left on screen) and 0 on straights; main_straight_s is the arc
    length where the straight after the last corner begins.
    """
    n = len(corners)
    arcs = []
    for i, (vx, vy, r, w) in enumerate(corners):
        if r <= w * 0.5:
            raise ValueError(f"Corner {i}: radius {r} must exceed half the track width {w}")
        px, py = corners[i - 1][:2]
        nx, ny = corners[(i + 1) % n][:2]
        lin = math.hypot(vx - px, vy - py)
        lout = math.hypot(nx - vx, ny - vy)
        din = ((vx - px) / lin, (vy - py) / lin)
        dout = ((nx - vx) / lout, (ny - vy) / lout)
        cross = din[0] * dout[1] - din[1] * dout[0]
        turn = math.acos(max(-1.0, min(1.0, din[0] * dout[0] + din[1] * dout[1])))
        sign = 1 if cross > 0 else -1
        t = r * math.tan(turn * 0.5)
        ax, ay = vx - din[0] * t, vy - din[1] * t
        bx, by = vx + dout[0] * t, vy + dout[1] * t
        ccx, ccy = ax - din[1] * r * sign, ay + din[0] * r * sign
        arcs.append((ax, ay, bx, by, ccx, ccy, r, sign * turn, t))

    for i in range(n):
        bx, by, t_out = arcs[i][2], arcs[i][3], arcs[i][8]
        ax, ay, t_in = arcs[(i + 1) % n][0], arcs[(i + 1) % n][1], arcs[(i + 1) % n][8]
        vx, vy = corners[i][:2]
        nx, ny = corners[(i + 1) % n][:2]
        if t_out + t_in > math.hypot(nx - vx, ny - vy) + 1e-6:
            raise ValueError(f"Corners {i} and {(i + 1) % n}: radii too large for edge length")

    points: list[Point] = []
    signs: list[int] = []
    corner_mid_s: list[float] = []
    main_straight_s = 0.0
    s = 0.0
    for i in range(n):
        ax, ay, bx, by, ccx, ccy, r, sweep, _ = arcs[i]
        a0 = math.atan2(ay - ccy, ax - ccx)
        steps = max(2, math.ceil(abs(sweep) * r / step))
        for k in range(steps):
            a = a0 + sweep * k / steps
            points.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
            signs.append(1 if sweep > 0 else -1)
        corner_mid_s.append(s + abs(sweep) * r * 0.5)
        s += abs(sweep) * r

        if i == n - 1:
            main_straight_s = s
        nx, ny = arcs[(i + 1) % n][0], arcs[(i + 1) % n][1]
        length = math.hypot(nx - bx, ny - by)
        steps = math.ceil(length / step) if length > 1e-6 else 0
        for k in range(steps):
            points.append((bx + (nx - bx) * k / steps, by + (ny - by) * k / steps))
            signs.append(0)
        s += length

    total = s
    widths = _blend_widths(points, corner_mid_s, [c[3] for c in corners], total)
    return points, widths, signs, corner_mid_s, main_straight_s


def _cumulative_lengths(points: list[Point]) -> list[float]:
    out = [0.0]
    for i in range(1, len(points)):
        out.append(out[-1] + math.dist(points[i - 1], points[i]))
    return out


def _blend_widths(
    points: list[Point], anchor_s: list[float], anchor_w: list[float], total: float
) -> list[float]:
    """Smoothstep track width between per-corner anchors (wrapping around the lap)."""
    widths = []
    m = len(anchor_s)
    for s in _cumulative_lengths(points):
        j = 0
        while j < m and anchor_s[j] <= s:
            j += 1
        s0, w0 = (anchor_s[j - 1], anchor_w[j - 1]) if j > 0 else (anchor_s[-1] - total, anchor_w[-1])
        s1, w1 = (anchor_s[j], anchor_w[j]) if j < m else (anchor_s[0] + total, anchor_w[0])
        t = (s - s0) / max(1e-9, s1 - s0)
        t = t * t * (3.0 - 2.0 * t)
        widths.append(w0 + (w1 - w0) * t)
    return widths


def _tangents(points: list[Point]) -> list[Point]:
    n = len(points)
    out = []
    for i in range(n):
        (x0, y0), (x1, y1) = points[i - 1], points[(i + 1) % n]
        d = math.hypot(x1 - x0, y1 - y0)
        out.append(((x1 - x0) / d, (y1 - y0) / d))
    return out


def _simplify_open(points: list[Point], tol: float) -> list[Point]:
    """Ramer-Douglas-Peucker (iterative) keeping both endpoints."""
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = points[i], points[j]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        best, best_d = -1, tol
        for k in range(i + 1, j):
            px, py = points[k]
            if length > 1e-9:
                d = abs(dy * (px - x1) - dx * (py - y1)) / length
            else:
                d = math.hypot(px - x1, py - y1)
            if d > best_d:
                best, best_d = k, d
        if best >= 0:
            keep[best] = True
            stack.append((i, best))
            stack.append((best, j))
    return [p for p, k in zip(points, keep) if k]


def _closed_loop_segments(points: list[Point], tol: float) -> list[Segment]:
    far = max(range(len(points)), key=lambda i: math.dist(points[0], points[i]))
    first = _simplify_open(points[: far + 1], tol)
    second = _simplify_open(points[far:] + [points[0]], tol)
    loop = first[:-1] + second[:-1]
    return [(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]


def _strip(pairs: list[tuple[Point, Point]]):
    """ffi Vector2 array for DrawTriangleStrip from cross-section pairs, wound CCW on screen."""
    (a0, b0), (a1, _) = pairs[0], pairs[1]
    # raylib's first strip triangle is (p2, p0, p1); it must be CCW on screen (negative cross, y down).
    cross = (a0[0] - a1[0]) * (b0[1] - a1[1]) - (a0[1] - a1[1]) * (b0[0] - a1[0])
    flat = [p for a, b in pairs for p in ((b, a) if cross > 0 else (a, b))]
    arr = ffi.new("struct Vector2[]", len(flat))
    for i, (x, y) in enumerate(flat):
        arr[i].x = x
        arr[i].y = y
    return arr, len(flat)


class Track:
    """Closed driving course: boundary walls, ordered checkpoint gates, and rendering.

    Shapes 0-2 are donut rings (outer and inner ovals approximated by line segments, with
    radial gates); shape 3 is a Grand Prix circuit built from GP_CORNERS.
    """

    def __init__(
        self,
        center: tuple[float, float] = (750, 400),
        outer_rx: float = 350.0,
        outer_ry: float = 250.0,
        inner_rx: float = 250.0,
        inner_ry: float = 150.0,
        num_segments: int = 36,
        shape: int = GRAND_PRIX_SHAPE,
    ):
        self.center = center
        self.outer_rx = outer_rx
        self.outer_ry = outer_ry
        self.inner_rx = inner_rx
        self.inner_ry = inner_ry
        self.num_segments = num_segments
        self.set_shape(shape)

    @property
    def shape_name(self) -> str:
        return SHAPE_NAMES[self.current_shape]

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
        """Advance layout (oval -> peanut -> clover -> grand prix -> ...) and rebuild geometry."""
        self.set_shape(self.current_shape + 1)

    def set_shape(self, shape_index: int) -> None:
        self.current_shape = shape_index % len(SHAPE_NAMES)
        self._asphalt = None
        self._decor = []
        if self.current_shape == GRAND_PRIX_SHAPE:
            self._build_grand_prix()
        else:
            self._boundary_segments = self._build_boundary_segments()
            self.checkpoints = self._build_checkpoints()
            self.checkpoint_tangents = self._ring_checkpoint_tangents()
            self.spawn_checkpoint = (3 * self.num_segments) // 4
        self._segment_grid = self._build_segment_grid()

    def get_spawn_info(
        self, checkpoint_idx: int
    ) -> tuple[tuple[float, float], float]:
        """Midpoint of a checkpoint gate and heading toward the next gate's midpoint."""
        n = len(self.checkpoints)
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

    def _ring_checkpoint_tangents(self) -> list[Point]:
        """Unit race direction at each radial gate (perpendicular to the gate radius)."""
        cx, cy = self.center
        out = []
        for (i1, i2) in self.checkpoints:
            rx = (i1[0] + i2[0]) * 0.5 - cx
            ry = (i1[1] + i2[1]) * 0.5 - cy
            n = math.hypot(rx, ry)
            out.append((-ry / n, rx / n) if n > 1e-9 else (1.0, 0.0))
        return out

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

    def _build_grand_prix(self) -> None:
        """Offset the filleted centerline into walls, gates, asphalt, kerbs and start line."""
        center, widths, signs, _, main_s = _fillet_centerline(GP_CORNERS, GP_SAMPLE_STEP)
        tangents = _tangents(center)
        n = len(center)
        side_a: list[Point] = []
        side_b: list[Point] = []
        for (x, y), (tx, ty), w in zip(center, tangents, widths):
            h = w * 0.5
            side_a.append((x - ty * h, y + tx * h))
            side_b.append((x + ty * h, y - tx * h))

        self._boundary_segments = _closed_loop_segments(
            side_a, GP_SIMPLIFY_TOLERANCE
        ) + _closed_loop_segments(side_b, GP_SIMPLIFY_TOLERANCE)

        s_along = _cumulative_lengths(center)
        total = s_along[-1] + math.dist(center[-1], center[0])
        n_gates = max(1, round(total / GP_GATE_SPACING))
        start_s = main_s + GP_START_LINE_OFFSET
        self.checkpoints = []
        self.checkpoint_tangents = []
        for g in range(n_gates):
            target = (start_s + g * total / n_gates) % total
            j = min(range(n), key=lambda i: abs(s_along[i] - target))
            self.checkpoints.append((side_a[j], side_b[j]))
            self.checkpoint_tangents.append(tangents[j])
        self.spawn_checkpoint = 0

        pairs = [(side_a[i], side_b[i]) for i in range(n)] + [(side_a[0], side_b[0])]
        self._asphalt = _strip(pairs)

        # Kerbs: red/white blocks on the inside of every corner, plus the outside of its exit.
        for i in range(n):
            sign = signs[i]
            if sign == 0 or signs[(i + 1) % n] != sign:
                continue
            near_exit = signs[(i + GP_EXIT_KERB_SAMPLES) % n] != sign
            tx, ty = tangents[i]
            d = GP_KERB_DEPTH
            # Right turns (sign > 0) have side_a on the inside; inward normals point at the centerline.
            for side, inside, (nx, ny) in (
                (side_a, sign > 0, (ty, -tx)),
                (side_b, sign < 0, (-ty, tx)),
            ):
                if not inside and not near_exit:
                    continue
                p0, p1 = side[i], side[(i + 1) % n]
                q0 = (p0[0] + nx * d, p0[1] + ny * d)
                q1 = (p1[0] + nx * d, p1[1] + ny * d)
                color = KERB_RED if (i // GP_KERB_BLOCK_SAMPLES) % 2 == 0 else KERB_WHITE
                self._decor.append((*_strip([(p0, q0), (p1, q1)]), color))

        # Checkered start/finish line across gate 0.
        a, b = self.checkpoints[0]
        tx, ty = self.checkpoint_tangents[0]
        rows, cols = 2, max(2, round(math.dist(a, b) / 6.0))
        for r in range(rows):
            for c in range(cols):
                u0, u1 = c / cols, (c + 1) / cols
                v0, v1 = (r - 1) * 6.0, r * 6.0
                p = [
                    (a[0] + (b[0] - a[0]) * u + tx * v, a[1] + (b[1] - a[1]) * u + ty * v)
                    for u, v in ((u0, v0), (u0, v1), (u1, v0), (u1, v1))
                ]
                color = colors.WHITE if (r + c) % 2 == 0 else colors.BLACK
                self._decor.append((*_strip([(p[0], p[1]), (p[2], p[3])]), color))

    def _build_segment_grid(self) -> dict[tuple[int, int], list[int]]:
        cell = SEGMENT_GRID_CELL
        grid: dict[tuple[int, int], list[int]] = {}
        for i, ((x1, y1), (x2, y2)) in enumerate(self._boundary_segments):
            for gx in range(math.floor(min(x1, x2) / cell), math.floor(max(x1, x2) / cell) + 1):
                for gy in range(math.floor(min(y1, y2) / cell), math.floor(max(y1, y2) / cell) + 1):
                    grid.setdefault((gx, gy), []).append(i)
        return grid

    def get_boundary_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return list of (start, end) line segments for raycast and collision."""
        return self._boundary_segments

    def segments_near(self, x: float, y: float, radius: float) -> list[Segment]:
        """Boundary segments whose grid cells overlap the square of half-size radius at (x, y)."""
        cell = SEGMENT_GRID_CELL
        ids: set[int] = set()
        for gx in range(math.floor((x - radius) / cell), math.floor((x + radius) / cell) + 1):
            for gy in range(math.floor((y - radius) / cell), math.floor((y + radius) / cell) + 1):
                ids.update(self._segment_grid.get((gx, gy), ()))
        return [self._boundary_segments[i] for i in ids]

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

        for bp1, bp2 in self.segments_near(cx, cy, math.hypot(hw, hh)):
            C = (bp1.x, bp1.y) if hasattr(bp1, "x") else (bp1[0], bp1[1])
            D = (bp2.x, bp2.y) if hasattr(bp2, "x") else (bp2[0], bp2[1])

            for A, B in car_edges:
                if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                    return True

        return False

    def render(self) -> None:
        """Draw the track surface (Grand Prix only) and boundaries."""
        if self._asphalt is not None:
            rl.DrawTriangleStrip(self._asphalt[0], self._asphalt[1], ASPHALT_COLOR)
            for arr, count, color in self._decor:
                rl.DrawTriangleStrip(arr, count, color)

        line_thick = 3.0
        line_color = colors.WHITE

        for (p1, p2) in self._boundary_segments:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineEx(start, end, line_thick, line_color)

    def render_checkpoints(self) -> None:
        """Draw checkpoint gate segments (toggle in sim)."""
        ck_color = getattr(colors, "SKYBLUE", getattr(colors, "BLUE", (102, 191, 255, 255)))
        for (p1, p2) in self.checkpoints:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineV(start, end, ck_color)
