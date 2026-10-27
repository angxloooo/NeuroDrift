"""Track module - circuit layouts, walls, checkpoint gates and rendering."""

import math
from raylib import rl, ffi, colors


# Circuits, clockwise on screen, easiest first. Each corner is a polygon vertex rounded off
# with a circular arc: (x, y, corner radius, track width at that corner). Radii are sized
# against car handling (turn radius ~111 px flat out, ~40 px at crawl): R >= ~110 is flat,
# R ~45-75 needs a braking zone. Width blends smoothly between corners. The last corner leads
# onto the main straight, where the start/finish line sits. Keep walls inside the play area
# between the HUD columns: x 455-1125 (x <= 1095 above y 345, under the Brain Cam), y 55-770.
LAYOUTS: list[tuple[str, list[tuple[float, float, float, float]]]] = [
    (
        "Speedway",  # Four flat-out bankings: learn to steer before learning to brake.
        [
            (530, 690, 150, 105),
            (560, 140, 160, 105),
            (1030, 160, 150, 105),
            (1060, 700, 170, 105),
        ],
    ),
    (
        "Grand Prix",  # Every corner type: braking into T1, esses, hairpin, chicane.
        [
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
        ],
    ),
    (
        "High Speed",  # Long full-throttle runs broken by two chicanes; fast sweepers between.
        [
            (850, 110, 50, 90),  # T1-T2: first chicane at the end of the main straight
            (900, 195, 50, 90),
            (1040, 195, 100, 100),  # T3: fast right
            (1040, 420, 50, 85),  # T4-T5: second chicane
            (960, 480, 50, 85),
            (960, 720, 100, 100),  # T6: onto the back straight
            (520, 720, 160, 100),  # T7: long flat right
            (520, 110, 120, 100),  # T8: final corner onto the main straight
        ],
    ),
    (
        "Technical",  # Esses, a double hairpin, then a long sweeping double-apex right.
        [
            (505, 715, 65, 105),  # T1
            (505, 590, 110, 85),  # T2-T4: esses
            (605, 470, 110, 85),
            (505, 340, 110, 85),
            (525, 110, 75, 95),  # T5
            (745, 110, 55, 100),  # T6-T9: double hairpin
            (745, 300, 55, 100),
            (895, 300, 55, 100),
            (895, 110, 55, 100),
            (1025, 110, 60, 95),  # T10
            (1045, 470, 180, 95),  # T11: double-apex sweeper
            (990, 715, 100, 100),  # T12: final corner onto the main straight
        ],
    ),
    (
        "Street Circuit",  # Narrow, walled, almost every corner a slow 90.
        [
            (500, 720, 60, 85),
            (500, 470, 50, 80),
            (640, 470, 50, 80),
            (640, 300, 50, 80),
            (500, 300, 50, 80),
            (500, 110, 60, 85),
            (800, 110, 50, 80),
            (800, 250, 50, 80),
            (1000, 250, 50, 80),
            (1000, 450, 50, 80),
            (860, 450, 50, 80),
            (860, 580, 50, 80),
            (1060, 580, 60, 85),
            (1060, 720, 60, 85),
        ],
    ),
]
SAMPLE_STEP = 4.0
GATE_SPACING = 45.0
# Start/finish gate distance past the final corner exit (capped at 40% of the main straight).
START_LINE_OFFSET = 120.0
# Max deviation (px) when reducing dense boundary samples to collision/raycast segments.
SIMPLIFY_TOLERANCE = 0.75
KERB_DEPTH = 7.0
KERB_BLOCK_SAMPLES = 2
# Outside (exit) kerbs cover this many samples before each corner arc ends.
EXIT_KERB_SAMPLES = 12

ASPHALT_COLOR = (44, 44, 48, 255)
KERB_RED = (210, 40, 45, 255)
KERB_WHITE = (235, 235, 235, 255)

# Cars line up this far behind the start/finish gate so their first move crosses it cleanly
# (spawning exactly on the gate left crossing it to float rounding on tilted gates).
SPAWN_BEHIND_LINE = 15.0

# Racing line: points every RACING_LINE_SPACING px along the centerline, kept this far off
# each wall (car half-width 6.5 px plus slack), relaxed for RACING_LINE_SWEEPS sweeps.
RACING_LINE_SPACING = 12.0
RACING_LINE_MARGIN = 10.0
RACING_LINE_SWEEPS = 1500
RACING_LINE_RELAX = 1.5
RACING_LINE_SLOW = (220, 50, 50, 255)
RACING_LINE_MID = (240, 200, 40, 255)
RACING_LINE_FAST = (60, 210, 90, 255)

# Uniform grid over boundary segments so sensors/collision only test nearby walls.
SEGMENT_GRID_CELL = 80.0

Point = tuple[float, float]
Segment = tuple[Point, Point]


def segment_hit(a: Point, b: Point, c: Point, d: Point) -> float:
    """Fraction t along a->b where it crosses segment c-d, or -1.0 if they don't intersect.

    Double precision on purpose: raylib's CheckCollisionLines (float32 + bounding-box tests)
    misses crossings when one segment is nearly axis-aligned, e.g. a car driving straight
    through a gate or a ray parallel to a wall.
    """
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return -1.0
    qx, qy = c[0] - a[0], c[1] - a[1]
    t = (qx * sy - qy * sx) / denom
    u = (qx * ry - qy * rx) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return -1.0


def _fillet_centerline(
    corners: list[tuple[float, float, float, float]], step: float
) -> tuple[list[Point], list[float], list[int], list[float], float, float]:
    """Sample a closed polygon whose vertices are rounded with tangent circular arcs.

    Returns (points, widths, turn_signs, corner_mid_s, main_straight_s, main_straight_len):
    turn_signs is +1/-1 on corner arcs (right/left on screen) and 0 on straights; the main
    straight is the one after the last corner.
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
    main_straight_s = main_straight_len = 0.0
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

        nx, ny = arcs[(i + 1) % n][0], arcs[(i + 1) % n][1]
        length = math.hypot(nx - bx, ny - by)
        if i == n - 1:
            main_straight_s, main_straight_len = s, length
        steps = math.ceil(length / step) if length > 1e-6 else 0
        for k in range(steps):
            points.append((bx + (nx - bx) * k / steps, by + (ny - by) * k / steps))
            signs.append(0)
        s += length

    total = s
    widths = _blend_widths(points, corner_mid_s, [c[3] for c in corners], total)
    return points, widths, signs, corner_mid_s, main_straight_s, main_straight_len


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


def _circle_radius(p0: Point, p1: Point, p2: Point) -> float:
    """Radius of the circle through three points (inf when collinear)."""
    a, b, c = math.dist(p0, p1), math.dist(p1, p2), math.dist(p0, p2)
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    return math.inf if area2 < 1e-9 else a * b * c / (2.0 * area2)


def _min_curvature_line(
    center: list[Point], normals: list[Point], limits: list[float]
) -> list[Point]:
    """Smoothest closed path with p_i = center_i + alpha_i * normal_i and |alpha_i| <= limit_i.

    Minimizes the sum of squared second differences (discrete curvature) with projected
    over-relaxed Gauss-Seidel on the biharmonic stencil.
    """
    m = len(center)
    alpha = [0.0] * m

    def pt(i: int) -> Point:
        i %= m
        return (center[i][0] + alpha[i] * normals[i][0], center[i][1] + alpha[i] * normals[i][1])

    for _ in range(RACING_LINE_SWEEPS):
        for i in range(m):
            (x1, y1), (x2, y2) = pt(i - 1), pt(i + 1)
            (x0, y0), (x3, y3) = pt(i - 2), pt(i + 2)
            tx = (4.0 * (x1 + x2) - (x0 + x3)) / 6.0
            ty = (4.0 * (y1 + y2) - (y0 + y3)) / 6.0
            target = (tx - center[i][0]) * normals[i][0] + (ty - center[i][1]) * normals[i][1]
            a = alpha[i] + RACING_LINE_RELAX * (target - alpha[i])
            alpha[i] = max(-limits[i], min(limits[i], a))
    return [pt(i) for i in range(m)]


def _speed_profile(
    line: list[Point], max_speed: float, turn_rate: float, accel: float, brake: float
) -> list[float]:
    """Fastest speed at each point: cornering limit v = turn_rate * R, then accel/brake passes."""
    m = len(line)
    v = [
        min(max_speed, turn_rate * _circle_radius(line[i - 2], line[i], line[(i + 2) % m]))
        for i in range(m)
    ]
    ds = [math.dist(line[i - 1], line[i]) for i in range(m)]
    for _ in range(2):  # twice so the passes wrap around the lap
        for i in range(m):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * accel * ds[i]))
        for i in range(m - 1, -1, -1):
            j = (i + 1) % m
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * brake * ds[j]))
    return v


def _lerp_color(a, b, t: float) -> tuple[int, int, int, int]:
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(4))


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
    """Closed circuit from LAYOUTS: boundary walls, ordered checkpoint gates, and rendering."""

    def __init__(self, shape: int = 0):
        self.set_shape(shape)

    @property
    def shape_name(self) -> str:
        return LAYOUTS[self.current_shape][0]

    def cycle_shape(self) -> None:
        """Advance to the next layout and rebuild geometry."""
        self.set_shape(self.current_shape + 1)

    def set_shape(self, shape_index: int) -> None:
        self.current_shape = shape_index % len(LAYOUTS)
        self._decor = []
        self._build(LAYOUTS[self.current_shape][1])
        self._segment_grid = self._build_segment_grid()
        self._racing_line = None

    def get_spawn_info(
        self, checkpoint_idx: int
    ) -> tuple[tuple[float, float], float]:
        """Grid slot just behind a checkpoint gate's midpoint, facing the race direction."""
        idx = checkpoint_idx % len(self.checkpoints)
        (i1, i2) = self.checkpoints[idx]
        tx, ty = self.checkpoint_tangents[idx]
        spawn_x = (i1[0] + i2[0]) * 0.5 - tx * SPAWN_BEHIND_LINE
        spawn_y = (i1[1] + i2[1]) * 0.5 - ty * SPAWN_BEHIND_LINE
        return ((spawn_x, spawn_y), math.atan2(ty, tx))

    def gate_progress(self, position, target_checkpoint: int) -> float:
        """Fraction (0-1) of the way from the previous gate to the target gate, along their midpoints."""
        n = len(self.checkpoints)
        (a1, a2), (b1, b2) = self.checkpoints[(target_checkpoint - 1) % n], self.checkpoints[target_checkpoint]
        ax, ay = (a1[0] + a2[0]) * 0.5, (a1[1] + a2[1]) * 0.5
        bx, by = (b1[0] + b2[0]) * 0.5, (b1[1] + b2[1]) * 0.5
        dx, dy = bx - ax, by - ay
        t = ((position[0] - ax) * dx + (position[1] - ay) * dy) / max(1e-9, dx * dx + dy * dy)
        return max(0.0, min(1.0, t))

    def _build(self, corners: list[tuple[float, float, float, float]]) -> None:
        """Offset the filleted centerline into walls, gates, asphalt, kerbs and start line."""
        center, widths, signs, _, main_s, main_len = _fillet_centerline(corners, SAMPLE_STEP)
        tangents = _tangents(center)
        self._centerline = (center, tangents, widths)
        n = len(center)
        side_a: list[Point] = []
        side_b: list[Point] = []
        for (x, y), (tx, ty), w in zip(center, tangents, widths):
            h = w * 0.5
            side_a.append((x - ty * h, y + tx * h))
            side_b.append((x + ty * h, y - tx * h))

        self._boundary_segments = _closed_loop_segments(
            side_a, SIMPLIFY_TOLERANCE
        ) + _closed_loop_segments(side_b, SIMPLIFY_TOLERANCE)

        s_along = _cumulative_lengths(center)
        total = s_along[-1] + math.dist(center[-1], center[0])
        n_gates = max(1, round(total / GATE_SPACING))
        start_s = main_s + min(START_LINE_OFFSET, 0.4 * main_len)
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
            near_exit = signs[(i + EXIT_KERB_SAMPLES) % n] != sign
            tx, ty = tangents[i]
            d = KERB_DEPTH
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
                color = KERB_RED if (i // KERB_BLOCK_SAMPLES) % 2 == 0 else KERB_WHITE
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
        """Draw the asphalt, kerbs, start line and boundaries."""
        rl.DrawTriangleStrip(self._asphalt[0], self._asphalt[1], ASPHALT_COLOR)
        for arr, count, color in self._decor:
            rl.DrawTriangleStrip(arr, count, color)

        line_thick = 3.0
        line_color = colors.WHITE

        for (p1, p2) in self._boundary_segments:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineEx(start, end, line_thick, line_color)

    def racing_line(
        self, max_speed: float, turn_rate: float, accel: float, brake: float
    ) -> tuple[list[Segment], list[tuple[int, int, int, int]], float]:
        """Minimum-curvature line colored by the car's achievable speed, plus its ideal lap time.

        Computed on first use per layout (about a second) and cached.
        """
        if self._racing_line is None:
            center, tangents, widths = self._centerline
            step = max(1, round(RACING_LINE_SPACING / SAMPLE_STEP))
            idx = range(0, len(center), step)
            line = _min_curvature_line(
                [center[i] for i in idx],
                [(-tangents[i][1], tangents[i][0]) for i in idx],
                [max(0.0, widths[i] * 0.5 - RACING_LINE_MARGIN) for i in idx],
            )
            v = _speed_profile(line, max_speed, turn_rate, accel, brake)
            v_lo, v_hi = min(v), max(v)
            segs, cols = [], []
            lap_time = 0.0
            for i in range(len(line)):
                j = (i + 1) % len(line)
                segs.append((line[i], line[j]))
                t = (0.5 * (v[i] + v[j]) - v_lo) / max(1e-9, v_hi - v_lo)
                cols.append(
                    _lerp_color(RACING_LINE_SLOW, RACING_LINE_MID, t * 2.0)
                    if t < 0.5
                    else _lerp_color(RACING_LINE_MID, RACING_LINE_FAST, t * 2.0 - 1.0)
                )
                lap_time += math.dist(line[i], line[j]) / max(1e-9, 0.5 * (v[i] + v[j]))
            self._racing_line = (segs, cols, lap_time)
        return self._racing_line

    def render_racing_line(
        self, max_speed: float, turn_rate: float, accel: float, brake: float
    ) -> None:
        """Draw the racing line: red where the car must be slow, green where it can be flat out."""
        segs, cols, _ = self.racing_line(max_speed, turn_rate, accel, brake)
        for ((x1, y1), (x2, y2)), color in zip(segs, cols):
            rl.DrawLineEx([x1, y1], [x2, y2], 2.5, color)

    def render_checkpoints(self) -> None:
        """Draw checkpoint gate segments (toggle in sim)."""
        ck_color = getattr(colors, "SKYBLUE", getattr(colors, "BLUE", (102, 191, 255, 255)))
        for (p1, p2) in self.checkpoints:
            start = [float(p1[0]), float(p1[1])]
            end = [float(p2[0]), float(p2[1])]
            rl.DrawLineV(start, end, ck_color)
