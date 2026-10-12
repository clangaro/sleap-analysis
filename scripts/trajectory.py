"""Helpers shared by the metrics scripts that read trajectory CSVs."""

import math


def parse_coord(value):
    """Return a CSV coordinate as a float, or NaN when the cell is empty or 'nan'."""
    return float(value) if value not in ('', 'nan') else float('nan')


def angle_clockwise_from_vertical(dx, dy):
    """Angle of (dx, dy) from image-vertical-up, clockwise positive, in [0, 360)."""
    a = math.degrees(math.atan2(dx, -dy))
    return a + 360.0 if a < 0 else a


def nose_path_length(rows):
    """Sum of nose displacements between consecutive valid frames.

    A missing frame breaks the path, so the step across a gap is not counted.
    Returns (total_px, n_steps).
    """
    total = 0.0
    n = 0
    px, py = None, None
    for r in rows:
        x, y = r['nose_x'], r['nose_y']
        if x != x or y != y:
            px, py = None, None
            continue
        if px is not None:
            total += math.hypot(x - px, y - py)
            n += 1
        px, py = x, y
    return total, n
