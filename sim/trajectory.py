"""
Probe trajectory classes.
Ported from the breast-palpation project (unchanged).

Each trajectory returns (vx, vy) velocity at time t given dt steps.
"""
import math


class Trajectory:
    def __init__(self, T, x0, y0):
        self.T = T
        self.current_time = 0
        self.x0 = x0
        self.y0 = y0

    def get_velocity(self, t) -> tuple[float, float]:
        raise NotImplementedError

    def get_initial_position(self) -> tuple[float, float]:
        return (self.x0, self.y0)

    def step(self, dt) -> tuple[float, float]:
        velocity = self.get_velocity(self.current_time)
        self.current_time += dt
        return velocity

    @staticmethod
    def from_dict(trajectory_type: str, params: dict) -> "Trajectory":
        registry = {
            "FixedVelocityTrajectory": FixedVelocityTrajectory,
            "ReturnTrajectory": ReturnTrajectory,
            "TwoPointTrajectory": TwoPointTrajectory,
            "ThreePointTrajectory": ThreePointTrajectory,
            "FourPointTrajectory": FourPointTrajectory,
            "StaticTrajectory": StaticTrajectory,
            "PiecewiseLinearTrajectory": PiecewiseLinearTrajectory,
        }
        cls = registry.get(trajectory_type)
        if cls is None:
            raise ValueError(f"Unknown trajectory type: {trajectory_type!r}")
        return cls(**params)


class StaticTrajectory(Trajectory):
    def __init__(self, T=1.0, x0=0.0, y0=0.0):
        super().__init__(T, x0, y0)

    def get_velocity(self, t):
        return (0.0, 0.0)


class FixedVelocityTrajectory(Trajectory):
    def __init__(self, T, x0, y0, vx, vy):
        super().__init__(T, x0, y0)
        self.vx = vx
        self.vy = vy

    def get_velocity(self, t):
        return (self.vx, self.vy) if 0 <= t <= self.T else (0.0, 0.0)


class ReturnTrajectory(Trajectory):
    """Go from (x0,y0) to (x1,y1), then back."""
    def __init__(self, T, x0, y0, x1, y1):
        super().__init__(T, x0, y0)
        self.x1, self.y1 = x1, y1

    def get_velocity(self, t):
        if 0 <= t <= self.T / 2:
            return (2 * (self.x1 - self.x0) / self.T, 2 * (self.y1 - self.y0) / self.T)
        elif t <= self.T:
            return (2 * (self.x0 - self.x1) / self.T, 2 * (self.y0 - self.y1) / self.T)
        return (0.0, 0.0)


class TwoPointTrajectory(Trajectory):
    """Move from (x0,y0) to (x1,y1) over T."""
    def __init__(self, T, x0, y0, x1, y1):
        super().__init__(T, x0, y0)
        self.x1, self.y1 = x1, y1

    def get_velocity(self, t):
        if 0 <= t <= self.T:
            return ((self.x1 - self.x0) / self.T, (self.y1 - self.y0) / self.T)
        return (0.0, 0.0)


class ThreePointTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1, x2, y2):
        super().__init__(T, x0, y0)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def get_velocity(self, t):
        if 0 <= t <= self.T / 2:
            return (2 * (self.x1 - self.x0) / self.T, 2 * (self.y1 - self.y0) / self.T)
        elif t <= self.T:
            return (2 * (self.x2 - self.x1) / self.T, 2 * (self.y2 - self.y1) / self.T)
        return (0.0, 0.0)


class FourPointTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1, x2, y2, x3, y3):
        super().__init__(T, x0, y0)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.x3, self.y3 = x3, y3

    def get_velocity(self, t):
        if 0 <= t <= self.T / 3:
            return (3 * (self.x1 - self.x0) / self.T, 3 * (self.y1 - self.y0) / self.T)
        elif t <= 2 * self.T / 3:
            return (3 * (self.x2 - self.x1) / self.T, 3 * (self.y2 - self.y1) / self.T)
        elif t <= self.T:
            return (3 * (self.x3 - self.x2) / self.T, 3 * (self.y3 - self.y2) / self.T)
        return (0.0, 0.0)


class PiecewiseLinearTrajectory(Trajectory):
    def __init__(self, T, points):
        super().__init__(T, points[0][0], points[0][1])
        self.points = points
        self.interval = T / (len(points) - 1)

    def get_velocity(self, t):
        seg = int(t / self.interval)
        if seg < len(self.points) - 1:
            x0, y0 = self.points[seg]
            x1, y1 = self.points[seg + 1]
            return ((x1 - x0) / self.interval, (y1 - y0) / self.interval)
        return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Trajectory helper functions for the arthroscopy probe scan
# ---------------------------------------------------------------------------

def poke_trajectory_points(
    x_pos: float,
    tissue_top_y: float,
    hover_height: float = 0.1,
    penetration_depth: float = 0.05,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Returns [start, end] for a (possibly tilted) poke at a given x position.

    Args:
        x_pos: x coordinate of the poke (at the START/hover point)
        tissue_top_y: y coordinate of the tissue surface
        hover_height: how far above tissue the probe starts
        penetration_depth: how deep below the surface the probe goes
            (penetration is measured vertically, i.e. end_y = tissue_top_y -
            penetration_depth, regardless of angle)
        angle_deg: tilt of the poke direction from vertical, in degrees.
            0 = straight down. Positive angles tilt the approach direction
            toward +x, so the end point is shifted in +x.
    """
    start = (x_pos, tissue_top_y + hover_height)
    if angle_deg == 0.0:
        end = (x_pos, tissue_top_y - penetration_depth)
    else:
        theta = math.radians(angle_deg)
        # Travel along the tilted direction d = (sin(theta), -cos(theta)) such
        # that the vertical drop equals hover_height + penetration_depth.
        vertical_drop = hover_height + penetration_depth
        x_shift = vertical_drop * math.tan(theta)
        end = (x_pos + x_shift, tissue_top_y - penetration_depth)
    return [start, end]


def return_poke_points(
    x_pos: float,
    tissue_top_y: float,
    hover_height: float = 0.1,
    penetration_depth: float = 0.05,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """Poke-and-return: go down (possibly tilted) and come back up."""
    start, bottom = poke_trajectory_points(
        x_pos, tissue_top_y, hover_height, penetration_depth, angle_deg
    )
    return [start, bottom, start]
