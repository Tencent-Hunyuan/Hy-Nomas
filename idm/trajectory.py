"""Map temporal camera classifications and velocities to camera-to-world poses.

Coordinates are right/down/forward. Angular velocity channels are yaw-right,
pitch-down and roll-clockwise in degrees per second. Translation retains the
model's native scale. Zoom and shake do not define an additional rigid motion.
"""
import numpy as np

from .postprocessing import select_rotation_sector


def classified_velocity(motion_prob, velocity6, threshold=0.5, rotation_threshold=0.5):
    """Use class decisions for direction and absolute velocity for magnitude.

Opposing translations and rolls choose the higher probability above threshold;
ties use the first channel. Pan/tilt share the exclusive eight-sector decision.
"""
    motion = np.asarray(motion_prob, dtype=np.float64)
    velocity = np.asarray(velocity6, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 18 or velocity.shape != (len(motion), 6):
        raise ValueError('Expected motion [N,18] and velocity [N,6].')
    if not np.isfinite(motion).all() or not np.isfinite(velocity).all():
        raise ValueError('Motion and velocity must be finite.')
    if np.any((motion < 0) | (motion > 1)):
        raise ValueError('Motion probabilities must be in [0,1].')
    if not 0 <= threshold <= 1 or not 0 <= rotation_threshold <= 1:
        raise ValueError('Thresholds must be in [0,1].')
    signs = np.zeros_like(velocity)
    for axis, channels, directions in [
        (0, (1, 3), (-1, 1)), (1, (13, 14), (-1, 1)),
        (2, (0, 2), (1, -1)), (5, (16, 17), (1, -1)),
    ]:
        probabilities = motion[:, channels]
        selected = probabilities.argmax(axis=1)
        active = probabilities[np.arange(len(motion)), selected] >= threshold
        signs[:, axis] = np.asarray(directions)[selected] * active
    yaw_pitch = np.asarray([(1, 0), (1, -1), (0, -1), (-1, -1),
                            (-1, 0), (-1, 1), (0, 1), (1, 1)])
    for i, probabilities in enumerate(motion[:, 4:12]):
        sector = select_rotation_sector(probabilities, rotation_threshold)
        if sector >= 0:
            signs[i, 3:5] = yaw_pitch[sector]
    return np.abs(velocity) * signs


def se3_increment(velocity6, dt):
    """Exact constant body-twist integration over dt seconds, in float64."""
    velocity = np.asarray(velocity6, dtype=np.float64)
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise ValueError('Expected six finite velocity components.')
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('dt must be finite and nonnegative.')
    # Positive pitch-down is negative rotation about the camera's right axis.
    phi = np.deg2rad([-velocity[4], velocity[3], velocity[5]]) * dt
    theta2 = float(phi @ phi)
    x, y, z = phi
    omega = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    omega2 = omega @ omega
    if theta2 < 1e-8:
        a = 1 - theta2 / 6 + theta2**2 / 120
        b = 0.5 - theta2 / 24 + theta2**2 / 720
        c = 1 / 6 - theta2 / 120 + theta2**2 / 5040
    else:
        theta = np.sqrt(theta2)
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta2
        c = (theta - np.sin(theta)) / (theta2 * theta)
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) + a * omega + b * omega2
    transform[:3, 3] = (np.eye(3) + b * omega + c * omega2) @ (velocity[:3] * dt)
    return transform


def integrate_trajectory(velocity6, bucket_start_seconds, timestamps_seconds):
    """Integrate piecewise-constant body velocities at arbitrary sample times.

The initial pose at the first bucket start is identity. The last bucket extends
through the last requested timestamp. Timestamps must be strictly increasing.
"""
    velocity = np.asarray(velocity6, dtype=np.float64)
    starts = np.asarray(bucket_start_seconds, dtype=np.float64)
    times = np.asarray(timestamps_seconds, dtype=np.float64)
    if (starts.ndim != 1 or len(starts) == 0 or velocity.shape != (len(starts), 6)
            or times.ndim != 1 or len(times) == 0):
        raise ValueError('Expected nonempty bucket starts, [N,6] velocities and timestamps.')
    if not all(np.isfinite(x).all() for x in (velocity, starts, times)):
        raise ValueError('Trajectory inputs must be finite.')
    if np.any(np.diff(starts) <= 0) or np.any(np.diff(times) <= 0) or times[0] < starts[0]:
        raise ValueError('Times must increase and cannot precede the first bucket.')
    poses = []
    pose, current, bucket = np.eye(4), float(starts[0]), 0
    for target in times:
        while current < target:
            stop = min(target, starts[bucket + 1]) if bucket + 1 < len(starts) else target
            pose = pose @ se3_increment(velocity[bucket], float(stop - current))
            current = float(stop)
            if bucket + 1 < len(starts) and current >= starts[bucket + 1]:
                bucket += 1
        poses.append(pose.copy())
    return np.asarray(poses)
