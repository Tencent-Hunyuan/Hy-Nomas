"""Threshold-based decisions derived from native model probabilities."""

DEFAULT_SHAKE_THRESHOLD = 0.7
DEFAULT_ROTATION_THRESHOLD = 0.5


def validate_shake_threshold(threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('SHAKE threshold must be between 0 and 1.')
    return float(threshold)


def shake_decisions(probabilities, threshold=DEFAULT_SHAKE_THRESHOLD):
    threshold = validate_shake_threshold(threshold)
    return {'shake_threshold': threshold,
            'shake_active': [probability >= threshold for probability in probabilities]}


def validate_rotation_threshold(threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Rotation threshold must be between 0 and 1.')
    return float(threshold)


def select_rotation_sector(probabilities, threshold=DEFAULT_ROTATION_THRESHOLD):
    """Return one of eight sectors, or -1 when none passes the threshold."""
    threshold = validate_rotation_threshold(threshold)
    if len(probabilities) != 8:
        raise ValueError('Expected eight rotation probabilities.')
    # max keeps the first channel when probabilities tie.
    winner = max(range(8), key=lambda index: probabilities[index])
    return winner if probabilities[winner] >= threshold else -1


def rotation_decisions(probabilities, threshold=DEFAULT_ROTATION_THRESHOLD):
    threshold = validate_rotation_threshold(threshold)
    return {'rotation_threshold': threshold,
            'rotation_sector': [select_rotation_sector(row, threshold) for row in probabilities]}
