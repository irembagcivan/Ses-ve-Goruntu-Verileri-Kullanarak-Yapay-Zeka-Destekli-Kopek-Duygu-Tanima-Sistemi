"""Project-authored 2D dog visualization, NOT an official SuperAnimal skeleton.

Names are model labels, not a veterinary bone/joint nomenclature. Legacy
chains are retained for compatibility; conservative excludes uncertain regions.
Neither profile claims a validated veterinary skeleton.
"""
from functools import lru_cache

from src.pose_estimation.metadata import PoseSchema

# Never substitute guessed integer indices. Keep upstream label spelling (`thai`).
DOG_VISUALIZATION_EDGES = (
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_earbase'),
    ('right_eye', 'right_earbase'),
    ('left_earbase', 'left_earend'),
    ('right_earbase', 'right_earend'),
    ('neck_base', 'neck_end'),
    ('back_base', 'back_middle'),
    ('back_middle', 'back_end'),
    ('tail_base', 'tail_end'),
    ('front_left_thai', 'front_left_knee'),
    ('front_left_knee', 'front_left_paw'),
    ('front_right_thai', 'front_right_knee'),
    ('front_right_knee', 'front_right_paw'),
    ('back_left_thai', 'back_left_knee'),
    ('back_left_knee', 'back_left_paw'),
    ('back_right_thai', 'back_right_knee'),
    ('back_right_knee', 'back_right_paw'),
)
DOG_VISUALIZATION_NAMES = frozenset(name for edge in DOG_VISUALIZATION_EDGES for name in edge)


# Conservative profile excludes labels whose anatomical annotation convention is
# not described by the official metadata (neck/back orientation and limb joints).
CONSERVATIVE_NAMES = frozenset({
    'nose', 'left_eye', 'right_eye', 'left_earbase', 'right_earbase',
    'left_earend', 'right_earend', 'tail_base', 'tail_end',
})
CONSERVATIVE_EDGES = tuple(edge for edge in DOG_VISUALIZATION_EDGES
                           if set(edge).issubset(CONSERVATIVE_NAMES))


def validate_visualization_edges(names, edges):
    """Structural QC only: valid labels do not establish anatomical truth."""
    if len(set(names)) != len(names):
        raise ValueError('Dog visualization requires unique keypoint names')
    known = set(names)
    seen = set()
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError('Each visualization edge must contain two names')
        a, b = edge
        if not isinstance(a, str) or not isinstance(b, str) or a not in known or b not in known:
            raise ValueError('Unknown visualization keypoint name')
        if a == b:
            raise ValueError('Self connections are not allowed')
        if 'antler' in a or 'antler' in b:
            raise ValueError('Antler labels are excluded from dog visualization')
        pair = frozenset(edge)
        if pair in seen:
            raise ValueError('Duplicate visualization connection')
        seen.add(pair)


def visualization_profile(mode):
    if mode is True or mode == 'legacy':
        return 'legacy'
    if mode == 'conservative':
        return 'conservative'
    raise ValueError('Unknown dog skeleton visualization profile')


@lru_cache(maxsize=8)
def dog_visualization_schema(names, profile='legacy'):
    """Resolve names without modifying upstream indices or official skeleton.

    Legacy preserves Task 5 output. Conservative is a reduced surface-marker
    guide, not a newly validated veterinary skeleton. Neither corrects predictions.
    """
    if profile not in {'legacy', 'conservative'}:
        raise ValueError('Unknown dog skeleton visualization profile')
    edges = DOG_VISUALIZATION_EDGES if profile == 'legacy' else CONSERVATIVE_EDGES
    validate_visualization_edges(names, edges)
    lookup = {name: index for index, name in enumerate(names)}
    return PoseSchema(names, tuple((lookup[a], lookup[b]) for a,b in edges))
