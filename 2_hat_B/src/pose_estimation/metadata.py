"""Fail-closed schema validation; never invent canine names or skeleton edges."""
from dataclasses import dataclass


class MetadataError(ValueError):
    pass


@dataclass(frozen=True)
class PoseSchema:
    names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            raise MetadataError("Verified checkpoint metadata mapping is required")
        names, skeleton = metadata.get("bodyparts"), metadata.get("skeleton")
        if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or not n for n in names):
            raise MetadataError("metadata.bodyparts must contain keypoint names")
        if len(set(names)) != len(names):
            raise MetadataError("Duplicate keypoint names")
        if not isinstance(skeleton, (list, tuple)) or not skeleton:
            raise MetadataError("Verified metadata.skeleton is missing; connections will not be guessed")
        edges = []
        for edge in skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(n not in names for n in edge) or edge[0] == edge[1]:
                raise MetadataError("Skeleton edges must reference two different bodypart names")
            edges.append((names.index(edge[0]), names.index(edge[1])))
        return cls(tuple(names), tuple(edges))


def load_superanimal_metadata():
    """Read facts extracted from the official DLC 3.0.2 project configuration.

    An empty official skeleton is valid and means points-only visualization.
    This is configuration evidence, not checkpoint content or weight licensing.
    """
    import json
    import hashlib
    import yaml
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'configs' / 'superanimal_quadruped_rtmpose_s.metadata.json'
    evidence = json.loads(path.read_text(encoding='utf-8'))
    metadata = evidence['metadata']
    official = path.parent / 'pose' / 'superanimal_quadruped.yaml'
    content = official.read_bytes()
    if hashlib.sha256(content).hexdigest() != evidence['sources']['project_config']['sha256']:
        raise MetadataError('Official project configuration checksum mismatch')
    project = yaml.safe_load(content)
    if metadata != {'bodyparts': project['bodyparts'], 'skeleton': project['skeleton']}:
        raise MetadataError('Extracted metadata disagrees with official project configuration')
    names = metadata.get('bodyparts')
    if not isinstance(names, list) or len(names) != 39 or len(set(names)) != 39:
        raise MetadataError('Official SuperAnimal metadata must define 39 unique bodyparts')
    if metadata.get('skeleton') == []:
        schema = PoseSchema(tuple(names), ())
    else:
        schema = PoseSchema.from_metadata(metadata)
    return evidence, schema
