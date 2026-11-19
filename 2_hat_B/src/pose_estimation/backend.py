"""Local-only SuperAnimal-Quadruped RTMPose-S integration for DLC 3.0.2.

Official configuration supplies keypoint order; checkpoints need not embed it.
No downloads, detector initialization, training, or video adaptation occur here.
"""
from copy import deepcopy
import hashlib
from importlib import metadata as packages
from pathlib import Path
import pickle
import yaml

from src.pose_estimation.device import inference_error
from src.pose_estimation.metadata import MetadataError, load_superanimal_metadata


def validate_checkpoint_path(checkpoint):
    path = Path(checkpoint)
    if path.suffix.lower() != '.pt' or not path.is_file():
        raise FileNotFoundError('Provide an existing local SuperAnimal RTMPose-S .pt checkpoint')
    if path.stat().st_size == 0:
        raise ValueError('Checkpoint file is empty')
    return path


def load_model_configuration(path, device='cpu'):
    """Accept the exact upstream base YAML, then add official project metadata.

    Keep `method: td`: upstream's convenience builder switches to bottom-up when
    detector_name=None, which is incorrect for externally supplied YOLO boxes.
    """
    evidence, schema = load_superanimal_metadata()
    content = Path(path).read_bytes()
    expected = evidence['sources']['model_config']['sha256']
    if hashlib.sha256(content).hexdigest() != expected:
        raise MetadataError('Model configuration provenance mismatch: use the unmodified DLC 3.0.2 rtmpose_s.yaml')
    config = yaml.safe_load(content)
    if (not isinstance(config, dict) or config.get('net_type') != 'rtmpose_s'
            or config.get('method') != 'td' or config.get('data', {}).get('colormode') != 'RGB'
            or config.get('model', {}).get('heads', {}).get('bodypart', {}).get('out_channels') != len(schema.names)):
        raise MetadataError('Invalid official RTMPose-S model configuration')
    config = deepcopy(config)
    config['device'] = device
    config['metadata'] = dict(bodyparts=list(schema.names), unique_bodyparts=[],
                              individuals=['animal0'], with_identity=False)
    config['runner']['load_weights_only'] = True
    # Do not attach a second detector; every crop supplies a single xywh box.
    config.pop('detector', None)
    return config, schema


def _default_model_config():
    return Path(__file__).resolve().parents[2] / 'configs' / 'pose' / 'rtmpose_s.yaml'


def pose_from_dlc_output(result, schema):
    """Convert postprocessed DLC bodyparts to the project's existing DogPose.

    Coordinates are crop-local pixels, already unscaled by DLC postprocessing.
    The repository has no separate PoseResult type; DogPose is its pose result.
    """
    import numpy as np
    from src.pose_estimation.interfaces import DogPose
    if not isinstance(result, (list, tuple)) or len(result) != 1 or not isinstance(result[0], dict) or 'bodyparts' not in result[0]:
        raise ValueError('Expected exactly one DLC result with bodyparts')
    points = np.asarray(result[0]['bodyparts'])
    if points.shape != (1, len(schema.names), 3) or not np.issubdtype(points.dtype, np.number):
        raise ValueError('Unexpected DLC bodyparts output shape or dtype')
    return DogPose(schema.names, tuple(tuple(map(float, row)) for row in points[0, :, :2]),
                   tuple(map(float, points[0, :, 2])))


class DeepLabCutRTMPose:
    def __init__(self, checkpoint, model_config=None, device='cpu', checkpoint_sha256=None):
        # Path checks precede heavy imports; CPU is the conservative default.
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or (model_config is not None and not Path(model_config).is_file()):
            raise FileNotFoundError('Local official checkpoint and matching DLC PyTorch YAML are required')
        if device not in {'cpu', 'mps'}:
            raise ValueError('Pose device must be cpu or mps')
        if model_config is not None:
            try:
                raw = yaml.safe_load(Path(model_config).read_text(encoding='utf-8'))
            except yaml.YAMLError as exc:
                raise MetadataError('Invalid DLC PyTorch configuration') from exc
            if not isinstance(raw, dict):
                raise MetadataError('Invalid DLC PyTorch configuration')
            # Diagnose old task-3 files without treating embedded labels as proof.
            if raw.get('net_type') is None:
                import torch
                try:
                    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=True)
                except pickle.UnpicklingError as exc:
                    raise MetadataError('Legacy checkpoint could not be safely inspected; use the official model configuration') from exc
                embedded = snapshot.get('metadata') if isinstance(snapshot, dict) else None
                if embedded is None:
                    raise MetadataError('Checkpoint has no verified metadata and config is not the official RTMPose-S configuration')
                raise MetadataError('Model configuration provenance mismatch; embedded labels cannot identify this model')
        checkpoint = validate_checkpoint_path(checkpoint)
        if checkpoint_sha256 is not None:
            if len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdefABCDEF' for c in checkpoint_sha256):
                raise ValueError('Expected checkpoint SHA-256 must have 64 hex characters')
            with checkpoint.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != checkpoint_sha256.lower():
                raise ValueError('Checkpoint SHA-256 mismatch')
        try:
            dlc_version = packages.version('deeplabcut')
        except packages.PackageNotFoundError as exc:
            raise ImportError('DeepLabCut 3.0.2 is not installed; see requirements-pose-macos.txt') from exc
        if dlc_version != '3.0.2':
            raise RuntimeError('This integration targets deeplabcut==3.0.2; other releases are unverified')
        config, self.schema = load_model_configuration(model_config or _default_model_config(), device)
        self.device = device
        try:
            from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
            # Runner loads snapshot['model'], moves to device, calls eval(), and
            # inference() uses torch.inference_mode. One runner/model per instance.
            runner = get_pose_inference_runner(model_config=config, snapshot_path=str(checkpoint),
                                               batch_size=1, device=device, max_individuals=1)
        except pickle.UnpicklingError as exc:
            raise RuntimeError('Checkpoint could not be loaded with weights_only=True; no unsafe pickle fallback') from exc
        except KeyError as exc:
            raise RuntimeError(f'Checkpoint/config does not match the verified DLC snapshot contract: {exc}') from exc
        except RuntimeError as exc:
            raise inference_error(exc, device) from exc
        self.adapter = VerifiedRunnerAdapter(runner, self.schema)
        if not self.schema.edges:
            import warnings
            warnings.warn('Official SuperAnimal skeleton is empty: keypoints will be drawn without anatomical connections', stacklevel=2)

    def predict_crop(self, crop):
        try:
            return self.adapter.predict_crop(crop)
        except RuntimeError as exc:
            raise inference_error(exc, self.device) from exc


class VerifiedRunnerAdapter:
    """DLC 3.0.2 runner receiving RGB crops and explicit COCO xywh boxes."""
    def __init__(self, runner, schema):
        self.runner, self.schema = runner, schema

    def predict_crop(self, crop):
        import numpy as np
        if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0 or crop.dtype != np.uint8:
            raise ValueError('Pose input must be a nonempty HxWx3 BGR image')
        height, width = crop.shape[:2]
        result = self.runner.inference(images=[(crop[:, :, ::-1].copy(), {
            'bboxes': np.asarray([[0, 0, width, height]], dtype=float),
            'bbox_scores': np.asarray([1.0]),
        })])
        pose = pose_from_dlc_output(result, self.schema)
        return pose.xy, pose.confidence
