"""Single-image pipeline, with no fabricated model output."""
from pathlib import Path

from src.preprocessing.interfaces import VideoFrame
from src.pose_estimation.visualization import draw_poses


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def validate_image_paths(input_path, output_path):
    source, target = Path(input_path), Path(output_path)
    if source.suffix.lower() not in IMAGE_SUFFIXES or target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError('Image input/output must be JPG, JPEG or PNG')
    if not source.is_file():
        raise FileNotFoundError(f'Image not found: {source}')
    if source.resolve() == target.resolve():
        raise ValueError('Input and output image paths must differ')
    if target.exists():
        raise FileExistsError('Output already exists; choose a new image path')
    return source, target


def process_image(input_path, output_path, detector, estimator, threshold=0.5, cv_module=None):
    source, target = validate_image_paths(input_path, output_path)
    if cv_module is None:
        import cv2 as cv_module
    pixels = cv_module.imread(str(source), cv_module.IMREAD_COLOR)
    if pixels is None:
        raise ValueError('Image could not be decoded')
    frame = VideoFrame(0, 0.0, pixels)
    dogs = detector.detect(frame)
    poses = estimator.estimate(frame, dogs)
    annotated = draw_poses(pixels, dogs, poses, estimator.schema, threshold,
                           cv_module, show_confidence=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv_module.imwrite(str(target), annotated):
        target.unlink(missing_ok=True)
        raise RuntimeError('Could not write output image')
    return {'dogs': len(dogs), 'keypoints_per_dog': len(estimator.schema.names),
            'skeleton_edges': len(estimator.schema.edges), 'output': str(target)}
