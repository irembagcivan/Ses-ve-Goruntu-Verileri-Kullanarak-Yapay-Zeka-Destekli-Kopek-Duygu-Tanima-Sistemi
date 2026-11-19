"""Draw model edges by default; optional project-authored dog visualization."""
import math


def visible_keypoints(pose, threshold=0.5, size=None):
    if not 0 <= threshold <= 1:
        raise ValueError("Keypoint threshold must be in [0, 1]")
    if not len(pose.xy) == len(pose.confidence) == len(pose.keypoint_names):
        raise ValueError("Inconsistent pose lengths")
    return tuple(math.isfinite(x) and math.isfinite(y) and math.isfinite(score)
                 and threshold <= score <= 1
                 and (size is None or 0 <= x < size[0] and 0 <= y < size[1])
                 for (x, y), score in zip(pose.xy, pose.confidence))


def draw_poses(image, dogs, poses, schema, threshold=0.5, cv_module=None, show_confidence=False, dog_skeleton=False):
    if cv_module is None:
        import cv2 as cv_module
    if len(dogs) != len(poses):
        raise ValueError("Detection/pose count mismatch")
    drawing_schema = schema
    selected_names = None
    if dog_skeleton:
        from src.pose_estimation.dog_skeleton import dog_visualization_schema, visualization_profile
        drawing_schema = dog_visualization_schema(schema.names, visualization_profile(dog_skeleton))
        selected_names = frozenset(schema.names[i] for edge in drawing_schema.edges for i in edge)
    canvas = image.copy()
    height, width = canvas.shape[:2]
    def pixel_point(point):
        x, y = map(round, point)
        # Rasterization only: an in-frame float near the last pixel can round
        # outside the raster. Never alter the underlying model coordinates.
        return (min(width-1, max(0, x)), min(height-1, max(0, y))) if dog_skeleton else (x, y)
    for dog, pose in zip(dogs, poses):
        if pose.keypoint_names != schema.names:
            raise ValueError("Pose names disagree with verified skeleton")
        valid = visible_keypoints(pose, threshold, (width, height))
        if selected_names is not None:
            valid = tuple(keep and name in selected_names
                          for keep, name in zip(valid, pose.keypoint_names))
        x1, y1, x2, y2 = map(round, dog.xyxy)
        cv_module.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv_module.putText(canvas, f"dog {dog.confidence:.2f}", (max(0,x1), max(12,y1)),
                          cv_module.FONT_HERSHEY_SIMPLEX, 0.4, (255,180,0), 1)
        for a, b in drawing_schema.edges:
            if valid[a] and valid[b]:
                cv_module.line(canvas, pixel_point(pose.xy[a]), pixel_point(pose.xy[b]), (0,255,0), 2)
        for point, score, keep in zip(pose.xy, pose.confidence, valid):
            if keep:
                cv_module.circle(canvas, pixel_point(point), 3, (0,0,255), -1)
                if show_confidence:
                    cv_module.putText(canvas, f"{score:.2f}", pixel_point(point),
                                      cv_module.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
    return canvas
