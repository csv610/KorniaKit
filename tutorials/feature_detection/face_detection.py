"""Detect faces, visualise keypoints, and optionally blur faces.

Demonstrates:
  - Using Kornia's FaceDetector.
  - Drawing bounding boxes and facial keypoints.
  - Applying Gaussian blur to detected face regions.
"""

import argparse

import cv2
import kornia as K
import numpy as np
import torch
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint


def draw_keypoint(img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:
    """Draw a single facial keypoint on the image.

    Args:
        img: Image to draw on (modified in-place).
        det: Face detection result.
        kpt_type: Keypoint type to draw.

    Returns:
        Image with the keypoint drawn.
    """
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


def apply_blur_face(img: torch.Tensor, img_vis: np.ndarray, det: FaceDetectorResult) -> None:
    """Apply Gaussian blur to the face region.

    Args:
        img: Original image tensor (used for cropping the ROI).
        img_vis: Visualization array to write the blurred face into.
        det: Face detection result.
    """
    x1, y1 = det.xmin.int(), det.ymin.int()
    x2, y2 = det.xmax.int(), det.ymax.int()
    roi = img[..., y1:y2, x1:x2]
    roi = K.filters.gaussian_blur2d(roi, (21, 21), (35.0, 35.0))
    roi = K.color.rgb_to_bgr(roi)
    img_vis[y1:y2, x1:x2] = K.tensor_to_image(roi)


def my_app(args: argparse.Namespace) -> None:
    """Run face detection and optionally blur / annotate results.

    Args:
        args: Parsed CLI arguments.
    """
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_raw is None:
        print(f"Error: could not load image {args.image}")
        return

    img = K.image.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    detector = FaceDetector().to(device)

    with torch.no_grad():
        dets = detector(img)

    dets = [FaceDetectorResult(o) for o in dets[0]]
    vis = img_raw.copy()

    for b in dets:
        if b.score < args.vis_threshold:
            continue
        vis = cv2.rectangle(vis, b.top_left.int().tolist(), b.bottom_right.int().tolist(), (0, 255, 0), 2)

        if args.blur_faces:
            apply_blur_face(img, vis, b)

        if args.vis_keypoints:
            for kpt in (FaceKeypoint.EYE_LEFT, FaceKeypoint.EYE_RIGHT, FaceKeypoint.NOSE,
                        FaceKeypoint.MOUTH_LEFT, FaceKeypoint.MOUTH_RIGHT):
                vis = draw_keypoint(vis, b, kpt)
            cx, cy = int(b.xmin), int(b.ymin + 12)
            vis = cv2.putText(vis, f"{b.score:.2f}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX,
                              0.5, (255, 255, 255))

    cv2.namedWindow("face_detection", cv2.WINDOW_NORMAL)
    cv2.imshow("face_detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face and landmark detection")
    parser.add_argument("--image", required=True, type=str, help="path to input image")
    parser.add_argument("--vis_threshold", default=0.8, type=float, help="score threshold for display")
    parser.add_argument("--vis_keypoints", action="store_true", help="draw facial keypoints")
    parser.add_argument("--cuda", action="store_true", help="use GPU if available")
    parser.add_argument("--blur_faces", action="store_true", help="apply blur to detected faces")
    args = parser.parse_args()
    my_app(args)
