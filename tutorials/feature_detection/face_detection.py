"""Script that finds faces and blurs using FaceDetection and blurring APIs."""
import argparse

import cv2
import kornia as K
import numpy as np
import torch
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint


def draw_keypoint(img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


def scale_image(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.0 * size / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def apply_blur_face(img: torch.Tensor, img_vis: np.ndarray, det: FaceDetectorResult) -> None:
    x1, y1 = det.xmin.int(), det.ymin.int()
    x2, y2 = det.xmax.int(), det.ymax.int()
    roi = img[..., y1:y2, x1:x2]

    roi = K.filters.gaussian_blur2d(roi, (21, 21), (35.0, 35.0))
    roi = K.color.rgb_to_bgr(roi)
    img_vis[y1:y2, x1:x2] = K.tensor_to_image(roi)


def my_app(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")

    img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_raw is None:
        print(f"Error: could not load image {args.image}")
        return

    img = K.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    face_detection = FaceDetector().to(device)

    with torch.no_grad():
        dets = face_detection(img)

    dets = [FaceDetectorResult(o) for o in dets[0]]

    img_vis = img_raw.copy()

    for b in dets:
        if b.score < args.vis_threshold:
            continue

        img_vis = cv2.rectangle(img_vis, b.top_left.int().tolist(), b.bottom_right.int().tolist(), (0, 255, 0), 2)

        if args.blur_faces:
            apply_blur_face(img, img_vis, b)

        if args.vis_keypoints:
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_LEFT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_RIGHT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.NOSE)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_LEFT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_RIGHT)

            cx = int(b.xmin)
            cy = int(b.ymin + 12)
            img_vis = cv2.putText(img_vis, f"{b.score:.2f}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    cv2.namedWindow("face_detection", cv2.WINDOW_NORMAL)
    cv2.imshow("face_detection", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face and Landmark Detection")
    parser.add_argument("--image", required=True, type=str, help="the image file to be detected")
    parser.add_argument("--vis_threshold", default=0.8, type=float, help="visualization threshold")
    parser.add_argument("--vis_keypoints", dest="vis_keypoints", action="store_true")
    parser.add_argument("--cuda", dest="cuda", action="store_true")
    parser.add_argument("--blur_faces", dest="blur_faces", action="store_true")
    args = parser.parse_args()
    my_app(args)
