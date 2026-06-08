"""Detect edges using a CNN-based model (DexiNed).

Demonstrates:
  - Loading a pre-trained edge detection model via EdgeDetectorBuilder.
  - GPU-accelerated inference with torch.no_grad.
  - Converting model output back to a viewable image.
"""

import argparse

import cv2
import kornia as K
import torch
from kornia.contrib import EdgeDetectorBuilder


def cnn_edge_detector() -> None:
    """Run CNN-based edge detection on the provided image and display results."""
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    if img_raw is None:
        print(f"Error: could not load image {args.image_file}")
        return

    img = K.image.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    detector = EdgeDetectorBuilder.build().to(device)

    with torch.no_grad():
        edges = detector(img)

    img_vis = K.tensor_to_image(edges.byte())

    cv2.namedWindow("edge_detection", cv2.WINDOW_NORMAL)
    cv2.imshow("edge_detection", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN-based edge detection")
    parser.add_argument("--image_file", required=True, type=str, help="path to input image")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="use GPU if available")
    args = parser.parse_args()
    cnn_edge_detector()
