import argparse

import cv2
import kornia as K
import torch
from kornia.contrib import EdgeDetectorBuilder


def cnn_edge_detector() -> None:
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")

    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    if img_raw is None:
        print(f"Error: could not load image {args.image_file}")
        return

    img = K.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    edge_detection = EdgeDetectorBuilder.build().to(device)

    with torch.no_grad():
        edges = edge_detection(img)

    img_vis = K.tensor_to_image(edges.byte())

    cv2.namedWindow("edge_detection", cv2.WINDOW_NORMAL)
    cv2.imshow("edge_detection", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Edge Detection")
    parser.add_argument("--image_file", required=True, type=str, help="the image file to be detected")
    parser.add_argument("--cuda", dest="cuda", action="store_true")
    args = parser.parse_args()
    cnn_edge_detector()
