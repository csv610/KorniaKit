"""Load an image and display its tensor representation.

Demonstrates:
  - Reading an image from disk with OpenCV.
  - Converting to a Kornia tensor and inspecting its shape.
  - Converting back to a numpy array for display.
"""

from tutorials._utils import from_tensor, load_image, parse_args, to_tensor


def main() -> None:
    """Load an image, convert to tensor, display shapes, and show results."""
    (filename,) = parse_args(1, "<image_path>")
    img = load_image(filename)

    import cv2
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    tensor = to_tensor(img, add_batch=True)
    print("Tensor shape (with batch):", tensor.shape)
    tensor_no_batch = tensor.squeeze(0)
    print("Tensor shape (no batch):", tensor_no_batch.shape)

    reconstructed = from_tensor(tensor)
    cv2.imshow("Reconstructed", reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
