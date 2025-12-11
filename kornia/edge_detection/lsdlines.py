import cv2
import numpy as np
import kornia
import sys

filename = sys.argv[1]

# Load image as a tensor
image = cv2.imread(filename)
image = kornia.image_to_tensor(image)

# Convert the image to grayscale
gray = kornia.color.rgb_to_grayscale(image)

# Use the LSDDetector to detect lines
lsd = kornia.line_detector.LSDDetector()
lines, widths, _ = lsd(gray)

# Convert the lines to endpoints
lines_endpoints = kornia.line_detector.lsd_to_lines(lines, widths)

# Draw the lines on the image
image_lines = kornia.drawing.draw_lines(image, lines_endpoints)

# Save the image with the lines drawn
kornia.imwrite('image_lines.jpg', image_lines)
