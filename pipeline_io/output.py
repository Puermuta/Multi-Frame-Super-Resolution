import cv2
import numpy as np

def _linear_to_srgb(image):
    # sRGB gamma curve
    mask = image <= 0.0031308
    image[mask] = image[mask] * 12.92
    image[~mask] = 1.055 * np.power(image[~mask], 1.0 / 2.4) - 0.055
    
    # Back to uint8
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

def output(image):
    output = _linear_to_srgb(image)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.png", output)