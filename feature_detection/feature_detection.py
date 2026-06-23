import cv2
import numpy as np

def get_features(images, reference_index = -1, debug = False):
    """
    Wrapper for the goodFeaturesToTrack method from opencv2.
    Returns a list of tuples (x, y) for features to track.
    """
    if reference_index == -1:
        reference_index = len(images) // 2
    reference_image = images[reference_index]

    # cv2 requires a bw image
    ref_bw = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    features= cv2.goodFeaturesToTrack(
        ref_bw,
        maxCorners = 200,
        qualityLevel = 0.01,
        minDistance = 10
    )

    return features