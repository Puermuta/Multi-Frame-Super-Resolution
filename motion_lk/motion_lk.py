import cv2
import numpy as np

def _to_8bit(image):
    normalized = cv2.normalize(image, np.zeros_like(image), 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def get_optical_flow(images, features, reference_index = -1, debug = False):
    """
    Does an optical flow analysis across all images except for the 
    reference image using the Lukas-Kanade sparse optical flow algorithm.
    Returns a list of value tuples which combined construct a vector field.
        Ex: [..., ( (x, y), (dx, dy) ), ...]
    """
    if reference_index == -1:
        reference_index = len(images) // 2
    reference_image = images[reference_index]

    ref_bw = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    ref_bw_8 = _to_8bit(ref_bw)

    features = np.array(features, dtype = np.float32)

    results = []
    for index, target in enumerate(images):
        if debug:
            print(f"LK-optical flow: {index}")
        if index == reference_index:
            # Fills a blank vector field with the correct shape 
            # for the reference image.
            results.append((np.zeros((len(features), 2)), np.zeros((len(features), 2))))
            continue

        target_bw = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # LK requires 8-bit datasets, so we do that.
        # TODO: do custom implementation that allows 16-bit datasets.
        target_bw_8 = _to_8bit(target_bw)  

        next_pts = np.zeros_like(features)
        tracked_features, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg = ref_bw_8,
            nextImg = target_bw_8,
            prevPts = features,
            nextPts = next_pts,
            winSize = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_reference = features[status == 1]
        good_tracked = tracked_features[status == 1]
        displacement = good_tracked - good_reference

        if debug:
            print(good_reference)
            print(good_tracked)
            print(displacement)
            print()
            #cv2.imwrite("debug_ref.png", ref_bw_8)
            #cv2.imwrite("debug_target.png", target_bw_8)

        results.append((good_reference, displacement))

    return results
        




