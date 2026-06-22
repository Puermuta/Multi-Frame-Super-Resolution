import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy

def tranform_ecc(images):
    # Setting the middle image as reference, assuming it was created as the median in time as well.
    middle_index = len(images) // 2
    reference_image = images[middle_index]

    # Converting ref image to BW to work with tranformECC. Note! Might be an idea to separate color channels instead.
    ref_bw = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    ref_bw = np.nan_to_num(ref_bw)

    # Rescaling the image to a more workable size
    scale = 0.25
    ref_bw = cv2.resize(ref_bw, None, fx=scale, fy=scale)

    # Criteria for a good transform matrix
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

    aligned_images = []
    for idx, image in enumerate(images):
        print(idx)
        if idx == middle_index:
            continue
        
        img_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_bw = cv2.resize(img_bw, None, fx=scale, fy=scale)
        img_bw = np.nan_to_num(img_bw)
        
        warp_init = np.eye(2, 3, dtype=np.float32)

        (cc, warp) = cv2.findTransformECC( # CC might be interesting later on for confidence-based evaluation.
            ref_bw,
            img_bw,
            warp_init,
            cv2.MOTION_EUCLIDEAN,
            criteria
        )

        # Rescales the final result to normal size.
        warp[0,2] *= 1/scale
        warp[1,2] *= 1/scale

        # Warps the current image to the reference image at subpixel level.
        aligned = cv2.warpAffine(
            image,
            warp,
            (reference_image.shape[1], reference_image.shape[0]),
            flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP # INTER_LINEAR makes data points that dont have a direct mapping use linear interpolation based on the closest predictions.
        )

        aligned_images.append(aligned)
    return aligned_images     

