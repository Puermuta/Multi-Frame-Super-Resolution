import rawpy
import numpy as np
import cv2
from pathlib import Path

def _normalize(img):
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img

def import_images(path: str, scale = 1.0, debug = False) -> list:
    folder = Path(path)
    
    images = []
    images_count = 0
    for file_path in folder.glob("*.ARW"): # Only testing with ARW as of now.
        images_count += 1
        with rawpy.imread(str(file_path)) as raw:
            rgb = raw.postprocess(
                output_bps = 16,                 
                no_auto_bright = True,
                gamma = (1, 1),                 # critical: keeps linear response
                use_camera_wb = True,           # closest to real scene colors
                auto_bright_thr = 0.0,          # disables exposure scaling
                highlight_mode = 0,             # no highlight reconstruction tricks
                demosaic_algorithm = rawpy.DemosaicAlgorithm.AHD #type: ignore
            )
    
        image = rgb.astype(np.float32) / 65535.0 * 255
        image = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image
        image = _normalize(image)
        
        images.append(image)

    if debug:
        print(f"Images found: {images_count}")
    return images