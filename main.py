# TODO: make the folder structure become a package/module based system, so that one
# can import as just -alignment_ecc import transform_ecc- instead of twice x.x.
from alignment_ecc.alignment_ecc import tranform_ecc
from pipeline_io.input import import_images
from feature_detection.feature_detection import get_features
from motion_lk.motion_lk import get_optical_flow
from fusion.fusion import fuse
from field.upscale import upscale
from field.interpolate import interpolate_displacement
from pipeline_io.output import output
import numpy as np
import sys

if __name__ == "__main__":
    """
    Testing the project pipeline while developing.
    """

    # Step 1: Import images, verify count
    print("---- Step 1 ----: Import images")
    path = "./input_images"
    images = []
    try:
        images = import_images(
            path = path,
            scale = 0.4,  # Doing this to reduce RAM usage on my 8GB macbook air :)
            #debug = True # Should output 8
        )
        print("Images imported and preprocessed successfully.")
    except Exception as e:
        print(f"Import of images failed.\nError: {e}")
        sys.exit(1)
    print("f", np.mean(images[0]))
    print(images[0].shape)
    # Step 2: Align images with global transformation
    print("---- Step 2 ----: Align images")
    try:
        aligned_images = tranform_ecc(images)
        print("Images aligned successfully.")
    except Exception as e:
        print(f"Could not align images. \nError: {e}")
        sys.exit(1)
    print("e", np.mean(aligned_images[0]))
    # Step 3: Pick interesting features to track for optical flow
    print("---- Step 3 ----: Find features to track.")
    features = []
    try:
        features = get_features(images) # Gets features from the middle image.
        print("Features picked succesfully.")
    except Exception as e:
        print(f"Could not get fetures from the image stack.\nError: {e}")
        sys.exit(1)
    print("d", np.mean(aligned_images[0]))

    # Step 4: Optical flow analysis
    print("---- Step 4 ----: Doing optical flow analysis.")
    try:
        good_reference, displacement = get_optical_flow(aligned_images, features, debug = False)
        print(f"Optical flow succeeded, vector fields constructed.")
    except Exception as e:
        print(f"Could not do optical flow analysis.\nError: {e}")
        sys.exit(1)
    print("a", np.mean(aligned_images[0]))
    # Step 5: Upscale vector field to right scale
    print("---- Step 5 ----: Upscaling optical flow field.")
    try:
        upscale(good_reference, displacement, 2)
        print("Field upscaled successfully.")
    except Exception as e:
        print(f"Could not upscale optical flow field. Error: {e}")
        sys.exit(1)
    print("b", np.mean(aligned_images[0]))
    # Step 6: Create dense optical flow field
    print("---- Step 6 ----: Building dense optical flow field.")
    width = images[0].shape[1] * 2
    height = images[0].shape[0] * 2
    try:
        dense_flows = interpolate_displacement(good_reference, displacement, width, height)
        print("Dense optical flow field created.")
    except Exception as e:
        print(f"Could not create dense optical field. Error: {e}")
        sys.exit(1)
    print("c", np.mean(aligned_images[0]))

    # Step 7: Fuse
    print(aligned_images[0].shape)
    print("---- Step 7 ----: Constructing new image.")
    try:
        image = fuse(aligned_images, dense_flows, width, height)
        print("Fusion succeeded.")
    except Exception as e:
        print(f"Could not do fusion. Error: {e}")
        sys.exit(1)

    # Step 8:
    print("---- Step 8 ----: Output result.")
    try:
        output(image)
    except Exception as e:
        pass        

    

