# TODO: make the folder structure become a package/module based system, so that one
# can import as just -alignment_ecc import transform_ecc- instead of twice x.x.
from alignment_ecc.alignment_ecc import tranform_ecc
from pipeline_io.input import import_images
from feature_detection.feature_detection import get_features
from motion_lk.motion_lk import get_optical_flow
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

    # Step 2: Align images with global transformation
    print("---- Step 2 ----: Align images")
    try:
        aligned_images = tranform_ecc(images)
        print("Images aligned successfully.")
    except Exception as e:
        print(f"Could not align images. \nError: {e}")
        sys.exit(1)
    
    # Step 3: Pick interesting features to track for optical flow
    print("---- Step 3 ----: Find features to track.")
    features = []
    try:
        features = get_features(images) # Gets features from the middle image.
        print("Features picked succesfully.")
        print(features)
    except Exception as e:
        print(f"Could not get fetures from the image stack.\nError: {e}")
        sys.exit(1)

    # Step 4: Optical flow analysis
    print("---- Step 4 ----: Doing optical flow analysis.")
    try:
        vector_fields = get_optical_flow(aligned_images, features, debug = True)
        print(f"Optical flow succeeded, vector fields constructed.")
    except Exception as e:
        print(f"Could not do optical flow analysis.\nError: {e}")
        sys.exit(1)
    
    # Step 5: Interpolate a new high-res image

    


