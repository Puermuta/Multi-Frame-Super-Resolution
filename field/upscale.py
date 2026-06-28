def upscale(good_reference, displacement, upscale_factor = 2):
    """
    Upscales all feature displacement fields with some factor.
    """
    
    for idx, _ in enumerate(good_reference):
        good_reference[idx] *= 2
        displacement[idx] *= 2