# Multi-Frame Super-Resolution

Takes N RAW images and upscales to a higher resolution JPEG. The image under shows very briefly how it is intended to work:
![Yes, it is generated with AI](images/pipeline.png)
Note: The "High-res" accumulation will be a continuous vector field for each pixel´s displacement relative to the reference frame. By doing this, the goal is to achieve sub-pixel accuracy, which in turn gives the ability to upscale the image.

### Credit where credit is due
This project is based on work done by Google researches in [this paper](https://sites.google.com/view/handheld-super-res/) and will be a simplified version of it. It will also be inspired by [this implementation](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution), done by some more skilled developers than myself!

## Progress
- Import and preprocess ✅
- Per-frame global alignement ✅
- Optical flow ✅
- Continuous vector field constructor ✅
- Fusion and export of JPEG ✅

## Dependencies

- numpy
- rawpy
- opencv-python
- matplotlib
- scipy

## TODO
- PRIORITIZED: Fix artifacts when generating photos.
- PRIORITIZED: Use vectorization with numpy to optimize the calculation speed.
- Figure out how to do LK with 16-bit datasets. Now limited to 8-bit (see motion_lk.py). Might implement custom version in C++ if needed. Need to investigate the need for this though, might be precise enough with 8-bit.
- Create __init__.py files for each module for a cleaner import experience.
  
_(No particular order)_
