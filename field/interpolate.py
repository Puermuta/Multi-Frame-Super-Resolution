"""
Take sparse feature tracking data from motion_lk.
Use scipy.griddata to interpolate a new dense field
to fill the holes.
Return a set of N frames with now dense motion fields.
"""
import numpy as np
from scipy.interpolate import griddata

def interpolate_displacement(good_reference, displacement, height, width):
    output = []
    print(len(good_reference))
    for idx, _ in enumerate(good_reference):
        if (np.mean(displacement[idx]) == 0.0):
            output.append(np.zeros((width, height)))
            continue

        #points = good_ref.reshape(-1, 2) *
        current_points = np.array(good_reference[idx]).reshape(-1, 2)
        print(current_points.shape)
        current_displacements = displacement[idx]

        grid_x, grid_y = np.mgrid[0:width, 0:height]

        current_frame = griddata(current_points, current_displacements, (grid_x, grid_y), method='linear')
        current_frame_nan = griddata(current_points, current_displacements, (grid_x, grid_y), method='nearest')

        current_frame[np.isnan(current_frame)] = current_frame_nan[np.isnan(current_frame)]

        output.append(current_frame)

    return output