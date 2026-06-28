from scipy.interpolate import griddata
import numpy as np

def fuse(images, dense_flows, width, height, scaling_factor = 2):

    accumulated = np.zeros((height, width, 3))
    weights = np.zeros((height, width))

    print(accumulated.shape)
    print(weights.shape)

    for image, flow in zip(images, dense_flows):
        print(np.mean(image))
        if np.mean(flow) == 0.0:
            continue # We dont care about the reference frame. It does not contribute.
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                color = image[y][x]

                adj_x = int(x * scaling_factor // 1)
                adj_y = int(y * scaling_factor // 1)
                if adj_x < 0: adj_x = 0
                if adj_y < 0: adj_y = 0
                if adj_x > width - 1: adj_x = width - 1
                if adj_y > height - 1: adj_y = height - 1
                #print("a")
                #print(flow[adj_x][adj_y][0])
                dx = int(x * scaling_factor + flow[adj_y][adj_x][0][0])
                dy = int(y * scaling_factor + flow[adj_y][adj_x][0][1])
                #print("b")
                if dx < 0: dx = 0
                if dy < 0: dy = 0
                if dx > width - 1: 
                    dx = width - 1
                if dy > height - 1: 
                    dy = height - 1

                try:
                    accumulated[dy][dx] += color
                    weights[dy][dx] += 1
                except:
                    print(dy, dx, height, width)
                #print("c")
                
    print("c")
    print(accumulated.shape)
    weights_expanded = weights[:, :, np.newaxis]  # (H, W, 1) broadcasts with (H, W, 3)
    output = np.where(weights_expanded > 0, accumulated / weights_expanded, np.nan)
    print("a")
    # Find positions that have values
    gap_mask = weights == 0  # 2D (H, W)

    if np.any(gap_mask):
        valid_y, valid_x = np.where(~gap_mask)
        nan_y, nan_x = np.where(gap_mask)

        for c in range(output.shape[2]):
            valid_values = output[valid_y, valid_x, c]
            filled = griddata(
                np.column_stack([valid_x, valid_y]),
                valid_values,
                np.column_stack([nan_x, nan_y]),
                method='nearest'
            )
            output[nan_y, nan_x, c] = filled

    return output


