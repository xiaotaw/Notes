#!/usr/bin/env python
# encoding=utf8

# conda activate tf18
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

nyu_dir = "/media/xt/8T/DATASETS/NYU_Depth_Dataset_V2/nyu_depth_v2_raw"
scene = "dining_room_0034"
scene_dir = os.path.join(nyu_dir, scene)

# depth image filelist
dl = glob.glob(os.path.join(scene_dir, "d-*.pgm"))
dl = [os.path.basename(x) for x in dl]
dt = [float(x.split("-")[1]) for x in dl]  # time stamp
dt.sort()

# color image filelist
rl = glob.glob(os.path.join(scene_dir, "r-*.ppm"))
rl = [os.path.basename(x) for x in rl]
rt = [float(x.split("-")[1]) for x in rl]
rt.sort()

# accel dump filelist
al = glob.glob(os.path.join(scene_dir, "a-*.dump"))
al = [os.path.basename(x) for x in al]
at = [float(x.split("-")[1]) for x in al]
at.sort()

# sync depth and color images
sync = []
diff = []

TIME_ERROR_LIMIT = 0.05

for i, d in enumerate(dt):
    d_prev = dt[i-1] if i >= 1 else d-TIME_ERROR_LIMIT
    d_next = dt[i+1] if i+1 < len(dt) else d+TIME_ERROR_LIMIT

    min_diff = d
    min_index = -1
    for j, r in enumerate(rt):
        if r < d_prev:
            continue
        elif r <= d_next:
            diff_temp = abs(d - r)
            if diff_temp <= min_diff:
                min_diff = diff_temp
                min_index = j
            else:
                break
                #
        else:
            break

    if min_index != -1 and min_diff < TIME_ERROR_LIMIT:
        sync.append((d, rt[min_index], min_diff))
        diff.append(min_diff)
    else:
        sync.append((d, None, 0))
        diff.append(0)


diff = np.array(diff)

# plt.hist(diff)
plt.plot(diff)
plt.show()
