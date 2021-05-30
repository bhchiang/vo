# Initial test for factor graph optimization
# Not incremental

from pathlib import Path

import cv2
import gtsam
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from gtsam import symbol
from gtsam.utils import plot
from IPython import embed
from mpl_toolkits import mplot3d

import calib

# Show plots
show = False
# Triangulate new features every `discover_freq` frames
discover_features = True
discover_freq = 10
from gtsam.utils import plot


def _load_text(p):
    with open(p, "r") as f:
        return [x.strip() for x in f.readlines()]


def _load_image(p):
    img = cv2.imread(str(data_path / p), cv2.IMREAD_GRAYSCALE)
    # img = onp.float32(img) / 255
    return img


poses_path = "/media/bryan/shared/kitti2/dataset/poses/00.txt"
poses = []
for l in _load_text(poses_path):
    poses.append(jnp.array([float(x) for x in l.split(" ")]).reshape((3, 4)))
poses = jnp.array(poses)
print(f"{poses.shape = }")
gt_positions = jnp.array([x[:, -1] for x in poses])

data_path = Path("/media/bryan/shared/kitti2/dataset/sequences/00")
calib_path = data_path / "calib.txt"
times_path = data_path / "times.txt"

left_img_paths = _load_text(data_path / "left_imgs.txt")
left_img_paths = [data_path / "image_0" / x for x in left_img_paths]

right_img_paths = _load_text(data_path / "right_imgs.txt")
right_img_paths = [data_path / "image_1" / x for x in right_img_paths]
times = _load_text(times_path)  # In seconds
times = [float(x) for x in times]

num_imgs = len(left_img_paths)
assert len(left_img_paths) == len(right_img_paths)
assert len(left_img_paths) == len(times)

# === Parameters ===

# Disparity
window_size = 5
min_disp = 0
num_disp = 64
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=16,
                               P1=8 * 3 * window_size**2,
                               P2=8 * 3 * window_size**2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32)

# Images
img_w, img_h = (1241, 376)

# Calibration
P0, P1 = calib._load_calib(calib_path)

# Calculate depth
cx = P0[0, 2]
cy = P0[1, 2]
fx_px = P0[0, 0]
fy_px = P0[1, 1]  # fx = fy for KITTI
baseline_px = P1[0, 3]
baseline_m = 0.54
K = gtsam.Cal3_S2Stereo(fx_px, fy_px, 0, cx, cy, baseline_m)

# Optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
#  - For simplicity this example is in the camera's coordinate frame
#  - X: right, Y: down, Z: forward
#  - Pose x1 is at the origin, Pose 2 is 1 meter forward (along Z-axis)
#  - x1 is fixed with a constraint, x2 is initialized with noisy values
#  - No noise on measurements


def X(idx):
    return symbol(ord('x'), idx)


def L(idx):
    return symbol(ord('l'), idx)


# Optimization
start_idx = 0
num_frames = 20
new_feature_threshold = 50


def _find_new_points(left,
                     points,
                     status,
                     disparity,
                     show=False,
                     title="new points"):
    """
    left: image to find new points in.
    points: existing points in the points.
    disparity: disparity of current stereo pair.
    """
    corners = cv2.goodFeaturesToTrack(left,
                                      maxCorners=500,
                                      qualityLevel=0.3,
                                      minDistance=50)
    corners = onp.squeeze(corners).astype(int)
    disparity_corners = disparity[corners[:, 1], corners[:, 0]]
    valid = disparity_corners > 10
    disparity_corners = disparity_corners[valid]
    corners = corners[valid]

    good_current_points = points[status == 1]
    bad_current_points = points[status == 0]

    # We want to find new points. The distance to existing **good** points > 50.
    new_points = []
    for c in corners:
        dist = onp.linalg.norm(good_current_points - c, axis=1)
        closest = onp.min(dist)
        if closest > new_feature_threshold:
            new_points.append(c)
    new_points = onp.array(new_points)

    if show:
        plt.figure()
        plt.imshow(left)
        plt.scatter(good_current_points[:, 0],
                    good_current_points[:, 1],
                    c='xkcd:pale lilac',
                    edgecolors='b',
                    label="good current")
        plt.scatter(bad_current_points[:, 0],
                    bad_current_points[:, 1],
                    c='w',
                    label="bad current")
        plt.scatter(new_points[:, 0],
                    new_points[:, 1],
                    c='xkcd:bright red',
                    label="new points")
        plt.title(title)
        plt.legend()
        plt.show()

    return new_points


cur_pose = gtsam.Pose3()

for i in range(start_idx, start_idx + num_frames):
    ## Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()

    ## add a constraint on the starting pose
    first_pose = gtsam.Pose3()
    graph.add(gtsam.NonlinearEqualityPose3(X(i), first_pose))

    ## Create realistic calibration and measurement noise model
    # format: fx fy skew cx cy baseline
    stereo_model = gtsam.noiseModel_Diagonal.Sigmas(onp.array([1.0, 1.0, 1.0]))

    ## Create initial estimate for camera poses and landmarks
    initialEstimate = gtsam.Values()
    initialEstimate.insert(X(i), first_pose)

    # Add estimate for next pose
    rot = first_pose.rotation()
    pos = first_pose.translation()
    next_pose = gtsam.Pose3(rot, gtsam.Point3(pos.x(), pos.y(), pos.z() + 1))

    # Move forward
    initialEstimate.insert(X(i + 1), next_pose)

    # Load in images
    left_path = left_img_paths[i]
    right_path = right_img_paths[i]

    next_left_path = left_img_paths[i + 1]
    next_right_path = right_img_paths[i + 1]

    left = _load_image(left_path)
    right = _load_image(right_path)

    next_left = _load_image(next_left_path)
    next_right = _load_image(next_right_path)

    points = cv2.goodFeaturesToTrack(left,
                                     maxCorners=500,
                                     qualityLevel=0.3,
                                     minDistance=50)
    points = onp.int32(points).reshape((-1, 2))
    num_points = len(points)

    # Track optical flow into next frame
    p0 = onp.float32(points).reshape((-1, 1, 2))
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_left,  #
        left,
        p0,
        None,
        **lk_params)
    st = st.flatten()
    p1 = p1.reshape((-1, 2))
    good_points = points[st == 1]
    next_good_points = p1[st == 1]

    num_good_points = len(good_points)

    disparity = stereo.compute(left, right) / 16.0
    next_disparity = stereo.compute(next_left, next_right) / 16.0

    for j in range(num_good_points):
        print(f"frame = {i}, point = {j}")
        # Current frame
        uL, v = good_points[j]
        d = disparity[v, uL]
        uR = uL - d

        z = (fx_px * baseline_m) / d
        x = (uL - cx) * (z / fx_px)
        y = (uR - cy) * (z / fy_px)

        # Next frame
        next_uL, next_v = next_good_points[j]
        next_d = next_disparity[next_v, next_uL]
        next_uR = next_uL - next_d

        next_z = (fx_px * baseline_m) / next_d
        next_x = (next_uL - cx) * (next_z / fx_px)
        next_y = (next_uR - cy) * (next_z / fy_px)

        if show:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(left)
            axs[0].set_title("corners on left")
            axs[0].scatter(uL, v, c='r')

            axs[1].imshow(right)
            axs[1].set_title("corners on right")
            axs[1].scatter(uL, v, c='c', label="original")
            axs[1].scatter(uR, v, c='r', label="adjusted")
            plt.legend()
            plt.show()

        graph.add(
            gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(uL, uR, v),
                                        stereo_model, X(i), L(j), K))
        initialEstimate.insert(L(j), gtsam.Point3(x, y, z))

        graph.add(
            gtsam.GenericStereoFactor3D(
                gtsam.StereoPoint2(next_uL, next_uR, v), stereo_model, X(i),
                L(j), K))
        initialEstimate.insert(L(j), )

    embed()

    # Track features
    p0 = onp.float32(p0).reshape((-1, 1, 2))
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_left,  #
        left,
        p0,
        None,
        **lk_params)

    st = st.flatten()
    p1 = onp.int32(p1.reshape((-1, 2)))
    disparity = stereo.compute(left, right) / 16.0

    print(f"Number good features = {len(p1[st==1])}")

    for j in range(num_features):
        # Only add factors for good features
        if st[j] == 0:
            continue

        # In bounds
        uL, v = p1[j]
        uL = onp.clip(uL, 0, img_w - 1)
        v = onp.clip(v, 0, img_h - 1)

        # Disparity is valid
        d = disparity[v, uL]
        if d < 10:
            continue

        uR = uL - d
        uR = onp.clip(uR, 0, img_w - 1)
        graph.add(
            gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(uL, uR, v),
                                        stereo_model, _xsym(i), _lsym(j), K))

    # Estimate the current camera pose
    initialEstimate.insert(
        _xsym(i), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(-0.1, -0.1, 1.5 * i)))

    # Add factor connecting current pose to previous pose
    # graph.add(
    #     gtsam.BetweenFactorPose3(_xsym(i - 1), _xsym(i), odometry,
    #                              odometry_noise))

    # Clean up optical flow
    p0 = p1
    old_left = left

    # print(graph)
    print("Optimizing...")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()

print(result)

embed()
plot.plot_3d_points(1, result)
plot.plot_trajectory(1, result)
plot.set_axes_equal(1)
plot.show()