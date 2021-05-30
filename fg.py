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

left_path = left_img_paths[0]
right_path = right_img_paths[0]

left = _load_image(left_path)
right = _load_image(right_path)

# embed()

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(left)
axs[0].set_title("left")
axs[1].imshow(right)
axs[1].set_title("right")
if show:
    plt.show()

# Extract disparity
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

# Convert to pixel-level disparity
disparity = stereo.compute(left, right) / 16.0

plt.figure()
plt.title("initial disparity")
plt.imshow(disparity)

if show:
    plt.show()

# Verify results
x = 800
y = 300
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(left)
axs[0].set_title("left")
axs[0].scatter(x, y, c='r')

axs[1].imshow(right)
axs[1].set_title("right")
axs[1].scatter(x, y, c='c', label="original")
axs[1].scatter(x - disparity[y, x], y, c='r', label="adjusted")
plt.legend()
if show:
    plt.show()

# embed()

# Find point features
corners = cv2.goodFeaturesToTrack(left,
                                  maxCorners=500,
                                  qualityLevel=0.3,
                                  minDistance=50)
corners = onp.squeeze(corners).astype(int)

disparity_corners = disparity[corners[:, 1], corners[:, 0]]

# embed()
valid = disparity_corners > 10
disparity_corners = disparity_corners[valid]
corners = corners[valid]

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(left)
axs[0].set_title("corners on left")
axs[0].scatter(corners[:, 0], corners[:, 1], c='r')

axs[1].imshow(right)
axs[1].set_title("corners on right")
axs[1].scatter(corners[:, 0], corners[:, 1], c='c', label="original")
axs[1].scatter(corners[:, 0] - disparity_corners,
               corners[:, 1],
               c='r',
               label="adjusted")
plt.legend()
if show:
    plt.show()

# Back project to 3D for the left camera
P0, P1 = calib._load_calib(calib_path)

# Calculate depth
cx = P0[0, 2]
cy = P0[1, 2]
fx_px = P0[0, 0]
fy_px = P0[1, 1]  # fx = fy for KITTI
baseline_px = P1[0, 3]
baseline_m = 0.54

z = (fx_px * baseline_m) / disparity_corners

# Camera coordinate system is as follows
# z pointing into the screen
# ------> x
# |
# |
# v
# y

# Calculate x, y coordinates
bp_x = (corners[:, 0] - cx) * (z / fx_px)
bp_y = (corners[:, 1] - cy) * (z / fy_px)
bp_z = z

# Plot backprojection results (2D)
fig, axs = plt.subplots(nrows=2, ncols=1)

im = axs[0].scatter(corners[:, 0], corners[:, 1], c=bp_z)
axs[0].invert_yaxis()
fig.colorbar(im, ax=axs[0])
axs[0].set_title("x, y")
axs[0].set_aspect('equal', adjustable='box')

axs[1].scatter(bp_x, bp_y, c=bp_z)
axs[1].invert_yaxis()
axs[1].set_title("after back projection")
axs[1].set_aspect('equal', adjustable='box')

if show:
    plt.show()

# Plot backprojected result in 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')

ax.scatter3D(corners[:, 0], corners[:, 1], bp_z, c=bp_z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.invert_xaxis()
ax.invert_zaxis()
ax.set_title("image x, y with depth")

# Set aspect ratio (space out the z-axis to see the depth more clearly)
ax.set_box_aspect(
    (onp.ptp(corners[:, 0]), onp.ptp(corners[:, 1]), 5 * onp.ptp(bp_z)))

if show:
    plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(bp_x, bp_y, bp_z, c=bp_z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.invert_xaxis()
ax.invert_zaxis()
ax.set_title("backprojected x, y with depth")

# Set aspect ratio (space out the z-axis to see the depth more clearly)
ax.set_box_aspect(
    (onp.ptp(corners[:, 0]), onp.ptp(corners[:, 1]), 5 * onp.ptp(bp_z)))

if show:
    plt.show()

# Observation model (project 3D points down to 2D)
# 4 x N
features = onp.vstack((bp_x, bp_y, bp_z)).T
num_features = len(features)
projected_features = P0 @ onp.append(
    features, onp.ones((num_features, 1)), axis=1).T  # 3 x N
# Normalize by homogenous coordinate
projected_features = (projected_features / projected_features[-1]).T
projected_features = round(projected_features[:, :2])
onp.testing.assert_allclose(corners, projected_features)

plt.figure()
plt.imshow(left)
plt.scatter(projected_features[:, 0],
            projected_features[:, 1],
            c='r',
            label="reprojected features")
plt.legend()
if show:
    plt.show()

plt.close('all')

#  - For simplicity this example is in the camera's coordinate frame
#  - X: right, Y: down, Z: forward
#  - Pose x1 is at the origin, Pose 2 is 1 meter forward (along Z-axis)
#  - x1 is fixed with a constraint, x2 is initialized with noisy values
#  - No noise on measurements


def _xsym(idx):
    return symbol(ord('x'), idx)


def _lsym(idx):
    return symbol(ord('l'), idx)


## Create graph container and add factors to it
graph = gtsam.NonlinearFactorGraph()

## add a constraint on the starting pose
first_pose = gtsam.Pose3()
graph.add(gtsam.NonlinearEqualityPose3(_xsym(0), first_pose))

## Create realistic calibration and measurement noise model
# format: fx fy skew cx cy baseline
K = gtsam.Cal3_S2Stereo(fx_px, fy_px, 0, cx, cy, baseline_m)
stereo_model = gtsam.noiseModel_Diagonal.Sigmas(onp.array([1.0, 1.0, 1.0]))

## Create initial estimate for camera poses and landmarks
initialEstimate = gtsam.Values()
initialEstimate.insert(_xsym(0), first_pose)

for i in range(num_features):
    uL, v = corners[i]
    x, y, z = features[i]
    d = disparity_corners[i]
    uR = uL - d
    graph.add(
        gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(uL, uR, v),
                                    stereo_model, _xsym(0), _lsym(i), K))
    initialEstimate.insert(_lsym(i), gtsam.Point3(x, y, z))
print("Created factors for initial pose")
embed()

# Start the loop - add more factors for future poses
start_idx = 1
num_frames = 20

# Set up optical flow
old_left = left
p0 = corners
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

odometry = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2.0, 0.0, 0.0))
odometry_noise = gtsam.noiseModel_Diagonal.Sigmas(onp.array([0.2, 0.2, 1]))

img_w, img_h = (1241, 376)

for i in range(start_idx, start_idx + num_frames):
    left_path = left_img_paths[i]
    right_path = right_img_paths[i]

    left = _load_image(left_path)
    right = _load_image(right_path)

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

positions = []
for i in range(0, start_idx + num_frames):
    pose = result.atPose3(_xsym(i))
    pos = pose.translation()
    positions.append([pos.x(), pos.y(), pos.z()])

positions = onp.array(positions)

plt.figure()
plt.plot(positions[:, 0], positions[:, 2], label="estimated")
plt.plot(gt_positions[:start_idx + num_frames, 0],
         gt_positions[:start_idx + num_frames, 2],
         label="ground truth")
plt.xlabel("x")
plt.ylabel("z")
plt.title("batch optimization")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()
