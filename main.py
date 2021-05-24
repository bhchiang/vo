from pathlib import Path
import calib

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from IPython import embed
from mpl_toolkits import mplot3d

data_path = Path("data/")


def _load_image(p):
    img = cv2.imread(str(data_path / p), cv2.IMREAD_GRAYSCALE)
    # img = onp.float32(img) / 255
    return img


# Create the state (only robot pose at the start)
mu0 = jnp.array([
    # 3D location
    0,
    0,
    0,
    # Rotation (quaternion)
    0,
    0,
    0,
    0,
])
sigma0 = 0.01 * jnp.array(7)

mus = [mu0]
sigmas = [sigma0]

left_path = "/media/bryan/shared/kitti2/dataset/sequences/00/image_0/000000.png"
right_path = "/media/bryan/shared/kitti2/dataset/sequences/00/image_1/000000.png"

left = _load_image(left_path)
right = _load_image(right_path)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(left)
axs[0].set_title("left")
axs[1].imshow(right)
axs[1].set_title("right")
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
plt.show()

embed()

# Find point features
corners = cv2.goodFeaturesToTrack(left,
                                  maxCorners=200,
                                  qualityLevel=0.01,
                                  minDistance=10)
corners = onp.squeeze(corners).astype(int)

disparity_corners = disparity[corners[:, 1], corners[:, 0]]

embed()
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
plt.show()

# Back project to 3D for the left camera
calib_path = "/media/bryan/shared/kitti2/dataset/sequences/00/calib.txt"
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

plt.show()

# Observation model (project 3D points down to 2D)
# 4 x N
features = onp.vstack((bp_x, bp_y, bp_z, onp.ones(len(bp_x))))
projected_features = P0 @ features  # 3 x N
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
plt.show()

embed()

# Start the filtering
for i, (left_path, right_path) in enumerate(img_paths[start_idx + 1:]):
    # Predict
    mu = mus[-1]
    sigma = sigmas[-1]

    #
    break

embed()


def _g(x):
    """
    Observation model.

    Project each 3D feature to 2D plane based on estimated pose.
    """


def _h(x, left_img, right_img):
    """
    Inverse observation model.

    Given current pose estimate, triangulate 3D feature location of all
    """
    pass
