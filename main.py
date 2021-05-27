from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from IPython import embed
from jaxlie import SO3
from mpl_toolkits import mplot3d

import calib
from q import _from_angle_axis, _from_wxyz

show = False


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
print(f"poses.shape = {poses.shape}")

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
                                  maxCorners=200,
                                  qualityLevel=0.01,
                                  minDistance=10)
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
if show:
    plt.show()

plt.close('all')

# Do EKF filtering


# Manage the state (only robot pose at the start)
def _split(x):
    # First 13 elements are the robot pose
    p = x[:3]
    q = _from_wxyz(x[3:7])
    v = x[7:10]
    w = x[10:13]

    # Rest of elements are the 3D feature locations
    features = x[13:]
    features = jnp.reshape(features, (-1, 3))
    return p, q, v, w, features


def _join(p, q, v, w, features):
    # features.shape = (N, 3)
    # q: jaxlie.SO3

    return jnp.array([
        *p,  #
        *q.wxyz,
        *v,
        *w,
        *features.flatten()
    ])


# Add features to state
# features.shape = N x 3
features = features.T[:, :3]

# Same as before, we assume that z-axis goes into the page, y-axis points down, and x-axis points to the right.
# This satisfies the right-hand rule.

mu0 = _join(
    # 3D location (xyz)
    [0, 0, 0],
    # Rotation (rotation quaternion, we start off as identity = no rotation)
    SO3.identity(),
    # Velocity (xyz),
    [0, 0, 0.5],
    # Angular velocity (xyz, rad/s),
    1e-8 + jnp.array([0, 0, 0]),
    features)

# Close to 0 process noise for the landmarks (assume stationary)
Q = jax.scipy.linalg.block_diag(jnp.identity(13),
                                1e-8 * jnp.identity(3 * features.shape[0]))
R = 1e-3 * jnp.identity(2 * len(features))
sigma0 = 0.01 * jnp.identity(len(mu0))

mus = [mu0]
sigmas = [sigma0]


# State transition
@jax.jit
def _f(x, dt):
    """
    Constant velocity motion model. 
    """
    p, q, v, w, features = _split(x)
    _p = p + v * dt

    # Calculate axis-angle representation
    w_norm = jnp.linalg.norm(w)
    theta = dt * w_norm
    _w = jnp.where(w_norm > 1e-4, w / w_norm, w)
    r = _from_angle_axis(_w, theta)

    # Ensure rotation
    r = r.normalize()

    # NOTE: the new instataneous rotation comes at the end, not before.
    _q = q @ r
    return _join(_p, _q, v, w, features)


def _normalize(wxyz):
    return SO3(wxyz).normalize().wxyz


_j_normalize = jax.jit(jax.jacfwd(_normalize))


def _normalize_q(x):
    p, q, v, w, features = _split(x)
    # Compute normalized quaternion
    wxyz = q.wxyz
    _q = _normalize(wxyz)
    # Compute Jacobian
    jq = _j_normalize(wxyz)
    return _join(p, SO3(_q), v, w, features), jq


# Observe
img_h, img_w = (1241, 376)


def _g(x):
    """
    Observation model.

    Project each 3D feature to 2D plane based on estimated pose.
    """
    p, q, v, w, features = _split(x)

    # Transform from world to view/camera space
    R_ = q.inverse().as_matrix()

    def l(xyz):
        # p.shape = (3,)
        # return xyz
        # return xyz - p
        return R_ @ (xyz - p)

    embed()
    features = jax.vmap(l)(features)
    # embed()

    # Append homogenous coordinates
    features = jnp.hstack((features, jnp.ones((len(features), 1))))
    projected_features = P0 @ features.T
    projected_features = (projected_features / projected_features[-1]).T
    # Clip features to correct range
    # final = jnp.array([
    #     jnp.clip(projected_features[:, 0], a_min=0, a_max=img_w),
    #     jnp.clip(projected_features[:, 1], a_min=0, a_max=img_h)
    # ]).T
    final = projected_features[:, :2]
    return final.flatten()


def _range(x):
    print(jnp.min(x), jnp.max(x))


# Set up optical flow tracking
old_left = left
p0 = corners
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_A = jax.jit(jax.jacfwd(_f))
_C = jax.jit(jax.jacfwd(_g))

# n = state dimension
n = len(mu0)

_O = []
observable = []

try:
    # Iterate through all images (measurements)
    for i in range(1, 100):
        mu = mus[-1]
        sigma = sigmas[-1]
        dt = times[i] - times[i - 1]
        print(f"i = {i}, dt = {dt}")

        # (1) Predict
        _mu = _f(mu, dt)
        A = _A(mu, dt)
        _sigma = A @ sigma @ A.T + Q

        # (2) Update

        # C, _g(_mu) will be computed for all 3D feature locations.
        # But we might not actually observe all of them in the image due to camera movement
        # occlusion. The mean should not be touched, the covariance should increase (update step
        # should not decrease covariance for unobserved features, just leave them alone).

        # Load images
        left_path = left_img_paths[i]
        right_path = right_img_paths[i]

        left = _load_image(left_path)
        right = _load_image(right_path)

        # Track features
        p0 = p0.astype(onp.float32).reshape((-1, 1, 2))
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_left,  #
            left,
            p0,
            None,
            **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        print(f"{jnp.max(good_new) = }, {jnp.min(good_new) = }")
    
        if show:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(old_left)
            axs[0].scatter(good_old[:, 0], good_old[:, 1], c='r')
            axs[0].set_title("old t")

            axs[1].imshow(left)
            axs[1].scatter(good_new[..., 0], good_new[..., 1], c='r')
            axs[1].set_title(f"t = {times[i]}, i = {i}")
            plt.show()

        p1 = p1.flatten()
        _y = _g(_mu)
        # embed()
        print(f"{jnp.max(_y) = }, {jnp.min(_y) = }")
        C = _C(_mu)

        # Observability
        # O = []
        # for i in jnp.arange(n):
        #     O.append(C @ jnp.linalg.matrix_power(A, i))
        # # Resulting matrix has shape (m * n, n) where m is the measurement dimension
        # O = jnp.vstack(O)
        # # Observable if O is full rank
        # observable = jnp.linalg.matrix_rank(O) == n
        # # embed()
        # _O.append(observable)

        K = _sigma @ C.T @ jnp.linalg.inv(C @ _sigma @ C.T + R)

        st = st.flatten()
        missed = jnp.argwhere(st == 0).flatten()

        inno = p1 - _y
        inno = inno.reshape((-1, 2)).at[missed].set(0)
        inno = inno.flatten()
        print(f"{jnp.max(inno) = }, {jnp.min(inno) = }")

        z_sigma = _sigma

        # TODO: vectorize this
        for ix in missed:
            j = 13 + 3 * ix
            z_sigma = z_sigma.at[j:j + 3].set(0)
            z_sigma = z_sigma.at[:, j:j + 3].set(0)

        mu = _mu + K @ inno
        sigma = _sigma - K @ C @ z_sigma

        # Check observability
        observable.append(jnp.linalg.norm(_g(mu) - p1))

        # Post-update processing (MonoSLAM)
        # Normalize quaternion and set covariance accordingly
        embed()
        mu, jq = _normalize_q(mu)
        sigma = sigma.at[3:7, 3:7].set(jq)

        mus.append(mu)
        sigmas.append(sigma)

        # Update feature tracking
        old_left = left
        p0 = p1.reshape((-1, 1, 2))

    mus = jnp.array(mus)
except Exception as e:
    print(e)
    embed()

embed()
