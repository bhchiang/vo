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
from q import _from_axis_angle, _from_wxyz, _rotate, _from_vector

# Show plots
show = False
# Triangulate new features every `discover_freq` frames
discover_features = True
discover_freq = 10


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
# Find point features
points = cv2.goodFeaturesToTrack(left,
                                 maxCorners=10,
                                 qualityLevel=0.3,
                                 minDistance=50)
points = onp.squeeze(points).astype(int)
status = onp.ones(len(points))

# Set initial features to some where in front of the camera
features = jnp.array([1e-5, 1e-5, 15] * len(points)).reshape((-1, 3))
mu0 = _join(
    # 3D location (xyz)
    [0, 0, 0],
    # Rotation (rotation quaternion, we start off as identity = no rotation)
    SO3.identity(),
    # Velocity (xyz),
    [0, 0, 2],
    # Angular velocity (xyz, rad/s),
    1e-8 + jnp.array([0, 0, 0]),
    features)

sigma0 = jax.scipy.linalg.block_diag(1 * jnp.identity(13),
                                     1e-8 * jnp.identity(3 * len(features)))
# Process noise
# Close to 0 process noise for the landmarks (assume stationary)
Q = jax.scipy.linalg.block_diag(0.1 * jnp.identity(13),
                                1e-8 * jnp.identity(3 * len(features)))
# Measurement noise
R = 1e-5 * jnp.identity(2 * len(features))

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
    _w = jnp.where(w_norm > 1e-8, w / w_norm, w)
    r = _from_axis_angle(_w, theta)
    _q = q @ r.normalize()
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


P0, P1 = calib._load_calib(calib_path)


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

    features = jax.vmap(l)(features)

    # Append homogenous coordinates
    features = jnp.hstack((features, jnp.ones((len(features), 1))))
    projected_features = P0 @ features.T
    projected_features = (projected_features / projected_features[-1]).T
    final = projected_features[:, :2]
    return final.flatten()


# Set up optical flow tracking
old_left = left
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_A = jax.jit(jax.jacfwd(_f))
_C = jax.jit(jax.jacfwd(_g))

# Iterate through all images (measurements)
embed()
for i in range(1, 200):
    mu = mus[-1]
    sigma = sigmas[-1]
    dt = times[i] - times[i - 1]
    print(f"i = {i}, dt = {dt}")

    # (1) Predict
    _mu = _f(mu, dt)
    A = _A(mu, dt)
    _sigma = A @ sigma @ A.T + Q

    # (2) Update
    # Load images
    left_path = left_img_paths[i]
    left = _load_image(left_path)

    # Track features
    inds_to_track = onp.argwhere(status == 1).flatten()
    points_to_track = points[inds_to_track]
    points_to_track = onp.float32(points_to_track).reshape((-1, 1, 2))
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_left,  #
        left,
        points_to_track,
        None,
        **lk_params)

    st = st.flatten()
    p1 = onp.int32(p1.reshape((-1, 2)))

    # These are global - ranging from (0, num_features)
    tracked_inds = inds_to_track[st == 1]
    untracked_inds = inds_to_track[st == 0]

    print(f"Number good features for frame = {len(tracked_inds)}")
    print(f"Number of missed features for frame = {len(untracked_inds)}")

    # Update status of points that we were unable to track
    status[untracked_inds] = 0
    tracked_points = p1[st == 1]
    points[tracked_inds] = tracked_points
    old_left = left  # Update old frame

    global_missed_inds = jnp.argwhere(status == 0)
    print(f"Total missed = {len(global_missed_inds)}")
    print(global_missed_inds)

    _y = _g(_mu)
    C = _C(_mu)

    embed()
    num_tracked_inds = len(tracked_inds)
    num_points = len(points)

    # Measuremnt length (after flattening)
    M = 2 * num_tracked_inds
    # Extract out relevant measurements
    C = C.reshape((num_points, -1))[tracked_inds].reshape((M, -1))

    # Only update the measurements that we care about
    R_ = R.reshape((num_points, -1))[tracked_inds].reshape((M, M))
    K = _sigma @ C.T @ jnp.linalg.inv(C @ _sigma @ C.T + R_)

    inno = p1 - _y

    # TODO: vectorize this
    for ix in missed:
        j = 13 + 3 * ix
        z_sigma = z_sigma.at[j:j + 3].set(0)
        z_sigma = z_sigma.at[:, j:j + 3].set(0)

    mu = _mu + K @ inno
    sigma = _sigma - K @ C @ z_sigma

    mu, jq = _normalize_q(mu)
    sigma = sigma.at[3:7, 3:7].set(jq)

    # Update feature tracking
    old_left = left
    p0 = p1.reshape((-1, 1, 2))

    mus.append(mu)
    sigmas.append(sigma)

mus = jnp.array(mus)
embed()
