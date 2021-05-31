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

first_idx = 0
num_frames = 200
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
all_poses = gtsam.Values()

all_poses.insert(X(first_idx), cur_pose)

for i in range(first_idx, first_idx + num_frames):
    print(f"optimizing transformation from {i} to {i+1}")
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

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(next_left, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = onp.float32([kp1[m.queryIdx].pt
                               for m in good]).reshape(-1, 2)
        dst_pts = onp.float32([kp2[m.trainIdx].pt
                               for m in good]).reshape(-1, 2)
        # embed()
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # h, w = left.shape
        # pts = onp.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
        #                    [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # next_left = cv2.polylines(next_left, [onp.int32(dst)], True, 255, 3,
        #                           cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None

    if show:
        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2)
        img3 = cv2.drawMatches(left, kp1, next_left, kp2, good, None,
                               **draw_params)
        plt.imshow(img3, 'gray')
        plt.show()

    disparity = stereo.compute(left, right) / 16.0
    next_disparity = stereo.compute(next_left, next_right) / 16.0

    mask = mask.flatten()

    good_src_pts = src_pts[mask == 1].reshape((-1, 2)).astype(onp.int32)
    good_dst_pts = dst_pts[mask == 1].reshape((-1, 2)).astype(onp.int32)

    for j in range(len(good_src_pts))[:300]:
        # print(f"frame = {i}, point = {j}")
        # Current frame
        uL, v = good_src_pts[j]
        d = disparity[v, uL]
        uR = uL - d

        z = (fx_px * baseline_m) / d
        x = (uL - cx) * (z / fx_px)
        y = (uR - cy) * (z / fy_px)

        # Next frame
        next_uL, next_v = good_dst_pts[j]
        next_d = next_disparity[next_v, next_uL]
        next_uR = next_uL - next_d

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
                gtsam.StereoPoint2(next_uL, next_uR, v), stereo_model,
                X(i + 1), L(j), K))

    # print(graph)
    print("Optimizing...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()
    # embed()

    # Update current_pose
    cur_pose = cur_pose.compose(result.atPose3(X(i + 1)))
    all_poses.insert(X(i + 1), cur_pose)

embed()
plot.plot_3d_points(1, result)
plot.plot_trajectory(1, result)
plot.set_axes_equal(1)
plot.show()

positions = []
for i in range(first_idx, first_idx + num_frames):
    pose = all_poses.atPose3(X(i))
    pos = pose.translation()
    positions.append([pos.x(), pos.y(), pos.z()])

positions = onp.array(positions)

plt.figure()
plt.plot(positions[:, 0], positions[:, 2], label="estimated")
plt.plot(gt_positions[first_idx:first_idx + num_frames, 0],
         gt_positions[first_idx:first_idx + num_frames, 2],
         label="ground truth")
plt.xlabel("x")
plt.ylabel("z")
plt.title("batch optimization")
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

with open("direct_poses.txt", "w") as f:
    for i in range(first_idx, first_idx + num_frames):
        m = all_poses.atPose3(X(i)).matrix()
        f.write(" ".join(map(str, m.flatten())))
        f.write("\n")
