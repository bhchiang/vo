# vo

Visual Odometry for Pose Tracking

- The filtering based approach is located in `direct.py`.
- The fixed-lag smoothing approach is located in `fixed_lag.py`.n
- `calib.py` contains the calibration data for KITTI.

You can run both approaches with:

```sh
python direct.py
python fixed_lag.py
```

You can install all dependencies from the `requirements.txt`.

You can download the KITTI odometry dataset from:

http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Other approaches I tried.

- `fg.py` optimizes a factor graph over the entire trajectory, with no new features.
- `fg_odom.py` smoothers the result of the filtering-based approach mentioned in the paper.
- `q.py` contains utilies for dealing with quaternions / SO3 in JAX for autodifferentiation.
- `ekf.py` contains code for my initial EKF-SLAM version that I wrote in JAX.
- `sift_fixed_lag.py` is my attempt at tracking SIFT features across many temporal frames instead of using optical flow.
- `isam.py` is my attempt at using Incremental Smoothing and Mapping (iSAM) for visual odometry. The results are much better than what's in my paper, but if gave me indeterminant system errors under in less than 100 frames, which was weird since I had a lot of measurements.
