from dataclasses import dataclass
import time
from typing import List, Tuple

import numpy as np


@dataclass
class LidarPoint:
    x: float
    y: float
    z: float
    intensity: float


@dataclass
class LidarData:
    """
    holds lidar data information, with functionality to convert to bird eye view
    """

    points: List[LidarPoint]
    gps: Tuple[float, float]
    compass: float

    def __post_init__(self):
        """
        Convert to bird eye view
        """
        self._bev = None

    @property
    def bev(self):
        """The bev property."""

        if self._bev is None:
            self._bev = _to_bev(self.points, self.gps, self.compass)

        return self._bev

    # def _to_bev(self) -> np.ndarray:
    #     # TODO: find ways to optimize this
    #     if len(self.points) <= 0:
    #         return np.zeros((3, 256, 256))
    #
    #     data: List[List[float]] = []
    #
    #     for point in self.points:
    #         data.append([point.x, point.y, point.z])
    #
    #     point_cloud = np.array(data)
    #
    #     lidar_transformed = point_cloud
    #
    #     # lidar_transformed[:, 1] *= -1  # invert
    #     lidar_transformed = np.expand_dims(
    #         _lidar_to_histogram_features(lidar_transformed), 0
    #     )
    #     lidar_transformed_degrees = lidar_transformed
    #     lidar_bev = lidar_transformed_degrees[::-1]
    #
    #     return lidar_bev[0]

    # def _to_bev(self) -> np.ndarray:
    #     direction = np.pi / 2 - self.compass
    #
    #     full_lidar = transform_2d_points(
    #         self.points,
    #         direction,
    #         -self.gps[0],
    #         -self.gps[1],
    #         direction,
    #         -self.gps[0],
    #         -self.gps[1],
    #     )
    #
    #     lidar_processed = lidar_to_histogram_features(full_lidar)
    #
    #     return lidar_processed


def _to_bev(points: List[LidarPoint], gps: Tuple[float, float], compass: float):
    start_time = time.time()

    if len(points) <= 0:
        return np.zeros((3, 256, 256))

    data: List[List[float]] = []

    for point in points:
        data.append([point.x, point.y, point.z])

    point_cloud = np.array(data)
    print("TIME TO PARSE POINTS TO NP ARRAY: ", time.time() - start_time)

    direction = np.pi / 2 - compass

    full_lidar = transform_2d_points(
        point_cloud,
        direction,
        -gps[0],
        -gps[1],
        direction,
        -gps[0],
        -gps[1],
    )

    lidar_processed = lidar_to_histogram_features(full_lidar)
    print("TIME TO CONVERT TO BEV: ", time.time() - start_time)

    return lidar_processed


def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    # Pick random lidar point

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28

        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )

        # pick random point

        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]

        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel

        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)

    return features


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


# def _lidar_to_histogram_features(lidar):
#     """
#     Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
#     """
#
#     below = lidar[lidar[..., 2] <= -2.3]
#     above = lidar[lidar[..., 2] > -2.3]
#     below_features = _splat_points(below)
#     above_features = _splat_points(above)
#     features = np.stack([above_features, below_features], axis=-1)
#     features = np.transpose(features, (2, 0, 1)).astype(np.float32)
#     features = np.rot90(features, -1, axes=(1, 2)).copy()
#     return features


# def _splat_points(point_cloud):
#     # 256 x 256 grid
#     pixels_per_meter = 8
#     hist_max_per_pixel = 5
#     x_meters_max = 16
#     y_meters_max = 32
#     xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
#     ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
#     hist = np.histogramdd(
#         point_cloud[..., :2],
#         bins=(xbins, ybins),
#     )[0]
#     hist[hist > hist_max_per_pixel] = hist_max_per_pixel
#     overhead_splat = hist / hist_max_per_pixel
#     return overhead_splat


@dataclass
class CameraManagerData:
    # contains each image from each sensor from a given frame
    images: List[np.ndarray]
    # contains the lidar data from each applied sensor from a given frame
    lidar_data: LidarData
    # Third person view camera image
    third_person_view: np.ndarray
    # radar: List[np.ndarray]
