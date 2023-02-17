from dataclasses import dataclass
from typing import List
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

    def __post_init__(self):
        """
        Convert to bird eye view
        """
        self.bev = self._to_bev()

    def _to_bev(self) -> np.ndarray:
        # TODO: find ways to optimize this
        if len(self.points) <= 0:
            return np.zeros((1, 3, 256, 256))

        data: List[List[float]] = []

        for point in self.points:
            data.append([point.x, point.y, point.z])

        point_cloud = np.array(data)

        lidar_transformed = point_cloud

        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = np.expand_dims(
            _lidar_to_histogram_features(lidar_transformed), 0
        )
        lidar_transformed_degrees = lidar_transformed
        lidar_bev = lidar_transformed_degrees[::-1]

        lidar_bev = np.append(
            lidar_bev,
            np.zeros((1, 1, 256, 256)),
            axis=1,
        )

        return lidar_bev


def _lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    below = lidar[lidar[..., 2] <= -2.3]
    above = lidar[lidar[..., 2] > -2.3]
    below_features = _splat_points(below)
    above_features = _splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1, 2)).copy()
    return features


def _splat_points(point_cloud):
    # 256 x 256 grid
    pixels_per_meter = 8
    hist_max_per_pixel = 5
    x_meters_max = 16
    y_meters_max = 32
    xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
    ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
    hist = np.histogramdd(
        point_cloud[..., :2],
        bins=(xbins, ybins),
    )[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    overhead_splat = hist / hist_max_per_pixel
    return overhead_splat


@dataclass
class CameraManagerData:
    # contains each image from each sensor from a given frame
    images: List[np.ndarray]
    # contains the lidar data from each applied sensor from a given frame
    lidar_data: LidarData
    third_person_view: np.ndarray
    # MIGHT COME IN THE FUTURE
    # radar: List[np.ndarray]
