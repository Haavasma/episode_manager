from typing import List

from episode_manager.agent_handler.models import (
    RGBCameraConfiguration,
    LidarConfiguration,
    LidarData,
    LidarPoint,
    CameraManagerData,
)

import numpy as np
import carla


def from_carla_lidar(data: carla.LidarMeasurement) -> LidarData:
    """
    Translate to LidarData from carla.LidarMeasurement
    """
    points = []

    for detection in data:
        points.append(
            LidarPoint(
                detection.point.x,
                detection.point.y,
                detection.point.z,
                detection.intensity,
            )
        )

    return LidarData(points)


class CameraManager:
    def __init__(
        self,
        parent_actor,
        camera_configs: List[RGBCameraConfiguration],
        lidar_config: LidarConfiguration,
    ):
        self._parent = parent_actor

        world = self._parent.get_world()
        self.sensors = []

        # Camera sensors
        self.image_data = [np.ndarray([]) for _ in camera_configs]

        def set_image_data(index, image):
            self.image_data[index] = image.raw_data

        bp_library = world.get_blueprint_library()
        for index, camera in enumerate(camera_configs):
            bp = bp_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", f"{camera.width}")
            bp.set_attribute("image_size_y", f"{camera.height}")
            bp.set_attribute("fov", f"{camera.fov}")
            bp.set_attribute("sensor_tick", f"{camera.sensor_tick}")

            sensor = world.spawn_actor(
                bp, camera.transform.get_carla_transform(), attach_to=self._parent
            )

            sensor.listen(lambda image: set_image_data(index, image))
            self.sensors.append(sensor)

        # Lidar sensors
        self.lidar_data: LidarData = LidarData([])

        def set_lidar_data(data):
            self.lidar_data = from_carla_lidar(data)

        if lidar_config is not None:
            bp = bp_library.find("sensor.lidar.ray_cast")
            bp.set_attribute("range", f"{lidar_config.range}")

            sensor = world.spawn_actor(
                bp, lidar_config.transform.get_carla_transform(), attach_to=self._parent
            )

            sensor.listen(lambda data: set_lidar_data(data))
            self.sensors.append(sensor)

        return

    def get_sensor_data(self) -> CameraManagerData:
        return CameraManagerData(self.image_data, self.lidar_data)

    def stop(self):
        """
        Stop the sensors
        """
        for sensor in self.sensors:
            sensor.stop()

        return

    def destroy(self):
        """
        Destroy the sensors
        """
        for sensor in self.sensors:
            sensor.destroy()

        return
