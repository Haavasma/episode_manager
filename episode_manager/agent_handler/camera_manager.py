from typing import List
import weakref

from episode_manager.agent_handler.models import (
    RGBCameraConfiguration,
    LidarConfiguration,
    LidarData,
    LidarPoint,
    CameraManagerData,
)

import numpy as np
import carla

from episode_manager.agent_handler.models.transform import Location, Rotation, Transform


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
    """
    Camera manager for the hero vehicle, adds sensors and listens to the data stream
    based on the configuration.
    """

    def __init__(
        self,
        parent_actor,
        camera_configs: List[RGBCameraConfiguration],
        lidar_config: LidarConfiguration,
        carla_fps: int = 10,
        enable_third_person_view: bool = False,
    ):
        self._parent = parent_actor
        self.camera_configs = camera_configs
        self.lidar_config = lidar_config
        self._enable_third_person_view = enable_third_person_view

        # Third person camera
        self.third_person_image = None
        self.third_person_camera_config = RGBCameraConfiguration(
            1280,
            720,
            103,
            1 / carla_fps,
            Transform(Location(-7, 0, 4), Rotation(-20, 0, 0)),
        )

        world = self._parent.get_world()
        self.sensors = []

        # Camera sensors
        self.image_data = [np.array([]) for _ in camera_configs]

        bp_library = world.get_blueprint_library()

        self.set_data = []

        self.sensors = []

        for index, camera in enumerate(camera_configs):
            bp = bp_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", f"{camera.width}")
            bp.set_attribute("image_size_y", f"{camera.height}")
            bp.set_attribute("fov", f"{camera.fov}")
            bp.set_attribute("sensor_tick", f"{camera.sensor_tick}")

            self.sensors.append(
                world.spawn_actor(
                    bp,
                    camera.transform.get_carla_transform(),
                    attach_to=self._parent,
                )
            )

            # Thank you chatGPT for fixing this <3
            def create_callback_function(index: int):
                def callback_function(image):
                    CameraManager._set_image_data(weakref.ref(self), index, image)

                return callback_function

            # To keep sensors alive
            self.sensors[index].listen(create_callback_function(index))

        if self._enable_third_person_view:
            self._setup_third_person_view(bp_library, world)

        # Lidar sensors
        self.lidar_data: LidarData = LidarData([])

        def set_lidar_data(data):
            self.lidar_data = from_carla_lidar(data)

        if lidar_config is not None:
            bp = bp_library.find("sensor.lidar.ray_cast")
            bp.set_attribute("channels", f"{lidar_config.channels}")
            bp.set_attribute("range", f"{lidar_config.range}")
            # bp.set_attribute("rotation_frequency", f"{10}")

            sensor = world.spawn_actor(
                bp, lidar_config.transform.get_carla_transform(), attach_to=self._parent
            )

            sensor.listen(lambda data: set_lidar_data(data))
            self.sensors.append(sensor)

        return

    def _setup_third_person_view(self, bp_library, world):
        camera = self.third_person_camera_config

        def set_third_person_data(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            image = array.reshape(
                camera.height,
                camera.width,
                4,
            )

            self.third_person_image = image

        bp = bp_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", f"{camera.width}")
        bp.set_attribute("image_size_y", f"{camera.height}")
        bp.set_attribute("fov", f"{camera.fov}")
        bp.set_attribute("sensor_tick", f"{camera.sensor_tick}")

        sensor = world.spawn_actor(
            bp, camera.transform.get_carla_transform(), attach_to=self._parent
        )

        sensor.listen(lambda image: set_third_person_data(image))
        self.sensors.append(sensor)

        return

    @staticmethod
    def _set_image_data(weak_ref, index, image):
        self = weak_ref()
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = img.reshape(
            self.camera_configs[index].height,
            self.camera_configs[index].width,
            4,
        )

        self.image_data[index] = img

    def get_sensor_data(self) -> CameraManagerData:
        return CameraManagerData(
            self.image_data, self.lidar_data, self.third_person_image
        )

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
