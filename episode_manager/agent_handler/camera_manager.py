import weakref
from typing import List, Optional, Tuple

import carla
import numpy as np

from episode_manager.agent_handler.models import (
    CameraManagerData,
    LidarConfiguration,
    LidarData,
    LidarPoint,
    RGBCameraConfiguration,
)
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform


def from_carla_lidar(
    data: carla.LidarMeasurement, gps: Tuple[float, float], compass: float
) -> LidarData:
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

    return LidarData(points, gps, compass)


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
        self.carla_fps = carla_fps
        self._enable_third_person_view = enable_third_person_view

        # Third person camera
        self.third_person_image = np.array([])
        self.third_person_camera_config = {
            "width": 1280,
            "height": 720,
            "fov": 103,
            "transform": Transform(Location(-7, 0, 4), Rotation(-20, 0, 0)),
        }

        world = self._parent.get_world()
        self.sensors = []

        # Camera sensors
        self.image_data = [
            np.zeros((config["height"], config["width"], 4))
            for config in camera_configs
        ]

        bp_library = world.get_blueprint_library()

        self.set_data = []

        self.sensors = []

        for index, camera in enumerate(camera_configs):
            bp = bp_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", f"{camera['width']}")
            bp.set_attribute("image_size_y", f"{camera['height']}")
            bp.set_attribute("fov", f"{camera['fov']}")
            bp.set_attribute("sensor_tick", f"{1 /self.carla_fps}")

            self.sensors.append(
                world.spawn_actor(
                    bp,
                    camera["transform"].get_carla_transform(),
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
        self.lidar_data: Optional[carla.LidarMeasurement] = None

        def set_lidar_data(data):
            self.lidar_data = data

        if lidar_config is not None and lidar_config["enabled"]:
            bp = bp_library.find("sensor.lidar.ray_cast")
            bp.set_attribute("channels", f"{lidar_config['channels']}")
            bp.set_attribute("range", f"{lidar_config['range']}")
            bp.set_attribute("rotation_frequency", f"{10}")
            bp.set_attribute("upper_fov", str(10))
            bp.set_attribute("lower_fov", str(-30))
            bp.set_attribute(
                "points_per_second", f"{lidar_config['points_per_second']}"
            )

            sensor = world.spawn_actor(
                bp,
                lidar_config["transform"].get_carla_transform(),
                attach_to=self._parent,
            )

            sensor.listen(lambda data: set_lidar_data(data))
            self.sensors.append(sensor)

        return

    def _setup_third_person_view(self, bp_library, world):
        camera = self.third_person_camera_config

        def set_third_person_data(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            image = array.reshape(
                camera["height"],
                camera["width"],
                4,
            )

            self.third_person_image = image

        bp = bp_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", f"{camera['width']}")
        bp.set_attribute("image_size_y", f"{camera['height']}")
        bp.set_attribute("fov", f"{camera['fov']}")
        bp.set_attribute("sensor_tick", f"{1 /self.carla_fps}")

        sensor = world.spawn_actor(
            bp, camera["transform"].get_carla_transform(), attach_to=self._parent
        )

        sensor.listen(lambda image: set_third_person_data(image))
        self.sensors.append(sensor)

        return

    @staticmethod
    def _set_image_data(weak_ref, index, image):
        self = weak_ref()
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = img.reshape(
            self.camera_configs[index]["height"],
            self.camera_configs[index]["width"],
            4,
        )

        self.image_data[index] = img

    def get_sensor_data(
        self, gps: Tuple[float, float], compass: float
    ) -> CameraManagerData:
        lidar = from_carla_lidar(self.lidar_data, gps, compass)

        return CameraManagerData(
            images=self.image_data,
            lidar_data=lidar,
            third_person_view=self.third_person_image,
        )

    def stop(self):
        """
        Stop the sensors
        """
        for sensor in self.sensors:
            print(f"Destroying sensor {sensor}")
            sensor.stop()

        return

    def destroy(self):
        """
        Destroy the sensors
        """
        # for sensor in self.sensors:
        #     sensor.destroy()

        return
