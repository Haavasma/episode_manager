from dataclasses import dataclass
from typing import List, Tuple

from episode_manager.agent_handler.models.transform import Location, Rotation, Transform


@dataclass
class RGBCameraConfiguration:
    width: int
    height: int
    fov: int
    transform: Transform


@dataclass
class LidarConfiguration:
    enabled: bool = True
    channels: int = 32
    range: float = 5000
    shape: Tuple[int, int, int] = (3, 256, 256)
    transform: Transform = Transform(Location(1.3, 0.0, 2.5), Rotation(0, -90, 0))


@dataclass
class CarConfiguration:
    model: str
    cameras: List[RGBCameraConfiguration]
    lidar: LidarConfiguration
    carla_fps = 10
    proximity_threshold = 20.0

    def __post_init__(self):
        self.carla_configuration = self.to_carla_leader_board_sensor_configuration()

    def get_carla_configuration(self):
        """
        get the carla confugration for the leaderboard
        """
        return self.carla_configuration

    def to_carla_leader_board_sensor_configuration(self):
        sensors = []
        for index, camera in enumerate(self.cameras):
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": camera.transform.location.x,
                    "y": camera.transform.location.y,
                    "z": camera.transform.location.z,
                    "roll": camera.transform.rotation.roll,
                    "pitch": camera.transform.rotation.pitch,
                    "yaw": camera.transform.rotation.yaw,
                    "width": camera.width,
                    "height": camera.height,
                    "fov": camera.fov,
                    "id": f"rgb_{index}",
                }
            )
        if self.lidar.enabled:
            sensors.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": self.lidar.transform.location.x,
                    "y": self.lidar.transform.location.y,
                    "z": self.lidar.transform.location.z,
                    "roll": self.lidar.transform.rotation.roll,
                    "pitch": self.lidar.transform.rotation.pitch,
                    "yaw": self.lidar.transform.rotation.yaw,
                    "id": "lidar",
                }
            )

        sensors.extend(
            [
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 1 / self.carla_fps,
                    "id": "imu",
                },
                {
                    "type": "sensor.other.gnss",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 1 / self.carla_fps,
                    "id": "gps",
                },
                {
                    "type": "sensor.speedometer",
                    "reading_frequency": self.carla_fps,
                    "id": "speed",
                },
            ]
        )

        return sensors
