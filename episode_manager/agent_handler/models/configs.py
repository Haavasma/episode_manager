from dataclasses import dataclass
from typing import List

from episode_manager.agent_handler.models.transform import Location, Rotation, Transform


@dataclass
class RGBCameraConfiguration:
    width: int
    height: int
    fov: int
    sensor_tick: float
    transform: Transform


@dataclass
class LidarConfiguration:
    enabled: bool = True
    channels: int = 32
    range: float = 50
    transform: Transform = Transform(Location(0, 0, 2), Rotation(0, 0, 0))


# TODO: Agent class that is used to act based on the observations,
# then returns an action, all in the format of the types specified above
# RLAgent:
# A wrapper to that agent should be compatible with running an autonomous agent
# in the CARLA simulator,
# CARLALeaderboardAgentWrapper(AutonomousAgent):
#     rl_agent: RLAgent
# The agent should have an interface that takes in the WorldState (only unprivileged information)
# and returns the encoded state from the model ()
# If render is true, auxillary information should also be returned
# (numpy array of the images we want to show from the auxillary predictions)
# And these should be passed to agent_handler's render method


# So for RL training, we use the RLAgent class,
# Where vision encoder can be passed, function that handles information
# from the env and returns visino encoding


@dataclass
class CarConfiguration:
    model: str
    cameras: List[RGBCameraConfiguration]
    lidar: LidarConfiguration
    carla_fps = 10

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
