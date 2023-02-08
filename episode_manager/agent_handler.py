from dataclasses import dataclass
from threading import Thread
from typing_extensions import override
import time
import weakref
import carla
import sys
import pygame
from manual_control import (
    HUD,
    World,
    GnssSensor,
    CollisionSensor,
    LaneInvasionSensor,
    IMUSensor,
    get_actor_display_name,
)

import numpy as np
from typing import List, Tuple

from srunner.scenariomanager.timer import GameTime


@dataclass
class Location:
    x: float
    y: float
    z: float

    def get_carla_location(self):
        return carla.Location(self.x, self.y, self.z)


@dataclass
class Rotation:
    pitch: float
    yaw: float
    roll: float

    def get_carla_rotation(self):
        return carla.Rotation(self.pitch, self.yaw, self.roll)


@dataclass
class Transform:
    location: Location
    rotation: Rotation

    def __post_init__(self):
        self.carla_transform = carla.Transform(
            self.location.get_carla_location(), self.rotation.get_carla_rotation()
        )

    def get_carla_transform(self) -> carla.Transform:
        """
        Returns a carla Transform with the data of the current Transform
        """
        return self.carla_transform


@dataclass
class LidarPoint:
    x: float
    y: float
    z: float
    intensity: float


@dataclass
class LidarData:
    points: List[LidarPoint]


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


@dataclass
class CameraManagerData:
    # contains each image from each sensor from a given frame
    images: List[np.ndarray]
    # contains the lidar data from each applied sensor from a given frame
    lidar_scans: List[LidarData]
    # radar: List[np.ndarray]


@dataclass
class RGBCameraConfiguration:
    width: int
    height: int
    fov: int
    sensor_tick: float
    transform: Transform


@dataclass
class LidarConfiguration:
    channels: int
    range: float
    transform: Transform


@dataclass
class CarConfiguration:
    model: str
    cameras: List[RGBCameraConfiguration]
    lidars: List[LidarConfiguration]


@dataclass
class VehicleState:
    sensor_data: CameraManagerData
    speed: float
    gps: Tuple[float, float]
    compass: float
    distance_to_red_light: float

    running: bool


# TODO: keep in mind that in the environment that uses this, There should be an injectable
# function that translates "SensorData" into tensor or data that is directly usable by the encoder model


# Injectable function that takes SensorData and encoder as input,
# then returns a compact vision_encoding numpy array
# The shape of the vision_encoding should also be specified
# Need to be able to configure n previous frames to include as input to the function
# The N frames should be in the environment configuration, while the episode manager
# Only has to work with 1 frame at a time.


@dataclass
class Action:
    throttle: float
    brake: float
    reverse: bool
    steer: float

    def carla_vehicle_control(self):
        return carla.VehicleControl(
            throttle=self.throttle,
            brake=self.brake,
            reverse=self.reverse,
            steer=self.steer,
        )


class AgentHandler(World):
    @override
    def __init__(
        self,
        carla_world: carla.World,
        car_configuration: CarConfiguration,
        render=False,
    ):
        self.render_enabled = render
        self.config = car_configuration

        pygame.init()
        pygame.font.init()
        width = 1280
        height = 720

        self.display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        self.display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(width, height)

        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print(
                "  Make sure it exists, has the same name of your town, and is correct."
            )
            sys.exit(1)
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.hud = hud
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    @override
    def restart(self):
        if self.restarted:
            return
        self.restarted = True
        self.stop()

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in possible_vehicles:
                if vehicle.attributes["role_name"] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break

        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(
            self.player, self.config.cameras, self.config.lidars
        )
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def apply_control(self, control: Action):
        """
        Apply the given control to the ego vehicle
        """
        self.player.apply_control(control.carla_vehicle_control())

        if self.render_enabled:
            self.render(self.display)
            pygame.display.flip()

        return

    def read_world_state(self) -> VehicleState:
        """
        Read the state of the world from the sensors of the vehicle
        """
        if (
            self.camera_manager is not None
            and self.gnss_sensor is not None
            and self.imu_sensor is not None
        ):
            # Extract the speed of the vehicle from the gyroscope
            gps = (self.gnss_sensor.lat, self.gnss_sensor.lon)
            compass = self.imu_sensor.compass
            speed = get_forward_speed(self.player)

            return VehicleState(
                self.camera_manager.get_sensor_data(), speed, gps, compass, 0, True
            )

        raise Exception("Camera manager is not initialized")

    def stop(self):
        """
        stop and destory all attached sensors
        """

        if self.camera_manager is not None:
            self.camera_manager.stop()
            self.camera_manager.destroy()

        for index, sensor in enumerate(
            [
                self.collision_sensor,
                self.lane_invasion_sensor,
                self.gnss_sensor,
                self.imu_sensor,
            ]
        ):
            if sensor is not None:
                print(f"Stopping sensor {index}")
                sensor.sensor.stop()
                sensor.sensor.destroy()


class CameraManager:
    def __init__(
        self,
        parent_actor,
        camera_configs: List[RGBCameraConfiguration],
        lidar_configs: List[LidarConfiguration],
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
        self.lidar_data: List[LidarData] = [LidarData([]) for _ in lidar_configs]

        def set_lidar_data(index, data):
            self.lidar_data[index] = from_carla_lidar(data)

        for index, lidar in enumerate(lidar_configs):
            bp = bp_library.find("sensor.lidar.ray_cast")
            bp.set_attribute("range", f"{lidar.range}")

            sensor = world.spawn_actor(
                bp, lidar.transform.get_carla_transform(), attach_to=self._parent
            )

            sensor.listen(lambda data: set_lidar_data(index, data))
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

    def render(self):
        """
        kept for compatibility
        """
        pass


def get_forward_speed(player):
    """Convert the vehicle transform directly to forward speed"""
    velocity = player.get_velocity()
    transform = player.get_transform()

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array(
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
    )
    speed = np.dot(vel_np, orientation)
    return speed
