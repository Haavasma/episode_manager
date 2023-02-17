from dataclasses import dataclass
from typing_extensions import override
import time
import carla
import sys
import numpy as np
import pygame
from manual_control import (
    GnssSensor,
    CollisionSensor,
    LaneInvasionSensor,
    IMUSensor,
    get_actor_display_name,
)

from typing import Tuple

from episode_manager.agent_handler.camera_manager import CameraManager


from .models import (
    CameraManagerData,
    CarConfiguration,
)


@dataclass
class VehicleState:
    sensor_data: CameraManagerData
    speed: float
    gps: Tuple[float, float]
    compass: float
    running: bool


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


@dataclass
class HUD:
    notificatio: str = ""

    def notification(self, notification: str):
        """
        notify the hud with some text
        """
        self._notification = notification


class AgentHandler:
    restarted = False

    @override
    def __init__(
        self,
        carla_world: carla.World,
        car_configuration: CarConfiguration,
        enable_third_person_view: bool = False,
    ):
        self.world = carla_world
        self.config = car_configuration
        self.enable_third_person_view = enable_third_person_view

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
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        print("RESTARTING")
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

        self.hud = HUD()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(
            self.player,
            self.config.cameras,
            self.config.lidar,
            enable_third_person_view=self.enable_third_person_view,
        )
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.world.tick()

    def apply_control(self, control: Action):
        """
        Apply the given control to the ego vehicle
        """
        self.player.apply_control(control.carla_vehicle_control())

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

            sensor_data = self.camera_manager.get_sensor_data()

            return VehicleState(sensor_data, speed, gps, compass, True)

        raise Exception("Sensors not initialized")

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
