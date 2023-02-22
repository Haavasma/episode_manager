from dataclasses import dataclass
from typing_extensions import override
import time
import carla
import sys
import numpy as np
import math
from manual_control import (
    GnssSensor,
    CollisionSensor,
    LaneInvasionSensor,
    IMUSensor,
    get_actor_display_name,
)

from typing import Any, Callable, Tuple

from episode_manager.agent_handler.camera_manager import CameraManager


from .models import (
    CameraManagerData,
    CarConfiguration,
)


@dataclass
class PrivilegedScenarioData:
    dist_to_traffic_light: float
    dist_to_vehicle: float
    dist_to_pedestrian: float
    dist_to_route: float


@dataclass
class VehicleState:
    sensor_data: CameraManagerData
    speed: float
    gps: Tuple[float, float]
    compass: float
    running: bool
    privileged: PrivilegedScenarioData


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

        self.stopped = True

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
        if not self.stopped:
            self.stop()
        self.player = None
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

        self.stopped = False

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

            return VehicleState(
                sensor_data,
                speed,
                gps,
                compass,
                True,
                self._get_privileged_scenario_data(),
            )

        raise Exception("Sensors not initialized")

    def _get_privileged_scenario_data(self) -> PrivilegedScenarioData:
        """
        Gathers privileged scenario data
        """
        actor_list = self.world.get_actors()

        # DISTANCE TO CLOSEST FACING TRAFFIC LIGHT

        player_location = self.player.get_location()

        def get_distance(actors, condition: Callable[[Any], bool]) -> float:
            min_angle = 180.0
            result_distance = -1.0
            if len(actors) > 0:
                for actor in actors:
                    traffic_location = actor.get_location()

                    magnitude, angle = compute_magnitude_angle(
                        traffic_location,
                        player_location,
                        self.player.get_transform().rotation.yaw,
                    )

                    # print(f"TRAFFIC LIGHT STATE: {light.state}")

                    if (
                        magnitude < 80.0
                        and angle < min(25.0, min_angle)
                        and condition(actor)
                    ):
                        min_angle = angle
                        result_distance = float(magnitude)
            return result_distance

        # DISTANCE TO CLOSEST FACING TRAFFIC LIGHT
        lights = actor_list.filter("*traffic_light*")
        red_light_distance = get_distance(
            lights, lambda light: light.state == carla.libcarla.TrafficLightState.Red
        )

        # DISTANCE TO CLOSEST FACING VEHICLE
        vehicles = actor_list.filter("*vehicle*")
        vehicle_front_distance = get_distance(vehicles, lambda _: True)

        # DISTANCE TO CLOSEST FACING PEDESTRIAN
        pedestrians = actor_list.filter("*walker*")
        pedestrian_front_distance = get_distance(pedestrians, lambda _: True)

        return PrivilegedScenarioData(
            red_light_distance, vehicle_front_distance, pedestrian_front_distance, 0
        )

    def stop(self):
        """
        stop and destory all attached sensors
        """
        if self.camera_manager is not None:
            self.camera_manager.stop()

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

        self.stopped = True


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array(
        [target_location.x - current_location.x, target_location.y - current_location.y]
    )
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))]
    )
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_target)
    )

    return (norm_target, d_angle)


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
