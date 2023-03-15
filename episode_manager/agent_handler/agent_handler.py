import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, List

import carla
import numpy as np
from manual_control import (
    CollisionSensor,
    GnssSensor,
    IMUSensor,
    LaneInvasionSensor,
    get_actor_display_name,
)
from typing_extensions import override

from episode_manager.agent_handler.camera_manager import CameraManager
from episode_manager.agent_handler.models.transform import (
    from_carla_transform,
)
from episode_manager.models.world_state import (
    PrivilegedScenarioData,
    ScenarioState,
    VehicleState,
)

from .models import (
    CarConfiguration,
)


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
    _notification: str = ""

    def notification(self, notification: str):
        """
        notify the hud with some text
        """
        self._notification = notification


class AgentHandler:
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

    def read_world_state(self, scenario_state: ScenarioState) -> VehicleState:
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
                sensor_data=sensor_data,
                speed=speed,
                gps=gps,
                compass=compass,
                privileged=self._get_privileged_scenario_data(scenario_state),
            )

        raise Exception("Sensors not initialized")

    def _get_privileged_scenario_data(
        self, scenario_state: ScenarioState
    ) -> PrivilegedScenarioData:
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

        # DISTANCE TO CLOSEST FACING VEHICLE TODO: rework to checking the lane instead
        vehicles = actor_list.filter("*vehicle*")
        vehicle_front_distance = self.distance_to_closest_vehicle(vehicles)

        # DISTANCE TO CLOSEST FACING PEDESTRIAN
        pedestrians = actor_list.filter("*walker*")
        pedestrian_front_distance = get_distance(pedestrians, lambda _: True)

        return PrivilegedScenarioData(
            dist_to_traffic_light=red_light_distance,
            dist_to_vehicle=vehicle_front_distance,
            dist_to_pedestrian=pedestrian_front_distance,
            transform=from_carla_transform(self.player.get_transform()),
        )

    def distance_to_closest_vehicle(self, vehicle_list: List[Any]) -> float:
        ego_vehicle_location = self.player.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        min_distance = self.config.proximity_threshold

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self.player.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(
                target_vehicle.get_location()
            )
            if (
                target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id
                or target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            ):
                continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(
                loc,
                ego_vehicle_location,
                self.player.get_transform().rotation.yaw,
                self.config.proximity_threshold,
            ):
                # calculate the distance to the vehicle in front
                distance = math.sqrt(
                    (loc.x - ego_vehicle_location.x) ** 2
                    + (loc.y - ego_vehicle_location.y) ** 2
                    + (loc.z - ego_vehicle_location.z) ** 2
                )

                if distance < min_distance:
                    min_distance = distance

        return min_distance if min_distance < self.config.proximity_threshold else -1.0

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


def is_within_distance_ahead(
    target_location, current_location, orientation, max_distance
):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array(
        [target_location.x - current_location.x, target_location.y - current_location.y]
    )
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))]
    )
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_target)
    )

    return d_angle < 45.0
