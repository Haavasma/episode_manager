from dataclasses import dataclass
from typing import Any, List, Tuple

from episode_manager.agent_handler.models.sensor import CameraManagerData
from episode_manager.agent_handler.models.transform import Location


@dataclass
class PrivilegedScenarioData:
    dist_to_traffic_light: float
    dist_to_vehicle: float
    dist_to_pedestrian: float
    ego_vehicle_location: Location


@dataclass
class VehicleState:
    sensor_data: CameraManagerData
    speed: float
    gps: Tuple[float, float]
    compass: float
    privileged: PrivilegedScenarioData


@dataclass
class ScenarioState:
    global_plan: List[Any]
    global_plan_world_coord: List[Any]
    done: bool


@dataclass
class WorldState:
    ego_vehicle_state: VehicleState
    scenario_state: ScenarioState
