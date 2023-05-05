from dataclasses import dataclass
from typing import Any, DefaultDict, List, Tuple

from episode_manager.agent_handler.models.sensor import CameraManagerData
from episode_manager.agent_handler.models.transform import Transform


@dataclass
class PrivilegedScenarioData:
    dist_to_traffic_light: float
    dist_to_vehicle: float
    dist_to_pedestrian: float
    transform: Transform
    collision_history: DefaultDict[Any, int]
    speed_limit: float = 6.0


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
    global_plan_world_coord: List[Tuple[Transform, int]]
    done: bool


@dataclass
class WorldState:
    ego_vehicle_state: VehicleState
    scenario_state: ScenarioState
