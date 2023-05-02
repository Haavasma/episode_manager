from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from episode_manager.agent_handler.models.sensor import CameraManagerData
from episode_manager.agent_handler.models.transform import Transform


@dataclass
class PrivilegedScenarioData:
    dist_to_traffic_light: float
    dist_to_vehicle: float
    dist_to_pedestrian: float
    transform: Transform
    collision_history: DefaultDict[Any, int]
    speed_limit: float = 8.0


@dataclass
class WorldState:
    input_data: Dict[str, Any]
    privileged: Optional[PrivilegedScenarioData]
    done: bool
