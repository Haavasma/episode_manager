from dataclasses import dataclass

from episode_manager.agent_handler.agent_handler import VehicleState
from episode_manager.scenario_handler import ScenarioState


@dataclass
class WorldState:
    ego_vehicle_state: VehicleState
    scenario_state: ScenarioState
    running: bool
