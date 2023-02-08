from dataclasses import dataclass
import pathlib
from typing import List
import carla


from random import Random
from episode_manager.agent_handler import (
    Action,
    AgentHandler,
    CarConfiguration,
    VehicleState,
)
from episode_manager.data import TRAINING_TYPE_TO_ROUTES, TrainingType

from episode_manager.scenario_handler import ScenarioHandler, ScenarioState


@dataclass
class EpisodeManagerConfiguration:
    host: str
    port: int
    training_type: TrainingType
    route_directory: pathlib.Path
    car_config: CarConfiguration


@dataclass
class EpisodeFiles:
    route: pathlib.Path
    scenario: pathlib.Path


@dataclass
class WorldState:
    ego_vehicle_state: VehicleState
    scenario_state: ScenarioState
    running: bool


class EpisodeManager:
    def __init__(
        self,
        config: EpisodeManagerConfiguration,
        agent_handler: AgentHandler,
        scenario_handler: ScenarioHandler = ScenarioHandler(),
    ):

        self.config = config
        self.scenario_handler = scenario_handler

        if agent_handler is None:
            self.agent_handler = setup_agent_handler(config)

        else:
            self.agent_handler = agent_handler

        def get_episodes(training_type: TrainingType) -> List[EpisodeFiles]:
            def get_path(dir: str, file: str):
                return config.route_directory / dir / file

            routes = []

            for path in TRAINING_TYPE_TO_ROUTES[training_type.value]:
                routes.append(
                    EpisodeFiles(
                        route=get_path(training_type.value, path[0]),
                        scenario=get_path("scenarios", path[1]),
                    )
                )
            return routes

        self.routes = get_episodes(
            config.training_type,
        )

        return

    def start_episode(self):
        """
        Starts a new route in the simulator based on the provided configurations
        """
        files = self.routes[Random().randint(0, len(self.routes))]

        print("Starting episode with route: " + str(files.route))

        # TODO: Pick a random scenario from the episodes, instead of hard-coding it to 0
        self.scenario_handler.start_episode(files.route, files.scenario, "0")
        self.agent_handler.restart()

        return

    def step(self, ego_vehicle_action: Action) -> VehicleState:
        """
        Runs one step/frame in the simulated scenario,
        performing the chosen action on the route environment
        """
        self.agent_handler.apply_control(ego_vehicle_action)

        self.scenario_handler.tick()

        return self.agent_handler.read_world_state()

    def stop_episode(self):
        self.agent_handler.stop()
        return self.scenario_handler.stop_episode()


def setup_agent_handler(config: EpisodeManagerConfiguration) -> AgentHandler:
    client = carla.Client(config.host, config.port)
    client.set_timeout(20.0)
    sim_world = client.get_world()
    agent_handler = AgentHandler(sim_world, config.car_config)
    return agent_handler
