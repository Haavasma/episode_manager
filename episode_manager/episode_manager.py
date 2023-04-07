from dataclasses import dataclass, field
import pathlib
from typing import List
from xml.etree import ElementTree as ET
import carla
import os

from random import Random

from episode_manager.agent_handler import (
    Action,
    AgentHandler,
    CarConfiguration,
)
from episode_manager.agent_handler.models.configs import (
    LidarConfiguration,
)
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
from episode_manager.data import (
    EVALUATION_ROUTES,
    SCENARIOS,
    TRAINING_ROUTES,
    TRAINING_TYPE_TO_ROUTES,
    TrainingType,
)
from episode_manager.models.world_state import WorldState
from episode_manager.renderer import WorldStateRenderer

from episode_manager.scenario_handler import ScenarioHandler


@dataclass
class EpisodeManagerConfiguration:
    port: int = 2000
    traffic_manager_port: int = 8000
    host: str = "127.0.0.1"
    render_server: bool = False
    render_client: bool = False
    training_type: TrainingType = TrainingType.TRAINING
    car_config: CarConfiguration = CarConfiguration(
        "tesla",
        [
            {
                "height": 400,
                "width": 400,
                "fov": 103,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
            },
            {
                "height": 400,
                "width": 400,
                "fov": 103,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
            },
            {
                "height": 400,
                "width": 400,
                "fov": 103,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
            },
        ],
        {
            "enabled": True,
            "channels": 32,
            "range": 5000,
            "shape": (3, 256, 256),
            "transform": Transform(Location(1.3, 0, 2.5), Rotation(0, -90, 0)),
        },
    )
    route_directory: pathlib.Path = pathlib.Path(
        os.path.join(os.path.dirname(__file__), "routes")
    )
    training_routes: List[str] = field(default_factory=lambda: TRAINING_ROUTES)
    testing_routes: List[str] = field(default_factory=lambda: EVALUATION_ROUTES)


@dataclass
class EpisodeFiles:
    route: pathlib.Path
    scenario: pathlib.Path


class EpisodeManager:
    def __init__(
        self,
        config: EpisodeManagerConfiguration,
        agent_handler: AgentHandler = None,
        scenario_handler: ScenarioHandler = None,
    ):

        self.config = config
        self.scenario_handler = scenario_handler
        self.agent_handler = agent_handler
        self.world_renderer = None
        self.stopped = True
        self.town = ""

        if config.render_client:
            self.world_renderer = WorldStateRenderer()

        if scenario_handler is None:
            self.scenario_handler = setup_scenario_handler(config)

        if agent_handler is None:
            self.agent_handler = setup_agent_handler(config)

        def get_episodes(training_type: TrainingType) -> List[EpisodeFiles]:
            routes: List[EpisodeFiles] = []

            for path in TRAINING_TYPE_TO_ROUTES[training_type.value]:
                routes.append(
                    EpisodeFiles(
                        config.route_directory / path,
                        config.route_directory / SCENARIOS[0],
                    )
                )

            return routes

        self.routes = get_episodes(
            config.training_type,
        )

        return

    def start_episode(self, town="Town03") -> WorldState:
        """
        Starts a new route in the simulator based on the provided configurations
        """

        self.town = town

        if not self.stopped:
            raise Exception("Episode has already started")

        self.stopped = False
        file = self.routes[Random().randint(0, len(self.routes) - 1)]
        tree = ET.parse(file.route)

        # pick random id from route
        ids: List[str] = []
        for route in tree.iter("route"):
            if route.attrib["town"] == town:
                ids.append(route.attrib["id"])

        if len(ids) == 0:
            raise Exception("No route found for town: " + town)

        rnd_index = Random().randint(0, len(ids) - 1)
        id = ids[rnd_index]

        self.scenario_handler.start_episode(
            file.route,
            file.scenario,
            id,
        )

        self.agent_handler.restart()

        scenario_state = self.scenario_handler.tick()
        agent_state = self.agent_handler.read_world_state(scenario_state)

        return WorldState(ego_vehicle_state=agent_state, scenario_state=scenario_state)

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario,
        performing the chosen action on the route environment
        """
        if self.stopped:
            raise Exception("Episode has already stopped")

        self.agent_handler.apply_control(ego_vehicle_action)
        scenario_state = self.scenario_handler.tick()
        agent_state = self.agent_handler.read_world_state(scenario_state)

        world_state: WorldState = WorldState(
            ego_vehicle_state=agent_state,
            scenario_state=scenario_state,
        )

        # Render the world state with world_renderer
        if self.world_renderer:
            self.world_renderer.render(world_state)

        if scenario_state.done:
            self.stop_episode()

        return world_state

    def stop_episode(self):
        if not self.stopped:
            self.agent_handler.stop()
            self.scenario_handler.stop_episode()
            self.stopped = True
        else:
            print("Episode has already stopped")
        return


def setup_agent_handler(config: EpisodeManagerConfiguration) -> AgentHandler:
    client = carla.Client(config.host, config.port)
    client.set_timeout(60)
    sim_world = client.get_world()

    agent_handler = AgentHandler(
        sim_world, config.car_config, enable_third_person_view=config.render_client
    )
    return agent_handler


def setup_scenario_handler(config: EpisodeManagerConfiguration) -> ScenarioHandler:
    scenario_handler = ScenarioHandler(
        config.host,
        config.port,
        config.traffic_manager_port,
        carla_fps=config.car_config.carla_fps,
    )

    return scenario_handler
