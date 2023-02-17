from dataclasses import dataclass
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
    RGBCameraConfiguration,
)
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
from episode_manager.data import TRAINING_TYPE_TO_ROUTES, TrainingType
from episode_manager.models.world_state import WorldState
from episode_manager.renderer import WorldStateRenderer

from episode_manager.scenario_handler import ScenarioHandler


@dataclass
class EpisodeManagerConfiguration:
    host: str = "127.0.0.1"
    port: int = 2000
    render_server: bool = False
    render_client: bool = False
    training_type: TrainingType = TrainingType.TRAINING
    car_config: CarConfiguration = CarConfiguration(
        "tesla",
        [
            RGBCameraConfiguration(
                400,
                400,
                120,
                0.1,
                Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
            ),
            RGBCameraConfiguration(
                400,
                400,
                120,
                0.1,
                Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
            ),
            RGBCameraConfiguration(
                400,
                400,
                120,
                0.1,
                Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
            ),
        ],
        LidarConfiguration(enabled=True),
    )
    route_directory: pathlib.Path = pathlib.Path(
        os.path.join(os.path.dirname(__file__), "routes")
    )


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

        if config.render_client:
            self.world_renderer = WorldStateRenderer()

        if agent_handler is None:
            self.agent_handler = setup_agent_handler(config)

        if scenario_handler is None:
            self.scenario_handler = setup_scenario_handler(config)

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
        files = self.routes[Random().randint(0, len(self.routes) - 1)]
        tree = ET.parse(files.route)

        # pick random id from route
        ids: List[str] = []
        for route in tree.iter("route"):
            ids.append(route.attrib["id"])
        id = ids[Random().randint(0, len(ids) - 1)]

        print("Starting episode with route: " + str(files.route))
        # self.scenario_handler.start_episode(
        #     files.route,
        #     files.scenario,
        #     id,
        # )

        self.scenario_handler.start_episode(
            "/lhome/haavasma/Documents/fordypningsoppgave/repositories/episode_manager/episode_manager/routes/training_routes/routes_town05_tiny.xml",
            "/lhome/haavasma/Documents/fordypningsoppgave/repositories/episode_manager/episode_manager/routes/scenarios/town05_all_scenarios.json",
            "214",
        )

        self.agent_handler.restart()
        self.agent_handler.apply_control(Action(0, 0, False, 0))
        self.scenario_handler.tick()

        return

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario,
        performing the chosen action on the route environment
        """
        self.agent_handler.apply_control(ego_vehicle_action)
        scenario_state = self.scenario_handler.tick()

        agent_state = self.agent_handler.read_world_state()

        world_state = WorldState(
            agent_state, scenario_state, self.scenario_handler.is_running()
        )

        # Render the world state with world_renderer
        if self.world_renderer:
            self.world_renderer.render(world_state)

        return world_state

    def stop_episode(self):
        self.agent_handler.stop()
        return self.scenario_handler.stop_episode()


def setup_agent_handler(config: EpisodeManagerConfiguration) -> AgentHandler:
    client = carla.Client(config.host, config.port)
    client.set_timeout(20.0)
    sim_world = client.get_world()

    # Disable rendering if configured
    settings = sim_world.get_settings()
    if not config.render_server:
        settings.no_rendering_mode = True
    settings.fixed_delta_seconds = 1 / config.car_config.carla_fps

    sim_world.apply_settings(settings)

    agent_handler = AgentHandler(
        sim_world, config.car_config, enable_third_person_view=config.render_client
    )
    return agent_handler


def setup_scenario_handler(config: EpisodeManagerConfiguration) -> ScenarioHandler:
    scenario_handler = ScenarioHandler(
        config.host, config.port, carla_fps=config.car_config.carla_fps
    )
    return scenario_handler
