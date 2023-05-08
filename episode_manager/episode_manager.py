from dataclasses import dataclass, field
import pathlib
from typing import List, Optional
from xml.etree import ElementTree as ET
import carla
import os

from random import Random

from srunner.scenarios.route_scenario import RunningStopTest
from carla_server import CarlaServer

from episode_manager.agent_handler import (
    Action,
    AgentHandler,
    CarConfiguration,
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
    carla_fps: float = 10.0
    render_server: bool = False
    render_client: bool = False
    training_type: TrainingType = TrainingType.TRAINING
    car_config: CarConfiguration = CarConfiguration(
        "tesla",
        [
            {
                "height": 300,
                "width": 400,
                "fov": 100,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
            },
            {
                "height": 600,
                "width": 800,
                "fov": 100,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
            },
            {
                "height": 300,
                "width": 400,
                "fov": 100,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
            },
        ],
        {
            "enabled": True,
            "channels": 64,
            "range": 85,
            "shape": (3, 256, 256),
            "points_per_second": 300000,
            "transform": Transform(Location(1.3, 0, 2.5), Rotation(0, 90, 0)),
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
        agent_handler: Optional[AgentHandler] = None,
        scenario_handler: Optional[ScenarioHandler] = None,
        reset_interval: int = 10,
        gpu_device: int = 0,
    ):
        def on_exit(return_code, stdout, stderr):
            print("Server exited with return code: ", return_code)
            if self.scenario_handler is not None:
                self.scenario_handler.destroy()

        self.iterations = 0
        self.reset_interval = reset_interval
        self.gpu_device = gpu_device

        self.config = config

        self.server = CarlaServer()
        host, port, tm_port = self.server.start_server(
            on_exit, gpu_device=self.gpu_device, wait_time=10
        )

        self.host = host
        self.port = port
        self.tm_port = tm_port

        self.scenario_handler = scenario_handler
        self.agent_handler = agent_handler
        self.world_renderer = None
        self.stopped = True
        self.town = ""

        if config.render_client:
            self.world_renderer = WorldStateRenderer()

        if scenario_handler is None:
            self.scenario_handler = setup_scenario_handler(
                host,
                port,
                tm_port,
                config.carla_fps,
            )

        if agent_handler is None:
            self.agent_handler = setup_agent_handler(host, port, config)

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

    def reset(self):
        if self.scenario_handler is not None:
            self.scenario_handler.destroy()
            del self.scenario_handler
            self.scenario_handler = None

        if self.agent_handler is not None:
            del self.agent_handler
            self.agent_handler = None
        if self.server is not None:
            self.server.stop_server()

        self.__init__(self.config, gpu_device=self.gpu_device)

    def start_episode(self, town="Town03") -> WorldState:
        """
        Starts a new route in the simulator based on the provided configurations
        """
        self.town = town

        if not self.stopped:
            raise Exception("Episode has already started")

        if self.iterations >= self.reset_interval:
            self.reset()
            self.iterations = 0

        self.iterations += 1

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

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        tries = 0
        max_tries = 3
        while tries < max_tries:
            try:
                tries += 1
                self.scenario_handler.start_episode(
                    file.route,
                    file.scenario,
                    id,
                )
                self.agent_handler.restart()
                break
            except RuntimeError as e:
                print("Error starting episode: ", e)
                self.reset()
        else:
            self.scenario_handler.destroy()
            self.server.stop_server()
            raise Exception("Could not start episode")

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

        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        try:
            self.agent_handler.apply_control(ego_vehicle_action)
            self.scenario_state = self.scenario_handler.tick()
            self.agent_state = self.agent_handler.read_world_state(self.scenario_state)
        except RuntimeError as e:
            print("Error stepping episode: ", e)
            self.server.stop_server()
            raise Exception("Error stepping episode: ", e)

        world_state: WorldState = WorldState(
            ego_vehicle_state=self.agent_state,
            scenario_state=self.scenario_state,
        )

        # Render the world state with world_renderer
        if self.world_renderer:
            self.world_renderer.render(world_state)

        if self.scenario_state.done:
            self.stop_episode()

        return world_state

    def stop_episode(self):
        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        if not self.stopped:
            self.agent_handler.stop()
            self.scenario_handler.stop_episode()
            self.stopped = True
        else:
            print("Episode has already stopped")

        return


def setup_agent_handler(
    host: str, port: int, config: EpisodeManagerConfiguration
) -> AgentHandler:
    client = carla.Client(host, port)
    client.set_timeout(60)
    sim_world = client.get_world()

    agent_handler = AgentHandler(
        sim_world, config.car_config, enable_third_person_view=config.render_client
    )
    return agent_handler


def setup_scenario_handler(host, port, tm_port, fps) -> ScenarioHandler:
    scenario_handler = ScenarioHandler(
        host,
        port,
        tm_port,
        carla_fps=fps,
    )

    return scenario_handler
