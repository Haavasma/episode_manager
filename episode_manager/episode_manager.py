from enum import Enum
import enum
import os
import pathlib
from dataclasses import dataclass, field
import random
from random import Random
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import carla
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
    TrafficType,
)
from episode_manager.models.world_state import ScenarioData, WorldState
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
    no_scenario: pathlib.Path


class EpisodeManager:
    def __init__(
        self,
        config: EpisodeManagerConfiguration,
        reset_interval: int = 10,
        gpu_device: int = 0,
        server_wait_time: int = 10,
        iterations: int = 0,
    ):
        def on_signal(sig, frame):
            print("Killing threads and stopping server")
            self.close()

            sys.exit(0)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        # signal.signal(signal.SIGKILL, on_signal)

        self.iterations = iterations
        self.reset_interval = reset_interval
        self.gpu_device = gpu_device
        self.server_wait_time = server_wait_time
        self.statistics = {}

        self.scenario_handler = None
        self.agent_handler = None
        self.world_renderer = None
        self.stopped = True
        self.server = None

        self.config = config

        def get_episodes(training_type: TrainingType) -> List[EpisodeFiles]:
            routes: List[EpisodeFiles] = []

            for path in TRAINING_TYPE_TO_ROUTES[training_type.value]:
                routes.append(
                    EpisodeFiles(
                        config.route_directory / path,
                        config.route_directory / SCENARIOS[0],  # All scenarios
                        config.route_directory / SCENARIOS[1],  # No scenarios
                    )
                )

            return routes

        self.training_routes = get_episodes(TrainingType.TRAINING)
        evaluation_routes = get_episodes(TrainingType.EVALUATION)

        # Shuffle the order of evaluation routes, so that running len(routes)
        # episodes will run all the routes, but n < len(routes)
        # will give a random sample of the routes
        random.shuffle(evaluation_routes)
        self.evaluation_routes = evaluation_routes

        return

    def setup_server(self):
        def on_exit(return_code, stdout, stderr):
            print("Server exited with return code: ", return_code)
            if self.scenario_handler is not None:
                self.scenario_handler.destroy()

        self.server = CarlaServer()
        host, port, tm_port = self.server.start_server(
            on_exit, gpu_device=self.gpu_device, wait_time=self.server_wait_time
        )

        if self.config.render_client:
            self.world_renderer = WorldStateRenderer()

        self.scenario_handler = setup_scenario_handler(
            host, port, tm_port, self.config.carla_fps
        )

        self.agent_handler = setup_agent_handler(host, port, self.config)

    def close(self):
        print("Stopping episode, killing threads and stopping server")
        self.stop_episode()

        # If thes server stopped abnormally, we should not report statistics, as they are not valid
        self.statistics = {}

        if self.agent_handler is not None:
            self.agent_handler = None

        if self.scenario_handler is not None:
            self.scenario_handler.destroy()
            self.scenario_handler = None

        print("Stopping server")
        if self.server is not None:
            self.server.stop_server()
            self.server = None

        print("Stopped server")

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

        self.__init__(
            self.config, gpu_device=self.gpu_device, iterations=self.iterations
        )

    def __del__(self):
        print("STOPPING EPISODE MANAGER")
        self.close()

    def start_episode(
        self,
        town: Optional[str] = None,
        traffic_type: TrafficType = TrafficType.SCENARIO,
    ) -> Tuple[WorldState, ScenarioData]:
        """
        Starts a new route in the simulator based on the provided configurations
        """
        if self.iterations % self.reset_interval == self.reset_interval - 1:
            self.close()

        if self.server is None:
            self.setup_server()

        if not self.stopped:
            raise Exception("Episode has already started")

        self.stopped = False

        routes = (
            self.training_routes
            if self.config.training_type == TrainingType.TRAINING
            else self.evaluation_routes
        )

        file = routes[Random().randint(0, len(routes) - 1)]

        with open(file.route, "r") as f:
            tree = ET.parse(f)

        print("RUNNING TRAINING TYPE: ", self.config.training_type.value)

        # pick random id from route
        ids: List[str] = []
        for route in tree.iter("route"):
            # Skip scenario 1 cause it just doesn't work lol
            if self.config.training_type == TrainingType.EVALUATION and (
                route.attrib["id"] == "1"
                or route.attrib["id"] == "21"
                or route.attrib["id"] == "0"
            ):
                continue

            if town is None or route.attrib["town"] == town:
                ids.append(route.attrib["id"])

        if len(ids) == 0:
            raise Exception("No route found for town: " + town)

        rnd_index = Random().randint(0, len(ids) - 1)

        if self.config.training_type == TrainingType.EVALUATION:
            rnd_index = self.iterations % len(ids)

        self.iterations += 1

        id = ids[rnd_index]

        print("STARTING SCENARIO ID: ", id)

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        self.scenario_data = self.scenario_handler.start_episode(
            file.route,
            file.scenario if traffic_type == TrafficType.SCENARIO else file.no_scenario,
            id,
            traffic_type=traffic_type,
        )
        self.agent_handler.restart()

        agent_state = self.agent_handler.read_world_state()

        return WorldState(ego_vehicle_state=agent_state, done=False), self.scenario_data

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario,
        performing the chosen action on the route environment
        """

        if self.stopped:
            return WorldState(ego_vehicle_state=self.agent_state, done=True)

        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        done = False
        try:
            self.agent_handler.apply_control(ego_vehicle_action)
            done = self.scenario_handler.tick()
            self.agent_state = self.agent_handler.read_world_state()
        except RuntimeError as e:
            print("Error stepping episode: ", e)
            self.server.stop_server()
            raise Exception("Error stepping episode: ", e)

        world_state: WorldState = WorldState(
            ego_vehicle_state=self.agent_state,
            done=done,
        )

        # Render the world state with world_renderer
        if self.world_renderer:
            self.world_renderer.render(world_state)

        if done:
            self.stop_episode()

        return world_state

    def stop_episode(self) -> Dict[str, Any]:
        if self.stopped:
            print("Episode has already stopped")
            return self.statistics

        self.stopped = True
        if self.agent_handler is None:
            raise Exception("Agent handler not initialized")

        if self.scenario_handler is None:
            raise Exception("Scenario handler not initialized")

        self.agent_handler.stop()
        self.statistics = self.scenario_handler.stop_episode()

        return self.statistics


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
    scenario_handler = ScenarioHandler(host, port, tm_port, carla_fps=fps)

    return scenario_handler
