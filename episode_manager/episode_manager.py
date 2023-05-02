import os
import pathlib
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from episode_manager.agent_handler import (
    Action,
)
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
    carla_fps: int = 10
    sensor_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 100,
                "height": 100,
                "fov": 100,
                "id": "rgb_front",
            },
            # {
            #     "type": "sensor.camera.rgb",
            #     "x": -1.3,
            #     "y": 0.0,
            #     "z": 2.3,
            #     "roll": 0.0,
            #     "pitch": 0.0,
            #     "yaw": 180.0,
            #     "width": 800,
            #     "height": 600,
            #     "fov": 100,
            #     "id": "rgb_rear",
            # },
            # {
            #     "type": "sensor.camera.rgb",
            #     "x": 1.3,
            #     "y": 0.0,
            #     "z": 2.3,
            #     "roll": 0.0,
            #     "pitch": 0.0,
            #     "yaw": -60.0,
            #     "width": 800,
            #     "height": 600,
            #     "fov": 100,
            #     "id": "rgb_left",
            # },
            # {
            #     "type": "sensor.camera.rgb",
            #     "x": 1.3,
            #     "y": 0.0,
            #     "z": 2.3,
            #     "roll": 0.0,
            #     "pitch": 0.0,
            #     "yaw": 60.0,
            #     "width": 800,
            #     "height": 600,
            #     "fov": 100,
            #     "id": "rgb_right",
            # },
            # {
            #     "type": "sensor.lidar.ray_cast",
            #     "x": 1.3,
            #     "y": 0.0,
            #     "z": 2.5,
            #     "roll": 0.0,
            #     "pitch": 0.0,
            #     "yaw": 0.0,
            #     "id": "lidar",
            # },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]
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
        # agent_handler: AgentHandler = None,
        scenario_handler: Optional[ScenarioHandler] = None,
    ):
        self.config = config
        self.scenario_handler = scenario_handler
        # self.agent_handler = agent_handler
        self.world_renderer = None
        self.stopped = True
        self.town = ""

        if config.render_client:
            self.world_renderer = WorldStateRenderer()

        if scenario_handler is None:
            self.scenario_handler = setup_scenario_handler(config)

        # if agent_handler is None:
        #     self.agent_handler = setup_agent_handler(config)

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

    def start_episode(self, town="Town03") -> Tuple[WorldState, List]:
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

        if self.scenario_handler is None:
            raise Exception("Scenario handler is not initialized")

        return_value = self.scenario_handler.start_episode(
            file.route,
            file.scenario,
            id,
        )

        print("GLOBAL PLAN: ", return_value[1])

        return return_value

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario,
        performing the chosen action on the route environment
        """
        if self.stopped:
            raise Exception("Episode has already stopped")

        if self.scenario_handler is None:
            raise Exception("Scenario handler is not initialized")

        # self.agent_handler.apply_control(ego_vehicle_action)

        world_state = self.scenario_handler.step(ego_vehicle_action)
        # agent_state = self.agent_handler.read_world_state(scenario_state)

        # Render the world state with world_renderer
        if self.world_renderer:
            self.world_renderer.render(world_state)

        if world_state.done:
            self.stop_episode()

        return world_state

    def stop_episode(self):
        if self.scenario_handler is None:
            raise Exception("Scenario handler is not initialized")

        if not self.stopped:
            # self.agent_handler.stop()
            self.scenario_handler.stop_episode()
            self.stopped = True
        else:
            print("Episode has already stopped")
        return


# def setup_agent_handler(config: EpisodeManagerConfiguration) -> AgentHandler:
#     client = carla.Client(config.host, config.port)
#     client.set_timeout(30)
#     sim_world = client.get_world()
#
#     agent_handler = AgentHandler(
#         sim_world, config.car_config, enable_third_person_view=config.render_client
#     )
#     return agent_handler


def setup_scenario_handler(config: EpisodeManagerConfiguration) -> ScenarioHandler:
    scenario_handler = ScenarioHandler(
        config.host,
        config.port,
        config.traffic_manager_port,
        carla_fps=config.carla_fps,
        sensor_list=config.sensor_list,
    )

    return scenario_handler
