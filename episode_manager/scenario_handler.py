import pathlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla
from leaderboard.autoagents import autonomous_agent
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.statistics_manager import StatisticsManager
from typing_extensions import override

from episode_manager.agent_handler.agent_handler import Action
from episode_manager.models.world_state import WorldState

# TODO:
# IMPLEMENT FOR LEADERBOARD_EVALUATOR
# ROUTEINDEXER
# AGENTWRAPPER
# INJECT AGENT INSTANCE
# PROFIT


# Backhanded solution to injecting information
# to the scenario runner and sending messages across threads


class DummyAgent(autonomous_agent.AutonomousAgent):
    _sensors_list: Optional[List[Dict]] = None

    def setup(self, sensors: List[Dict]) -> bool:
        self._sensors_list = sensors

        self.control = carla.VehicleControl(throttle=0, steer=0.0)
        return True

    def set_control(self, control: Action):
        self.control = control.carla_vehicle_control()
        return

    def get_world_state(self) -> dict:
        return self.sensor_interface.get_data()

    def run_step(self, input_data, timestamp):
        return self.control

    def sensors(self):
        print("GETTING SENSORS")
        print("SENSORS: ", self._sensors_list)
        return self._sensors_list


@dataclass
class ScenarioHandler:
    """
    Should interact with Scenario runner to set up the given route, as well as tick the scenario whenever ordered to.
    """

    host: str
    port: int
    traffic_manager_port: int
    carla_fps: int = 10
    sensor_list: List[Dict] = field(default_factory=lambda: [])

    def __post_init__(self):
        print("Setting up scenario handler")
        self._scenario_runner: Optional[ScenarioRunnerControlled] = None

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenario_file: pathlib.Path,
        route_id: str,
    ) -> Tuple[WorldState, List]:
        if self._scenario_runner is None:
            self._scenario_runner = self.setup_scenario_runner(
                route_file, scenario_file, route_id
            )

        route_index = RouteIndexer(route_file, scenario_file, 1, route_id)
        route_config = route_index.next()

        self._scenario_runner._load_and_run_scenario(
            self.get_args(route_file, scenario_file, route_id),
            route_config,
        )

        world_state = self.step(Action(0, 1.0, False, 0.0))

        if self._scenario_runner.agent_instance is None:
            raise Exception("Agent instance is None")

        if self._scenario_runner.agent_instance._global_plan is None:
            raise Exception("Global plan is None")

        return (
            world_state,
            self._scenario_runner.agent_instance._global_plan,
        )

    def setup_scenario_runner(self, route_file, scenario_file, route_id):
        statistics_manager = StatisticsManager()
        return ScenarioRunnerControlled(
            self.get_args(route_file, scenario_file, route_id),
            statistics_manager,
            self.sensor_list,
        )

    def get_args(self, route_file, scenario_file, route_id, timeout=10.0):
        output = (
            pathlib.Path("./output/{int(time.time())}_{uuid.uuid4().int}")
            .absolute()
            .resolve()
        )

        pathlib.Path(output).mkdir(parents=True, exist_ok=True)

        return Namespace(
            sync=True,
            port=self.port,
            timeout=timeout,
            routes=route_file,
            route_id=route_id,
            scenarios=scenario_file,
            host=self.host,
            agent=__file__,
            agent_config="",
            track="SENSORS",
            checkpoint=str(output / "checkpoint.json"),
            debug=False,
            openscenario=None,
            repetitions=1,
            reloadWorld=True,
            traffic_manager_port=self.traffic_manager_port,
            traffic_manager_seed=0,
            waitForEgo=False,
            record="",
            outputDir=output,
            junit=True,
            json=True,
            file=True,
            output=True,
        )

    def step(self, action: Action) -> WorldState:
        # wait for all ticks on server to be processed
        if self._scenario_runner is None:
            raise Exception("Scenario runner not initialized")

        if self._scenario_runner.agent_instance is None:
            raise Exception("Agent instance not initialized")

        self._scenario_runner.agent_instance.set_control(action)
        self._scenario_runner.step()

        input_data = self._scenario_runner.agent_instance.get_world_state()

        return WorldState(
            input_data=input_data,
            done=not self._scenario_runner.manager._running,
            privileged=None,
        )

    def destroy(self):
        client = carla.Client(self.host, self.port)
        tm = client.get_trafficmanager(self.traffic_manager_port)
        tm.shut_down()

    def stop_episode(self):
        """
        Stop the scenario runner loop
        """
        if self._scenario_runner is None:
            raise Exception("Scenario runner not initialized")

        self._scenario_runner.stop()


def get_entry_point():
    return "DummyAgent"


class ScenarioRunnerControlled(LeaderboardEvaluator):
    @override
    def __init__(
        self,
        args,
        statistics_manager: StatisticsManager,
        sensor_list: List[Dict],
        fps: float = 10,
    ):
        """
        Setup CARLA client and world
        """

        print("SRUNNER INIT")

        self._sensor_list = sensor_list
        self.agent_instance: Optional[DummyAgent] = None
        self.frame_rate = fps
        return super().__init__(args, statistics_manager)

    def _load_and_run_scenario(self, args, config):
        self.agent_instance = DummyAgent(self._sensor_list)
        config.agent = self.agent_instance

        return super()._load_and_run_scenario(args, config)
