import datetime
import os
import pathlib
import threading
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List, Optional

import carla
import traceback
import py_trees
from scenario_runner import RouteParser, RouteScenario, ScenarioManager, ScenarioRunner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenarios.route_scenario import BackgroundActivity, interpolate_trajectory
from srunner.tools.route_manipulation import downsample_route
from typing_extensions import override

from episode_manager.agent_handler.models.transform import from_carla_transform
from episode_manager.data import TrafficType
from episode_manager.models.world_state import ScenarioData

# Backhanded solution to injecting information
# to the scenario runner and sending messages across threads


def runner_loop(
    args,
    carla_fps: int,
    tick_queue: Queue,
    ticked_event: threading.Event,
    episode_started: threading.Event,
    episode_stopped: threading.Event,
    trajectory: List,
    route_queue: Queue,
    episode_statistics: Queue,
):
    if not pathlib.Path(args.outputDir).exists():
        pathlib.Path(args.outputDir).mkdir(parents=True)

    manager = ScenarioManagerControlled(
        tick_queue,
        ticked_event,
        episode_started,
        episode_statistics,
        trajectory,
        args.debug,
        args.sync,
        args.timeout,
    )

    scenario_runner = ScenarioRunnerControlled(args, manager)
    scenario_runner.frame_rate = carla_fps

    while True:
        route_file, scenario_file, route_id, traffic_type = route_queue.get()
        if route_file == "stop":
            break
        scenario_runner.finished = False

        route_configs = RouteParser.parse_routes_file(
            route_file, scenario_file, route_id
        )

        assert len(route_configs) == 1
        scenario_runner._run_route(route_configs[0], traffic_type)
        scenario_runner._cleanup()
        episode_stopped.set()


@dataclass
class ScenarioHandler:
    """
    Should interact with Scenario runner to set up the given route, as well as tick the scenario whenever ordered to.
    """

    host: str
    port: int
    traffic_manager_port: int
    carla_fps: int = 10
    _runner_thread: Optional[threading.Thread] = None
    _route_queue: Queue = field(default_factory=lambda: Queue())
    _tick_queue: Queue = field(default_factory=lambda: Queue())
    _ticked_event: threading.Event = field(default_factory=lambda: threading.Event())
    _trajectory: List = field(default_factory=lambda: [])
    _episode_started: threading.Event = field(default_factory=lambda: threading.Event())
    _episode_stopped: threading.Event = field(default_factory=lambda: threading.Event())
    _episode_statistics: Queue = field(default_factory=lambda: Queue())
    _tick_timeout = 10.0
    _episode_timeout = 60.0
    destroyed = False

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenario_file: pathlib.Path,
        route_id: str,
        traffic_type: TrafficType = TrafficType.SCENARIO,
    ) -> ScenarioData:
        self._episode_started.clear()
        self._episode_stopped.clear()

        self.traffic_type = traffic_type

        if self._runner_thread is None:
            self.start_runner_thread(
                route_file, scenario_file, route_id, self._episode_timeout
            )

        self._route_queue.put((route_file, scenario_file, route_id, traffic_type))
        print("WAITING FOR EPISODE TO START")
        ok = self._episode_started.wait(timeout=self._episode_timeout)
        if not ok:
            print("Scenario runner did not start episode in time")
            raise RuntimeError("Scenario runner did not start episode in time")
        print("EPISODE STARTED")

        trajectory = [dict_to_carla_location(x) for x in self._trajectory]
        gps_route, route = interpolate_trajectory(trajectory)
        ds_ids_hack = downsample_route(route, 1)

        # Privileged high-res route
        global_plan_world_coord_privileged = [
            (route[x][0], route[x][1]) for x in ds_ids_hack
        ]

        self._global_plan_world_coord_privileged = [
            (from_carla_transform(point[0]), point[1])
            for point in global_plan_world_coord_privileged
        ]

        global_plan = [gps_route[x] for x in ds_ids_hack]

        # Downsampled low-res route
        ds_ids = downsample_route(global_plan_world_coord_privileged, 50)
        global_plan_world_coord = [
            (
                global_plan_world_coord_privileged[x][0],
                global_plan_world_coord_privileged[x][1],
            )
            for x in ds_ids
        ]

        self._global_plan = [global_plan[x] for x in ds_ids]
        self._global_plan_world_coord = [
            (from_carla_transform(point[0]), point[1])
            for point in global_plan_world_coord
        ]

        self._global_plan = [gps_route[x] for x in ds_ids]
        self.tick()

        return ScenarioData(
            global_plan=self._global_plan,
            global_plan_world_coord=self._global_plan_world_coord,
            global_plan_world_coord_privileged=self._global_plan_world_coord_privileged,
        )

    def start_runner_thread(self, route_file, scenario_file, route_id, timeout=60.0):
        self._route_queue = Queue()
        args: Namespace = Namespace(
            route=[route_file, scenario_file, route_id],
            sync=True,
            port=self.port,
            timeout=f"{timeout}",
            host=self.host,
            agent=None,
            debug=False,
            openscenario=None,
            repetitions=1,
            reloadWorld=True,
            trafficManagerPort=self.traffic_manager_port,
            trafficManagerSeed="0",
            waitForEgo=False,
            record="",
            outputDir=f"./output/{int(time.time())}_{uuid.uuid4().int}",
            junit=True,
            json=True,
            file=True,
            output=True,
        )

        self._runner_thread = threading.Thread(
            target=runner_loop,
            args=(
                args,
                self.carla_fps,
                self._tick_queue,
                self._ticked_event,
                self._episode_started,
                self._episode_stopped,
                self._trajectory,
                self._route_queue,
                self._episode_statistics,
            ),
        )
        self._runner_thread.start()

    def tick(self) -> bool:
        # wait for all ticks on server to be processed
        if self._runner_thread is None:
            raise RuntimeError("Scenario runner thread not started")

        if self._episode_stopped.isSet():
            return True

        # if self.traffic_type == TrafficType.NO_TRAFFIC:
        #     self._remove_all_traffic()

        self._ticked_event.clear()
        self._tick_queue.put("tick")

        ok = self._ticked_event.wait(timeout=self._tick_timeout)
        if not ok and not self._episode_stopped.isSet():
            message = "Scenario runner did not start episode in time"
            print(message)
            raise RuntimeError(message)

        self._ticked_event.clear()

        return self._episode_stopped.isSet()

    def destroy(self):
        if not self.destroyed:
            self._route_queue.put(("stop", "stop", "stop", "stop"))
            tm = carla.Client(self.host, self.port).get_trafficmanager(
                self.traffic_manager_port
            )

            tm.shut_down()
            self.destroyed = True

    def stop_episode(self) -> Dict[str, Any]:
        """
        Stop the scenario runner loop
        """
        if self._episode_stopped.isSet():
            return self._episode_statistics.get(timeout=self._episode_timeout)

        self._tick_queue.put("stop")
        ok = self._episode_stopped.wait(timeout=self._episode_timeout)
        if not ok:
            message = "Scenario runner did not stop episode in time"
            print(message)
            raise RuntimeError(message)

        return self._episode_statistics.get(timeout=self._episode_timeout)


class RouteScenarioControlled(RouteScenario):
    def __init__(
        self,
        world,
        config,
        debug_mode=False,
        criteria_enable=True,
        timeout=300,
        traffic_type: TrafficType = TrafficType.SCENARIO,
    ):
        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None

        self._update_route(world, config, debug_mode)

        ego_vehicle = self._update_ego_vehicle()

        self.list_scenarios = self._build_scenario_instances(
            world,
            ego_vehicle,
            self.sampled_scenarios_definitions,
            scenarios_per_tick=5,
            timeout=self.timeout,
            debug_mode=debug_mode,
        )

        if traffic_type != TrafficType.NO_TRAFFIC:
            self.list_scenarios.append(
                BackgroundActivity(
                    world, ego_vehicle, self.config, self.route, timeout=self.timeout
                )
            )

        super(RouteScenario, self).__init__(
            name=config.name,
            ego_vehicles=[ego_vehicle],
            config=config,
            world=world,
            debug_mode=False,
            terminate_on_failure=False,
            criteria_enable=criteria_enable,
        )


class ScenarioRunnerControlled(ScenarioRunner):
    @override
    def __init__(self, args, manager: ScenarioManager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self._args = args

        if args.timeout:
            self.client_timeout = float(args.timeout)

        print("SETTING UP NEW CLIENT")
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        self.manager = manager

        # Create signal handler for SIGINT
        self._shutdown_requested = False
        self._start_wall_time = datetime.datetime.now()

    def _run_route(self, config, traffic_type: TrafficType):
        """
        Load and run the scenario given by config
        """
        result = False
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace("_", "")
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(
                    self._args.agentConfig
                )
                config.agent = self.agent_instance
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))
        tm = self.client.get_trafficmanager(int(self._args.trafficManagerPort))
        tm.set_random_device_seed(int(self._args.trafficManagerSeed))
        if self._args.sync:
            tm.set_synchronous_mode(True)

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            scenario = RouteScenarioControlled(
                world=self.world,
                config=config,
                debug_mode=self._args.debug,
                traffic_type=traffic_type,
            )

            # if traffic_type == TrafficType.NO_TRAFFIC:
            scenario.list_scenarios = []
        except Exception as exception:  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        try:
            if self._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv("SCENARIO_RUNNER_ROOT", "./"),
                    self._args.record,
                    config.name,
                )
                self.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Provide outputs if required
            self._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(
                    self.manager.scenario.get_criteria(), recorder_name
                )

            result = True

        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result


class ScenarioManagerControlled(ScenarioManager):
    def __init__(
        self,
        tick_queue: Queue,
        ticked_event: threading.Event,
        episode_started: threading.Event,
        episode_statistics: Queue,
        trajectory: List,
        debug,
        sync,
        timeout,
    ):
        self.tick_queue = tick_queue
        self.ticked_event = ticked_event
        self.episode_started = episode_started
        self.episode_statistics = episode_statistics
        self.trajectory = trajectory

        super().__init__(debug, sync, timeout)

    @override
    def run_scenario(self):
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog = Watchdog(float(self._timeout))
        self._watchdog.start()
        self._running = True

        # clear trajectory
        self.trajectory.clear()

        # add waypoints to trajectory
        for waypoint in self.scenario_class.config.trajectory:
            self.trajectory.append(carla_location_to_dict(waypoint))

        # message that scenario has started
        self.episode_started.set()
        while self._running:
            # wait for ticks
            message = self.tick_queue.get()
            if message == "stop":
                self._running = False

            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)
                # message that tick has been processed
                self.ticked_event.set()

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        scenario_statistics = {}
        for criterion in self.scenario.get_criteria():
            scenario_statistics[criterion.name] = criterion.actual_value

        # log statistics
        self.episode_statistics.put(scenario_statistics)
        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")


def carla_location_to_dict(location: carla.Location) -> Dict[str, float]:
    return {
        "x": location.x,
        "y": location.y,
        "z": location.z,
    }


def dict_to_carla_location(location: Dict[str, float]) -> carla.Location:
    return carla.Location(x=location["x"], y=location["y"], z=location["z"])
