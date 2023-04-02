from argparse import Namespace
from dataclasses import dataclass, field
import datetime
from multiprocessing.sharedctypes import SynchronizedBase
import pathlib
from queue import Queue
import threading
import uuid
import time
from typing import Dict, List, Optional
from typing_extensions import override
from scenario_runner import RouteParser, ScenarioManager, ScenarioRunner
from srunner.scenariomanager.timer import GameTime

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenarios.route_scenario import interpolate_trajectory
from srunner.tools.route_manipulation import downsample_route
from episode_manager.agent_handler.models.transform import from_carla_transform

from episode_manager.models.world_state import ScenarioState

import multiprocessing as mp


# Backhanded solution to injecting information
# to the scenario runner and sending messages across threads


def tick(shared_value):
    with shared_value.get_lock():
        shared_value.value += 1


def decrease(shared_value):
    with shared_value.get_lock():
        shared_value.value -= 1


def set_value(shared_value, value):
    with shared_value.get_lock():
        shared_value.value = value


def stop(shared_value):
    with shared_value.get_lock():
        shared_value.value = -1


def get_value(shared_value) -> int:
    with shared_value.get_lock():
        return shared_value.value


def runner_loop(
    args,
    carla_fps: int,
    tick_value: SynchronizedBase,
    scenario_started: SynchronizedBase,
    trajectory: List,
    route_queue: Queue,
):
    if not pathlib.Path(args.outputDir).exists():
        pathlib.Path(args.outputDir).mkdir(parents=True)

    manager = ScenarioManagerControlled(
        tick_value,
        scenario_started,
        trajectory,
        args.debug,
        args.sync,
        args.timeout,
    )

    scenario_runner = ScenarioRunnerControlled(args, manager)

    scenario_runner.frame_rate = carla_fps

    scenario_runner.manager = manager

    while True:
        route_file, scenario_file, route_id = route_queue.get()
        if route_file == "stop":
            break

        route_configs = RouteParser.parse_routes_file(
            route_file, scenario_file, route_id
        )

        assert len(route_configs) == 1

        try:
            scenario_runner._load_and_run_scenario(route_configs[0])
        except Exception as e:
            print("SOMETHING WENT WRONG DURING SCENARIO RUN, EXITING LOOP")
            print(e)
            return
        scenario_runner._cleanup()


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
    _tick_value = mp.Value("i", 0)
    _scenario_started = mp.Value("b", False)
    _trajectory: List = field(default_factory=lambda: [])

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenario_file: pathlib.Path,
        route_id: str,
        reload_world: bool = True,
    ):
        timeout = 30
        if self._runner_thread is None:
            self.start_runner_thread(route_file, scenario_file, route_id)

        self._route_queue.put((route_file, scenario_file, route_id))

        start = time.time()

        set_value(self._tick_value, 0)
        set_value(self._scenario_started, False)

        tries = 2
        current_try = 0

        while not get_value(self._scenario_started) and current_try < tries:
            if (start + timeout) < time.time():
                stop(self._tick_value)
                self._route_queue.put(("stop", None, None))
                self.start_runner_thread(route_file, scenario_file, route_id)
                current_try += 1
                print("Waiting for scenario to set up timed out, TRYING AGAIN")
                self._route_queue.put((route_file, scenario_file, route_id))
                start = time.time()

            if self._runner_thread is not None and not self._runner_thread.is_alive():
                raise RuntimeError("Scenario runner thread died")
            pass

        if current_try >= tries:
            raise RuntimeError("Scenario runner thread did not start")

        trajectory = [dict_to_carla_location(x) for x in self._trajectory]

        # set up global plan for the scenario (gps and world coordinates)
        gps_route, route = interpolate_trajectory(trajectory)

        ds_ids = downsample_route(route, 1)

        global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan_world_coord = [
            (from_carla_transform(point[0]), point[1])
            for point in global_plan_world_coord
        ]

        self._global_plan = [gps_route[x] for x in ds_ids]

        return self.tick()

    def start_runner_thread(self, route_file, scenario_file, route_id, timeout=60):
        self._scenario_started = mp.Value("b", False)
        self._tick_value = mp.Value("i", 0)
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
                self._tick_value,
                self._scenario_started,
                self._trajectory,
                self._route_queue,
            ),
        )
        self._runner_thread.start()

    def tick(self) -> ScenarioState:
        # wait for all ticks on server to be processed
        if self._runner_thread is None:
            raise RuntimeError("Scenario runner thread not started")

        if not get_value(self._scenario_started):
            return ScenarioState(
                global_plan=self._global_plan,
                global_plan_world_coord=self._global_plan_world_coord,
                done=True,
            )

        tick(self._tick_value)
        while True:
            if get_value(self._tick_value) == 0:
                break

            if not get_value(self._scenario_started):
                break

        return ScenarioState(
            global_plan=self._global_plan,
            global_plan_world_coord=self._global_plan_world_coord,
            done=not get_value(self._scenario_started),
        )

    def stop_episode(self):
        """
        Stop the scenario runner loop
        """
        stop(self._tick_value)

        # wait for scenario runner to stop
        while get_value(self._scenario_started):
            pass


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


class ScenarioManagerControlled(ScenarioManager):
    def __init__(
        self,
        tick_value: SynchronizedBase,
        scenario_started: SynchronizedBase,
        trajectory: List,
        debug,
        sync,
        timeout,
    ):
        self.tick_value = tick_value
        self.scenario_started = scenario_started
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

        for waypoint in self.scenario_class.config.trajectory:
            self.trajectory.append(carla_location_to_dict(waypoint))

        # message that scenario has started
        set_value(self.scenario_started, True)

        while self._running:
            if get_value(self.tick_value) > 0:
                decrease(self.tick_value)
                timestamp = None
                world = CarlaDataProvider.get_world()
                if world:
                    snapshot = world.get_snapshot()
                    if snapshot:
                        timestamp = snapshot.timestamp
                if timestamp:
                    self._tick_scenario(timestamp)

            if get_value(self.tick_value) == -1:
                self._running = False

        set_value(self.scenario_started, False)
        stop(self.tick_value)

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            stop(self.tick_value)

            print("ScenarioManager: Terminated due to failure")


def carla_location_to_dict(location: carla.Location) -> Dict[str, float]:
    return {
        "x": location.x,
        "y": location.y,
        "z": location.z,
    }


def dict_to_carla_location(location: Dict[str, float]) -> carla.Location:
    return carla.Location(x=location["x"], y=location["y"], z=location["z"])
