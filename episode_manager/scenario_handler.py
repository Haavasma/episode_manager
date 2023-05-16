from argparse import Namespace
from dataclasses import dataclass, field
import datetime
from multiprocessing.sharedctypes import SynchronizedBase
import os
import pathlib
from queue import Queue
import sys
import threading
import uuid
import time
from typing import Any, Dict, List, Optional
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
        route_file, scenario_file, route_id = route_queue.get()
        if route_file == "stop":
            break
        scenario_runner.finished = False

        route_configs = RouteParser.parse_routes_file(
            route_file, scenario_file, route_id
        )

        assert len(route_configs) == 1
        scenario_runner._load_and_run_scenario(route_configs[0])
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
    ) -> ScenarioData:
        self._episode_started.clear()
        self._episode_stopped.clear()

        if self._runner_thread is None:
            self.start_runner_thread(
                route_file, scenario_file, route_id, self._episode_timeout
            )

        self._route_queue.put((route_file, scenario_file, route_id))
        print("WAITING FOR EPISODE TO START")
        ok = self._episode_started.wait(timeout=self._episode_timeout)
        if not ok:
            print("Scenario runner did not start episode in time")
            raise RuntimeError("Scenario runner did not start episode in time")
        print("EPISODE STARTED")

        trajectory = [dict_to_carla_location(x) for x in self._trajectory]
        gps_route, route = interpolate_trajectory(trajectory)
        ds_ids_hack = downsample_route(route, 1)

        print("FOUND TRAJECTORY")

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
        print("TICKING SCENARIO")
        self.tick()

        print("RETURNING SCENARIO DATA")
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
            self._route_queue.put(("stop", "stop", "stop"))
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
