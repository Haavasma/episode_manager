from argparse import Namespace
from dataclasses import dataclass
import datetime
from multiprocessing.managers import ListProxy
from multiprocessing.sharedctypes import SynchronizedBase
import pathlib
import threading
import time
from typing import Any, Dict, List
from typing_extensions import override
from scenario_runner import ScenarioManager, ScenarioRunner
import numpy as np
from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.timer import GameTime

import time

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
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
    print("SETTING VALUE")
    with shared_value.get_lock():
        print("SET VALUE TO", value)
        shared_value.value = value


def stop(shared_value):
    with shared_value.get_lock():
        shared_value.value = -1


def get_value(shared_value) -> int:
    with shared_value.get_lock():
        return shared_value.value


scenario_started = False


@dataclass
class ScenarioHandler:
    """
    Should interact with Scenario runner to set up the given route, as well as tick the scenario whenever ordered to.
    """

    host: str
    port: int
    traffic_manager_port: int
    carla_fps: int = 10

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenario_file: pathlib.Path,
        route_id: str,
        reload_world: bool = True,
    ):

        timeout = 60
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
            outputDir=f"./output/{int(time.time())}",
            junit=True,
            json=True,
            file=True,
            output=True,
        )

        self.tick_value = mp.Value("i", 0)
        self.scenario_started = mp.Value("b", False)

        mp_manager = mp.Manager()
        self.trajectory = mp_manager.list([])

        def setup_and_run_scenario(
            args, carla_fps, tick_value, scenario_started, trajectory: ListProxy
        ):
            if not pathlib.Path(args.outputDir).exists():
                pathlib.Path(args.outputDir).mkdir(parents=True)

            scenario_runner = ScenarioRunner(args)

            scenario_runner.frame_rate = carla_fps

            manager = ScenarioManagerControlled(
                tick_value,
                scenario_started,
                trajectory,
                args.debug,
                args.sync,
                args.timeout,
            )

            scenario_runner.manager = manager
            scenario_runner.run()

            print("DESTROYING SCENARIO RUNNER")
            scenario_runner.destroy()
            del scenario_runner
            print("DESTROYED SCENARIO RUNNER")

        self.runner_process = mp.Process(
            target=setup_and_run_scenario,
            args=(
                args,
                self.carla_fps,
                self.tick_value,
                self.scenario_started,
                self.trajectory,
            ),
        )

        self.runner_process.start()
        client = carla.Client(self.host, self.port)
        CarlaDataProvider.set_world(client.get_world())

        start = time.time()

        print("WAITING FOR SCENARIO TO START")
        set_value(self.tick_value, 0)

        while not get_value(self.scenario_started):
            if (start + timeout) < time.time():
                stop(self.tick_value)
                raise TimeoutError("Waiting for scenario to set up timed out")

            if not self.runner_process.is_alive():
                raise RuntimeError("Scenario runner thread died")
            pass

        trajectory = [dict_to_carla_location(x) for x in self.trajectory]

        # set up global plan for the scenario (gps and world coordinates)
        gps_route, route = interpolate_trajectory(trajectory)

        ds_ids = downsample_route(route, 1)

        global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan_world_coord = [
            (from_carla_transform(point[0]), point[1])
            for point in global_plan_world_coord
        ]

        self._global_plan = [gps_route[x] for x in ds_ids]

        self._global_plan_world_coord = []
        self._global_plan = []

        return self.tick()

    def tick(self) -> ScenarioState:
        # wait for all ticks on server to be processed
        tick(self.tick_value)

        while True:
            if get_value(self.tick_value) == 0:
                break

            if not self.runner_process.is_alive():
                break

        return ScenarioState(
            global_plan=self._global_plan,
            global_plan_world_coord=self._global_plan_world_coord,
            done=not self.runner_process.is_alive(),
        )

    def stop_episode(self):
        """
        Stop the scenario runner loop
        """
        stop(self.tick_value)

        # wait for scenario runner to stop
        while self.runner_process.is_alive():
            pass


class ScenarioManagerControlled(ScenarioManager):
    def __init__(
        self,
        tick_value: SynchronizedBase,
        scenario_started: SynchronizedBase,
        trajectory: ListProxy,
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
                print("SCENARIO WAS STOPPED")
                self._running = False

        stop(self.tick_value)

        set_value(self.scenario_started, False)

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
