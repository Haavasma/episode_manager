from argparse import Namespace
from dataclasses import dataclass
import pathlib
import threading
import time
from typing import Any, List
from typing_extensions import override
from scenario_runner import ScenarioManager, ScenarioRunner
import numpy as np
from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.timer import GameTime

import time

import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenarios.route_scenario import interpolate_trajectory
from srunner.tools.route_manipulation import downsample_route

from episode_manager.models.world_state import ScenarioState


@dataclass
class Observation:
    test: np.ndarray


# Backhanded solution to injecting information
# to the scenario runner and sending messages across threads
tick_queue = 0

scenario_started = False


@dataclass
class ScenarioHandler:
    """
    Should interact with Scenario runner to set up the given route, as well as tick the scenario whenever ordered to.
    """

    host: str
    port: int
    carla_fps: int = 10

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenario_file: pathlib.Path,
        route_id: str,
    ):

        timeout = 60
        args: Namespace = Namespace(
            route=[route_file, scenario_file, route_id],
            sync=True,
            port=self.port,
            timeout=f"{timeout}",
            host=self.host,
            agent=None,
            debug=True,
            openscenario=None,
            repetitions=1,
            reloadWorld=True,
            trafficManagerPort=self.port + 1000,
            trafficManagerSeed="0",
            waitForEgo=False,
            record="",
            outputDir=f"./output/{int(time.time())}",
            junit=True,
            json=True,
            file=True,
            output=True,
        )

        # create output directory if it does not exist
        if not pathlib.Path(args.outputDir).exists():
            pathlib.Path(args.outputDir).mkdir(parents=True)

        self.scenario_runner = ScenarioRunner(args)

        self.scenario_runner.frame_rate = self.carla_fps

        self.scenario_runner.manager = ScenarioManagerControlled(
            args.debug, args.sync, args.timeout
        )

        self.runner_thread = threading.Thread(target=self.scenario_runner.run)
        self.runner_thread.start()

        start = time.time()

        print("WAITING FOR SCENARIO TO START")
        global tick_queue
        tick_queue = 0
        while not scenario_started:
            if (start + timeout) < time.time():
                # Stop runner_thread with kill signal
                tick_queue = -1
                raise TimeoutError("Waiting for scenario to set up timed out")

            if not self.runner_thread.is_alive():
                raise RuntimeError("Scenario runner thread died")
            pass

        # set up global plan for the scenario (gps and world coordinates)
        gps_route, route = interpolate_trajectory(
            self.scenario_runner.manager.scenario_class.config.trajectory
        )

        ds_ids = downsample_route(route, 1)
        self._global_plan_world_coord = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]

        return

        # return self.tick()

    def tick(self) -> ScenarioState:
        global tick_queue

        # wait for all ticks on server to be processed
        while tick_queue > 0 and self.runner_thread.is_alive():
            pass

        tick_queue += 1

        return ScenarioState(
            self._global_plan,
            self._global_plan_world_coord,
            not self.runner_thread.is_alive(),
        )

    def stop_episode(self):
        """
        Stop the scenario runner loop
        """
        global tick_queue
        tick_queue = -1

        # wait for scenario runner to stop
        while self.runner_thread.is_alive():
            pass

        if self.scenario_runner:
            self.scenario_runner.destroy()
            del self.scenario_runner
            self.scenario_runner = None


class ScenarioManagerControlled(ScenarioManager):
    @override
    def run_scenario(self):
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog = Watchdog(float(self._timeout))
        self._watchdog.start()
        self._running = True

        # message that scenario has started
        global scenario_started
        scenario_started = True

        global tick_queue

        while self._running:
            if tick_queue > 0:
                tick_queue -= 1
                timestamp = None
                world = CarlaDataProvider.get_world()
                if world:
                    snapshot = world.get_snapshot()
                    if snapshot:
                        timestamp = snapshot.timestamp
                if timestamp:
                    self._tick_scenario(timestamp)

            if tick_queue == -1:
                print("SCENARIO WAS STOPPED")
                self._running = False

        tick_queue = -1

        scenario_started = False
        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            tick_queue = -1
            print("ScenarioManager: Terminated due to failure")
