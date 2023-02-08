from argparse import Namespace
from dataclasses import dataclass
import pathlib
import threading
import time
from typing import List
from typing_extensions import override
from scenario_runner import ScenarioManager, ScenarioRunner
import numpy as np
from srunner.scenariomanager.timer import GameTime

import time

import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog


@dataclass
class Observation:
    test: np.ndarray


# Backhanded solution to injecting information
# to the scenario runner and sending messages across threads
tick_queue = 0


# Backhanded solution to injecting information to the scenario runner
scenario_started = False


@dataclass
class ScenarioState:
    dist_to_traffic_light: float
    dist_to_vehicle: float
    dist_to_pedestrian: float
    dist_to_route: float


class ScenarioHandler:
    """
    Should interact with Scenario runner to set up the given route, as well as tick the scenario whenever ordered to.
    """

    def __init__(self):
        return

    def start_episode(
        self,
        route_file: pathlib.Path,
        scenarios_file: pathlib.Path,
        route_id: "str",
        port: int = 2000,
    ):
        args: Namespace = Namespace(
            route=[route_file, scenarios_file, route_id],
            sync=True,
            port=port,
            timeout="10.0",
            host="127.0.0.1",
            agent=None,
            debug=True,
            openscenario=None,
            repetitions=1,
            reloadWorld=True,
            trafficManagerPort=8000,
            trafficManagerSeed="0",
            waitForEgo=False,
            record="",
            outputDir=f"../output/{int(time.time())}",
            junit=True,
            json=True,
            file=True,
            output=True,
        )

        # create output directory if it does not exist
        if not pathlib.Path(args.outputDir).exists():
            pathlib.Path(args.outputDir).mkdir(parents=True)

        self.scenario_runner = ScenarioRunner(args)
        self.scenario_runner.manager = ScenarioManagerControlled(
            args.debug, args.sync, args.timeout
        )

        self.runner_thread = threading.Thread(target=self.scenario_runner.run)
        self.runner_thread.start()

        timeout = 60 * 1e9  # Timeout of 60 seconds
        start = time.time()

        while not scenario_started:
            if (start + timeout) < time.time():
                raise TimeoutError("Waiting for scenario to set up timed out")

            if not self.runner_thread.is_alive():
                raise RuntimeError("Scenario runner thread died")
            pass

    def is_running(self):
        return self.runner_thread.is_alive()

    def tick(self):
        global tick_queue
        tick_queue += 1

    def stop_episode(self):
        """ """
        global tick_queue
        tick_queue = -1


class ScenarioManagerControlled(ScenarioManager):
    @override
    def run_scenario(self):
        print("RUNNING OVERRIDEN RUN_SCENARIO")
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

        scenario_started = False
        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")
