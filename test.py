import random
import time

import carla
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
from episode_manager.data import TrainingType
import queue

from episode_manager.episode_manager import (
    Action,
    EpisodeManager,
    EpisodeManagerConfiguration,
)


def main():
    config = EpisodeManagerConfiguration()

    config.car_config.carla_fps = 10

    config.render_client = True
    config.render_server = False

    config.car_config.lidar["enabled"] = False

    # for camera in config.car_config.cameras:
    #     camera["width"] = 100
    #     camera["height"] = 100

    manager = EpisodeManager(config, None, None)

    fpses = queue.Queue()

    for _ in range(10):
        fpses.put(0)

    for i in range(50):
        state = manager.start_episode()
        for j in range(500):
            start = time.time()
            print("\n")

            # action = Action(1.0, 0.0, False, 0.0)
            #
            # if (
            #     state.ego_vehicle_state.privileged.dist_to_vehicle < 10
            #     and state.ego_vehicle_state.privileged.dist_to_vehicle >= 0.0
            # ):
            action = Action(0.0, 1.0, False, 0.0)

            if random.random() < 0.5:
                action = Action(1.0, 0.0, False, 0.0)

            # EPISODE 1, STEP 418
            print(f"EPISODE: {i}, STEP: {j}")
            state = manager.step(action)
            if state.scenario_state.done:
                print("DONE")
                break

            fpses.put(1 / (time.time() - start))
            fpses.get()

            print(f"FPS: {sum(fpses.queue) / fpses.qsize()}")

        manager.stop_episode()

    print("ENDED EPISODES")

    return


def test_server():
    client = carla.Client("localhost", 2003)

    world = client.get_world()

    # set to synchronous mode

    settings = world.get_settings()

    settings.fixed_delta_seconds = 1 / 10
    settings.synchronous_mode = True

    world.apply_settings(settings)

    for i in range(100000):
        print(f"TICKING {i}")
        world.tick()


if __name__ == "__main__":
    # test_server()
    main()
