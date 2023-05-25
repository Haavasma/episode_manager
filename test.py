import random
import time

import carla
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
from episode_manager.data import TrafficType, TrainingType
import queue

from episode_manager.episode_manager import (
    Action,
    EpisodeManager,
    EpisodeManagerConfiguration,
)


def main():
    config = EpisodeManagerConfiguration()

    config.car_config.carla_fps = 10

    config.render_client = False
    config.render_server = False

    # config.car_config.cameras = []
    config.car_config.lidar["enabled"] = False
    config.training_type = TrainingType.EVALUATION

    # for camera in config.car_config.cameras:
    #     camera["width"] = 100
    #     camera["height"] = 100

    manager = EpisodeManager(config)

    fpses = queue.Queue()

    for _ in range(10):
        fpses.put(0)

    for i in range(1):
        traffic_type = TrafficType.TRAFFIC
        # if i % 3 == 0:
        #     traffic_type = TrafficType.SCENARIO
        # elif i % 3 == 1:
        #     traffic_type = TrafficType.SCENARIO

        state, _ = manager.start_episode(
            traffic_type=traffic_type,
        )
        for j in range(100):
            start = time.time()
            print("\n")

            action = Action(1.0, 0.0, False, 0.0)
            #
            # if (
            #     state.ego_vehicle_state.privileged.dist_to_vehicle < 10
            #     and state.ego_vehicle_state.privileged.dist_to_vehicle >= 0.0
            # ):
            # action = Action(0.0, 1.0, False, 0.0)
            #
            if random.random() < 0.2:
                action = Action(0.0, 1.0, False, 0.0)

            # EPISODE 1, STEP 418
            print(f"EPISODE: {i}, STEP: {j}")
            if state.ego_vehicle_state.speed > 5.0:
                action = Action(0.0, 0.0, False, 0.0)

            state = manager.step(action)
            if state.done:
                print("DONE")
                break

            fpses.put(1 / (time.time() - start))
            fpses.get()

            print(f"FPS: {sum(fpses.queue) / fpses.qsize()}")

        _ = manager.stop_episode()

        # print("STATISTICS: ", statistics)

    print("ENDED EPISODES")

    manager.close()

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
