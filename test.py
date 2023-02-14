import pathlib
import time
from episode_manager.agent_handler.models.configs import LidarConfiguration
from episode_manager.data import TrainingType

from episode_manager.episode_manager import (
    Action,
    CarConfiguration,
    EpisodeManager,
    EpisodeManagerConfiguration,
)


def main():

    config = EpisodeManagerConfiguration(
        "localhost",
        2000,
        TrainingType.TRAINING,
        CarConfiguration("temp", [], LidarConfiguration(enabled=False), 10),
    )

    manager = EpisodeManager(config, None, None)

    manager.start_episode()
    for _ in range(20):
        state = manager.step(Action(3.0, 0.0, False, 0.0))
        if state.running is False:
            break

        # print("STATE: " + str(state.ego_vehicle_state.speed))
        time.sleep(1.0)
    manager.stop_episode()

    # manager.start_episode()
    # for _ in range(20):
    #     state = manager.step(Action(1.0, 0.0, False, 0.0))
    #     if state.running is False:
    #         break
    #
    #     time.sleep(1.0)
    # manager.stop_episode()

    return


if __name__ == "__main__":

    main()
