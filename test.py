import pathlib
import time
from episode_manager.data import TrainingType

from episode_manager.episode_manager import (
    Action,
    CarConfiguration,
    EpisodeManager,
    EpisodeManagerConfiguration,
    setup_agent_handler,
)


def main():

    config = EpisodeManagerConfiguration(
        "localhost",
        2000,
        TrainingType.TRAINING,
        pathlib.Path("../routes"),
        CarConfiguration("temp", [], []),
    )

    manager = EpisodeManager(config, setup_agent_handler(config))

    manager.start_episode()
    for _ in range(20):
        state = manager.step(Action(1.0, 0.0, False, 0.0))
        if state.running is False:
            break

        print("STATE: " + str(state))
        time.sleep(1.0)
    manager.stop_episode()

    print("BREAK BEFORE STARTING NEXT EPISODE")
    time.sleep(10.0)

    manager.start_episode()
    for _ in range(20):
        state = manager.step(Action(1.0, 0.0, False, 0.0))
        if state.running is False:
            break

        print("STATE: " + str(state))
        time.sleep(1.0)
    manager.stop_episode()

    return


if __name__ == "__main__":

    main()
