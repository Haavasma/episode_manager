import time
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
from episode_manager.data import TrainingType

from episode_manager.episode_manager import (
    Action,
    EpisodeManager,
    EpisodeManagerConfiguration,
)


def main():
    config = EpisodeManagerConfiguration()

    config.car_config.carla_fps = 10

    config.render_client = True
    config.render_server = True

    manager = EpisodeManager(config, None, None)

    for _ in range(50):
        state = manager.start_episode()
        for _ in range(300):
            action = Action(1.0, 0.0, False, 0.0)

            if (
                state.ego_vehicle_state.privileged.dist_to_vehicle < 10
                and state.ego_vehicle_state.privileged.dist_to_vehicle >= 0.0
            ):
                action = Action(0.0, 1.0, False, 0.0)

            state = manager.step(action)
            if state.scenario_state.done:
                print("DONE")
                break

            time.sleep(0.01)

        manager.stop_episode()

    print("ENDED EPISODES")

    return


if __name__ == "__main__":

    main()
