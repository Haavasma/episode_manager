from typing import Dict, List
import pygame
import numpy as np
import math

from episode_manager.models.world_state import WorldState


class WorldStateRenderer:
    """
    Class for rendering the WorldState to a pygame surface
    """

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = None

        self.height = 720
        self.width = 1280

        return

    def render(self, state: WorldState):
        surface = generate_pygame_surface(state)

        if self.display is None:
            self.display = pygame.display.set_mode(
                (surface.get_width(), surface.get_height())
            )
            self.height = surface.get_height()
            self.width = surface.get_width()
        elif surface.get_width() != self.width or surface.get_height() != self.height:

            pygame.quit()
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode(
                (surface.get_width(), surface.get_height())
            )
            self.height = surface.get_height()
            self.width = surface.get_width()

        self.display.blit(surface, (0, 0))
        pygame.display.flip()

        return


def generate_pygame_surface(state: WorldState) -> pygame.surface.Surface:
    """
    Generates a pygame surface frame from the WorldState sensor data
    """
    surface = create_sensor_data_surface(state)

    surface = display_text(
        surface,
        {
            "Vehicle Speed": str(state.ego_vehicle_state.speed),
            "Distance to traffic light": str(
                state.ego_vehicle_state.privileged.dist_to_traffic_light
            ),
            "distance to vehicle": str(
                state.ego_vehicle_state.privileged.dist_to_vehicle
            ),
            "distance to route": str(state.ego_vehicle_state.privileged.dist_to_route),
            "distance to pedestrian": str(
                state.ego_vehicle_state.privileged.dist_to_pedestrian
            ),
        },
    )

    height = surface.get_height()
    width = surface.get_width()

    height_limit = 900
    if surface.get_height() > height_limit:
        scale_factor = height_limit / height
        new_width = round(width * scale_factor)
        new_height = round(height * scale_factor)

        surface = pygame.transform.scale(surface, (new_width, new_height))

    return surface


def create_sensor_data_surface(state: WorldState) -> pygame.surface.Surface:
    """
    Generates a pygame surface that draws all sensor data, as well as a third person
    view of the ego vehicle
    """
    third_person_view = state.ego_vehicle_state.sensor_data.third_person_view
    surface = pygame.surfarray.make_surface(np.zeros((1280, 720, 3)))

    images: List[np.ndarray] = []

    for image in state.ego_vehicle_state.sensor_data.images:
        if image is not None and image.shape[0] > 0:
            array = image[:, :, :3]
            array = array[:, :, ::-1]

            images.append(array.swapaxes(0, 1))

    camera_pos = (0, 0)
    lidar_pos = (0, 0)
    third_person_pos = (0, 0)
    width = 0
    height = 0

    camera_surface = None
    lidar_surface = None
    third_person_surface = None

    if len(images) > 0:
        concat_image = np.concatenate(images, axis=0)
        width = concat_image.shape[0]
        height = concat_image.shape[1]
        camera_surface = pygame.surfarray.make_surface(concat_image)

    if state.ego_vehicle_state.sensor_data.lidar_data is not None:
        lidar_bev = state.ego_vehicle_state.sensor_data.lidar_data.bev
        if lidar_bev is not None and lidar_bev.shape[0] > 0:
            array = lidar_bev[0].T
            lidar_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1) * 255)

            if array.shape[1] > height:
                height = array.shape[0]
            lidar_pos = (width, 0)
            width += array.shape[1]

    if third_person_view is not None and third_person_view.shape[0] > 0:
        array = third_person_view[:, :, :3]
        array = array[:, :, ::-1]

        third_person_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        third_person_pos = (0, height)
        width = max(width, array.shape[1])
        height += array.shape[0]

    surface = pygame.Surface((width, height))

    if camera_surface is not None:
        surface.blit(camera_surface, camera_pos)
    if lidar_surface is not None:
        surface.blit(lidar_surface, lidar_pos)
    if third_person_surface is not None:
        surface.blit(third_person_surface, third_person_pos)

    return surface


def display_text(
    surface: pygame.surface.Surface, content: Dict[str, str]
) -> pygame.surface.Surface:
    font = pygame.font.SysFont("Arial", 20)

    x = 50
    y = surface.get_height() - 50

    for key, value in content.items():
        text = font.render(f"{key}: {value}", True, (255, 0, 0))
        surface.blit(text, (x, y))

        y -= 30

    return surface
