from dataclasses import dataclass
import carla


@dataclass
class Location:
    x: float
    y: float
    z: float

    def get_carla_location(self):
        return carla.Location(x=self.x, y=self.y, z=self.z)


@dataclass
class Rotation:
    pitch: float
    yaw: float
    roll: float

    def get_carla_rotation(self):
        return carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)


@dataclass
class Transform:
    location: Location
    rotation: Rotation

    def __post_init__(self):
        self.carla_transform = carla.Transform(
            self.location.get_carla_location(), self.rotation.get_carla_rotation()
        )

    def get_carla_transform(self) -> carla.Transform:
        """
        Returns a carla Transform with the data of the current Transform
        """
        return self.carla_transform
