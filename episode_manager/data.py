from enum import Enum
import enum


# Create enum with types Evaluation and Training
class TrainingType(Enum):
    TRAINING = "training_routes"
    EVALUATION = "evaluation_routes"


class TrafficType(Enum):
    NO_TRAFFIC = enum.auto()
    TRAFFIC = enum.auto()
    SCENARIO = enum.auto()


TRAINING_ROUTES = ["routes_training.xml"]

EVALUATION_ROUTES = ["routes_devtest.xml"]

SCENARIOS = ["all_towns_traffic_scenarios.json", "no_scenarios.json"]


TRAINING_TYPE_TO_ROUTES = {
    TrainingType.TRAINING.value: TRAINING_ROUTES,
    TrainingType.EVALUATION.value: EVALUATION_ROUTES,
}
