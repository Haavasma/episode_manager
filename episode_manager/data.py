from enum import Enum

# Create enum with types Evaluation and Training
class TrainingType(Enum):
    TRAINING = "training_routes"
    EVALUATION = "evaluation_routes"


TRAINING_ROUTES = ["routes_training.xml"]

EVALUATION_ROUTES = ["routes_devtest.xml"]

SCENARIOS = ["all_towns_traffic_scenarios.json"]


TRAINING_TYPE_TO_ROUTES = {
    TrainingType.TRAINING.value: TRAINING_ROUTES,
    TrainingType.EVALUATION.value: EVALUATION_ROUTES,
}
