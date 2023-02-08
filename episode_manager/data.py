from enum import Enum

# Create enum with types Evaluation and Training
class TrainingType(Enum):
    TRAINING = "training_routes"
    VALIDATION = "validation_routes"
    EVALUATION = "evaluation_routes"


TRAINING_ROUTES = [
    ["routes_town01_long.xml", "town01_all_scenarios.json"],
    ["routes_town01_short.xml", "town01_all_scenarios.json"],
    ["routes_town01_tiny.xml", "town01_all_scenarios.json"],
    ["routes_town02_long.xml", "town02_all_scenarios.json"],
    ["routes_town02_short.xml", "town02_all_scenarios.json"],
    ["routes_town02_tiny.xml", "town02_all_scenarios.json"],
    ["routes_town03_long.xml", "town03_all_scenarios.json"],
    ["routes_town03_short.xml", "town03_all_scenarios.json"],
    ["routes_town03_tiny.xml", "town03_all_scenarios.json"],
    ["routes_town04_long.xml", "town04_all_scenarios.json"],
    ["routes_town04_short.xml", "town04_all_scenarios.json"],
    ["routes_town04_tiny.xml", "town04_all_scenarios.json"],
    ["routes_town05_long.xml", "town05_all_scenarios.json"],
    ["routes_town05_short.xml", "town05_all_scenarios.json"],
    ["routes_town05_tiny.xml", "town05_all_scenarios.json"],
    ["routes_town06_long.xml", "town06_all_scenarios.json"],
    ["routes_town06_short.xml", "town06_all_scenarios.json"],
    ["routes_town06_tiny.xml", "town06_all_scenarios.json"],
    ["routes_town07_short.xml", "town07_all_scenarios.json"],
    ["routes_town07_tiny.xml", "town07_all_scenarios.json"],
    ["routes_town10_short.xml", "town10_all_scenarios.json"],
    ["routes_town10_tiny.xml", "town10_all_scenarios.json"],
]

VALIDATION_ROUTES = [
    ["routes_town05_short.xml", "town05_all_scenarios.json"],
    ["routes_town05_tiny.xml", "town05_all_scenarios.json"],
]

EVALUATION_ROUTES = [["routes_town05_long.xml", "town05_all_scenarios.json"]]


TRAINING_TYPE_TO_ROUTES = {
    TrainingType.TRAINING.value: TRAINING_ROUTES,
    TrainingType.VALIDATION.value: VALIDATION_ROUTES,
    TrainingType.EVALUATION.value: EVALUATION_ROUTES,
}
