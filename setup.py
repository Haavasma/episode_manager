from setuptools import setup, find_packages


setup(
    packages=[
        package for package in find_packages() if package.startswith("episode_manager")
    ],
    name="episode_manager",
    version="0.9.13",
    py_modules=["episode_manager"],
    package_data={
        "episode_manager": [
            "routes/training_routes/*.xml",
            "routes/validation_routes/*.xml",
            "routes/evaluation_routes/*.xml",
            "routes/scenarios/*.json",
        ]
    },
    install_requires=[
        "scenario_runner @ git+https://github.com/Haavasma/scenario_runner.git@v0.9.13-setup-script",
        "typing-extensions>=4.0.0",
        "pygame>=2.0.0",
    ],
)


# Set up the scenario runner from carla and install the package including scenario_runner.py and the srunner pacakage
