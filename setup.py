from setuptools import setup, find_packages


setup(
    packages=[
        package
        for package in find_packages()
        if package.startswith("episode_manager") or package.startswith("carla_server")
    ],
    name="episode_manager",
    version="0.9.13",
    py_modules=["episode_manager", "carla_server"],
    package_data={
        "episode_manager": [
            "routes/*.xml",
            "routes/*.json",
            "Makefile",
            "carla_server.def",
        ]
    },
    install_requires=[
        "scenario_runner @ git+https://github.com/Haavasma/scenario_runner.git@v0.9.13-setup-script",
        "typing-extensions>=4.0.0",
        "pygame>=2.0.0",
        "nvidia-ml-py3>=7.352.0",
    ],
)


# Set up the scenario runner from carla and install the package including scenario_runner.py and the srunner pacakage
