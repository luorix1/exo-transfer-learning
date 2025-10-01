from setuptools import setup

setup(
    name="opensim",
    version="0.0.0.dev0",
    description="Local OpenSim Python bindings (prebuilt)",
    packages=["opensim"],
    package_dir={"opensim": "."},
    include_package_data=True,
    package_data={
        "opensim": [
            "*.so",
            "*.py",
        ]
    },
    python_requires=">=3.12,<3.13",
)


