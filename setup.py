from glob import glob

from setuptools import setup

package_name = "autoware_ml_model_launchers"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (
            f"share/{package_name}",
            ["package.xml", "README.md", "requirements-open-yolo.txt"],
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"lib/{package_name}", ["scripts/check_environment"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="YoshiRi",
    maintainer_email="yoshiri@example.com",
    description="Launchers and small helper nodes for Autoware ML model validation.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "compressed_yolo_node = autoware_ml_model_launchers.compressed_yolo_node:main",
            "tlr_yolox_roi_adapter = autoware_ml_model_launchers.tlr_yolox_roi_adapter:main",
        ],
    },
)
