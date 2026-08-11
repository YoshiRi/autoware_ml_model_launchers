cmake_minimum_required(VERSION 3.14)
project(autoware_ml_model_launchers)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    autoware_ml_model_launchers/compressed_yolo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME compressed_yolo_node
)

install(
  PROGRAMS
    scripts/open_detector_image
    scripts/open_detector_node
    scripts/open_detector_smoke
    scripts/reusable_bbox_tracker_node
    scripts/record_topics
    scripts/record_session
    scripts/record_video
    scripts/launcher_dashboard_ui
    scripts/param_snapshot_ui
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    autoware_ml_model_launchers/tlr_yolox_roi_adapter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tlr_yolox_roi_adapter
)

install(
  PROGRAMS
    scripts/check_environment
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs
)

install(
  FILES
    README.md
    requirements-open-detector-common.txt
    requirements-open-detector-dfine.txt
    requirements-open-detector-rfdetr.txt
    requirements-open-tracker-bbox.txt
    requirements-open-yolo.txt
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_flake8(
    CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
    autoware_ml_model_launchers
    setup.py
    test
  )
  ament_pep257(autoware_ml_model_launchers)
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  ament_add_pytest_test(
    open_detector_tests
    test
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  unset(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD})
endif()

ament_package()
