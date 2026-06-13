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
    requirements-open-yolo.txt
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)

  ament_flake8(
    CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
    autoware_ml_model_launchers
    setup.py
  )
  ament_pep257(autoware_ml_model_launchers)
endif()

ament_package()
