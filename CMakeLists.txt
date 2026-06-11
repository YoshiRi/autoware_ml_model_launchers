cmake_minimum_required(VERSION 3.14)
project(autoware_ml_model_launchers)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

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
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
