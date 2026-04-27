cmake_minimum_required(VERSION 3.10)
project(MotionDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(main
    src/main.cpp
    src/dataset.cpp
    src/sift.cpp
    src/matching.cpp
    src/motion.cpp
    src/bbox.cpp
    src/handle_ground_truth.cpp
)

add_executable(evaluate_metrics
    src/evaluate_metrics.cpp
    src/metrics.cpp
)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(evaluate_metrics ${OpenCV_LIBS})