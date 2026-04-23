cmake_minimum_required(VERSION 3.10)
project(MotionDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(main
    src/main.cpp
    src/dataset.cpp
    src/sift.cpp
)

target_link_libraries(main ${OpenCV_LIBS})