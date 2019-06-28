cmake_minimum_required(VERSION 2.8.9)
project(speed_prediction_theta)
SET(CMAKE_CXX_FLAGS "-std=c++11")

# Bring the headers into the project
include_directories(include)

# wildcard additions
file(GLOB SOURCES "src/*.cpp")

add_executable(speed_prediction_theta ${SOURCES})

