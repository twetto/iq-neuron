cmake_minimum_required(VERSION 2.8.9)
project(iq-neuron)
set(CMAKE_CXX_STANDARD 11)

# Bring the headers into the project
include_directories(include)

# wildcard additions
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

#find_package(OpenMP REQUIRED)
#target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})

