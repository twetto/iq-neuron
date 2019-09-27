cmake_minimum_required(VERSION 2.8.9)
project(IQIF)
#project(libiq)
set(CMAKE_CXX_STANDARD 11)

# Bring the headers into the project
include_directories(include)

# wildcard additions
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

#find_package(OpenMP REQUIRED)
#target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
#add_library(iq-network SHARED "src/iq_network.cpp" "src/iq_neuron.cpp")
#add_library(iz-network SHARED "src/iz_network.cpp" "src/iz_neuron.cpp")

