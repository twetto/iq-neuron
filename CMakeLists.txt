cmake_minimum_required(VERSION 3.9)
project(IQIF)
#project(libiq)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")

# Bring the headers into the project
include_directories(include)

# wildcard additions
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

add_library(iq-network SHARED "src/iq_network.cpp" "src/iq_neuron.cpp" "src/weight_index_list.cpp")
add_library(iz-network SHARED "src/iz_network.cpp" "src/iz_neuron.cpp" "src/weight_index_list.cpp")
add_library(lif-network SHARED "src/lif_network.cpp" "src/lif_neuron.cpp" "src/weight_index_list.cpp")
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(iq-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(iz-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lif-network PRIVATE OpenMP::OpenMP_CXX)
#add_library(iq-network SHARED "src/iq_network.cpp" "src/iq_neuron.cpp" "src/weight_index_list.cpp")
#add_library(iz-network SHARED "src/iz_network.cpp" "src/iz_neuron.cpp")

