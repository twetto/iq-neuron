cmake_minimum_required(VERSION 3.9)
project(libiq)
include(GNUInstallDirs)

set(CPACK_PACKAGE_DESCRIPTION "IQIF neuron")
set(CPACK_PACKAGE_DESCRIPTIONSUMMARY
    "This library creates IQIF/Izhikevich/LIF networks and runs them")
set(CPACK_PACKAGE_VENDOR "CCLo Lab")
set(CPACK_PACKAGE_CONTACT "chen_fu_yeh@lolab-nthu.org")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "1")

set(PROJECT_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -DVDEBUG")

set(CMAKE_CXX_STANDARD 11)

add_library(iq-network SHARED "src/iq_network.cpp" "src/iq_neuron.cpp" "src/weight_index_list.cpp")
set_target_properties(iq-network PROPERTIES VERSION ${PROJECT_VERSION})
add_library(iz-network SHARED "src/iz_network.cpp" "src/iz_neuron.cpp" "src/weight_index_list.cpp")
set_target_properties(iz-network PROPERTIES VERSION ${PROJECT_VERSION})
add_library(lif-network SHARED "src/lif_network.cpp" "src/lif_neuron.cpp" "src/weight_index_list.cpp")
set_target_properties(lif-network PROPERTIES VERSION ${PROJECT_VERSION})

# Bring the headers into the project
include_directories(include)

# wildcard additions
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(lif-network PROPERTIES VERSION ${PROJECT_VERSION})
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(iq-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(iz-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lif-network PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS iq-network DESTINATION lib)
install(TARGETS iz-network DESTINATION lib)
install(TARGETS lif-network DESTINATION lib)

