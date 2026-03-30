cmake_minimum_required(VERSION 3.9)
project(libiq)
include(GNUInstallDirs)

set(CPACK_PACKAGE_DESCRIPTION "IQIF neuron")
set(CPACK_PACKAGE_DESCRIPTIONSUMMARY
    "This library creates IQIF/Izhikevich/LIF networks and runs them")
set(CPACK_PACKAGE_VENDOR "CCLo Lab")
set(CPACK_PACKAGE_CONTACT "chen_fu_yeh@lolab-nthu.org")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "3")
set(CPACK_PACKAGE_VERSION_PATCH "0")

set(PROJECT_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -DVDEBUG")

set(CMAKE_CXX_STANDARD 11)

add_library(iq-network SHARED "src/iq_network.cpp" "src/iq_neuron.cpp" "src/weight_index_list.cpp")
add_library(iz-network SHARED "src/iz_network.cpp" "src/iz_neuron.cpp" "src/weight_index_list.cpp")
add_library(lif-network SHARED "src/lif_network.cpp" "src/lif_neuron.cpp" "src/weight_index_list.cpp")

# Versioned sonames for system installs; plain .so for pip wheels
if(NOT DEFINED SKBUILD)
    set_target_properties(iq-network PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(iz-network PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(lif-network PROPERTIES VERSION ${PROJECT_VERSION})
endif()

# Bring the headers into the project
include_directories(include)

# macOS: help cmake find Homebrew libomp
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH_USE_LINK_RPATH ON)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(iq-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(iz-network PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lif-network PRIVATE OpenMP::OpenMP_CXX)

# --- Install rules ---
if(DEFINED SKBUILD)
    # pip install: place .so/.dylib/.dll inside the iqif package
    install(TARGETS iq-network iz-network lif-network
            LIBRARY DESTINATION iqif
            RUNTIME DESTINATION iqif)
else()
    # Standalone cmake install: system lib directory
    install(TARGETS iq-network DESTINATION lib)
    install(TARGETS iz-network DESTINATION lib)
    install(TARGETS lif-network DESTINATION lib)
endif()

# --- Standalone binary (only when not building via pip) ---
if(NOT DEFINED SKBUILD)
    file(GLOB SOURCES "src/*.cpp")
    add_executable(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
