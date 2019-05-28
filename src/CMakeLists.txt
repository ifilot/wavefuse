 #************************************************************************#
 #   This file is part of Wavefuse                                        #
 #   https://gitlab.tue.nl/ifilot/Wavefuse                                #
 #                                                                        #
 #   Author: Ivo Filot <i.a.w.filot@tue.nl>                               #
 #                                                                        #
 #   Wavefuse is free software: you can redistribute it and/or modify     #
 #   it under the terms of the GNU General Public License as published    #
 #   by the Free Software Foundation, either version 3 of the License,    #
 #   or (at your option) any later version.                               #
 #                                                                        #
 #   Wavefuse is distributed in the hope that it will be useful,          #
 #   but WITHOUT ANY WARRANTY; without even the implied warranty          #
 #   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.              #
 #   See the GNU General Public License for more details.                 #
 #                                                                        #
 #   You should have received a copy of the GNU General Public License    #
 #   along with this program.  If not, see http://www.gnu.org/licenses/.  #
 #                                                                        #
 #*************************************************************************

# set minimum cmake requirements
cmake_minimum_required(VERSION 2.8)
project (wavefuse)

# add custom directory to look for .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules )

# prepare configuration file
SET(PROGNAME "Wavefuse")
SET(VERSION_MAJOR "0")
SET(VERSION_MINOR "3")
SET(VERSION_MICRO "0")
configure_file(config.h.in ../src/config.h @ONLY)

# Enable release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# add OS specific
if(APPLE)
    add_definitions(-D_APPLE)
    SET(BOOST_INCLUDEDIR "/usr/local/include")
    SET(BOOST_LIBRARYDIR "/usr/local/lib")
else()
    SET(BOOST_INCLUDEDIR "/usr/include")
    SET(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu")
endif()

# set Boost
set (Boost_NO_SYSTEM_PATHS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (BOOST_ALL_DYN_LINK OFF)

# Include libraries
find_package(PkgConfig REQUIRED)
find_package(CUDA 9.0 REQUIRED)
find_package(Boost COMPONENTS regex REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
pkg_check_modules(GLM glm REQUIRED)
pkg_check_modules(TCLAP tclap REQUIRED)

# Set include folders
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_BINARY_DIR}
                    ${GLM_INCLUDE_DIRS}
                    ${TCLAP_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS})

# Add sources
file(GLOB SOURCES "*.cpp")
file(GLOB CUDA_FILES "src/" *.cu)

CUDA_ADD_LIBRARY(wavefuse_gpu ${CUDA_FILES})

# Set C++14
add_definitions(-std=c++17 -march=native)

# Link libraries
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=\$ORIGIN/lib")
add_executable(wavefuse ${SOURCES})
target_link_libraries(wavefuse wavefuse_gpu ${Boost_LIBRARIES})
