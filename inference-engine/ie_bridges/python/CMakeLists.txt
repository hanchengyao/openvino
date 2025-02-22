# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.13)

# Set the project name
project (ie_python_api)

if(DEFINED OpenVINO_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    set(InferenceEngine_LIBRARIES IE::inference_engine)
endif()

option(ENABLE_CONDA_FOLDER "Create output folder with conda python bindings" OFF)
cmake_dependent_option(ENABLE_WHEEL "Create wheel package" OFF
    "PYTHONINTERP_FOUND;NOT CMAKE_SOURCE_DIR STREQUAL ie_python_api_SOURCE_DIR" OFF)

set(PYTHON_BRIDGE_CPACK_PATH "python")

if(UNIX)
    # cython generated files requires public visibility. Force visibility required.
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_C_VISIBILITY_PRESET default)
endif()

include (cmake/UseCython.cmake)

# Check Cython version
if(CYTHON_VERSION VERSION_LESS "0.29")
    message(FATAL_ERROR "OpenVINO Python API needs at least Cython version 0.29, found version ${CYTHON_VERSION}")
else()
    message(STATUS "Found Cython version ${CYTHON_VERSION}")
endif()

if(PYTHONLIBS_VERSION_STRING MATCHES "^([0-9]+)\.([0-9]+).*")
    set(PYTHON_VERSION python${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
else()
    message(FATAL_ERROR "Failed to extract python major.minor from ${PYTHONLIBS_VERSION_STRING}")
endif()

if(ENABLE_CONDA_FOLDER)
    set(PYTHON_COMPONENT conda_${PYTHON_VERSION})
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/Conda/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/Conda/${PYTHON_VERSION}/openvino)
    endif()
else()
    set(PYTHON_COMPONENT ${PYTHON_VERSION})
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
    endif()
endif()

function(ov_python_disable_intel_warnings target)
    if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # 1292: unknown attribute "fallthrough"
        target_compile_options(${target} PRIVATE -diag-disable=1292)
    endif()
endfunction()

set (PYTHON_BRIDGE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (src/openvino/inference_engine)
add_subdirectory (src/openvino/offline_transformations)

if (ENABLE_TESTS)
    add_subdirectory(src/openvino/test_utils)
endif()

if(ENABLE_WHEEL)
    add_subdirectory(wheel)
endif()

if(TARGET _pyngraph)
    add_dependencies(ie_api _pyngraph)
endif()

# install

ie_cpack_add_component(${PYTHON_COMPONENT})

install(FILES requirements.txt
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}
        COMPONENT ${PYTHON_COMPONENT})

install(FILES requirements.txt
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}
        COMPONENT ${PYTHON_COMPONENT})

install(PROGRAMS src/openvino/__init__.py
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}/openvino
        COMPONENT ${PYTHON_COMPONENT})

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY sample/
        DESTINATION ${IE_CPACK_IE_DIR}/samples/python
        COMPONENT python_samples)

ie_cpack(${PYTHON_COMPONENT} python_samples)
