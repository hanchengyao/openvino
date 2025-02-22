# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# RPATH is always enabled for unit tests
set(CMAKE_SKIP_RPATH OFF)

# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

add_subdirectory(inference_engine)

if (ENABLE_MKL_DNN)
    add_subdirectory(cpu)
endif ()

if (ENABLE_GNA)
    add_subdirectory(gna)
endif ()

if (ENABLE_MYRIAD)
    add_subdirectory(vpu)
endif ()

if(NGRAPH_ONNX_FRONTEND_ENABLE)
    add_subdirectory(frontends/onnx_import)
endif()
