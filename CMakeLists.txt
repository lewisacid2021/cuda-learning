cmake_minimum_required(VERSION 3.18)
project(CUDASamples LANGUAGES CUDA CXX)

# CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 输出目录配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# GPU 架构：RTX 3090 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# 遍历 src 下的每个子目录
file(GLOB SAMPLE_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/*)

foreach(DIR ${SAMPLE_DIRS})
    # 查找该目录下的 .cu 文件（通常只有一个）
    file(GLOB CU_FILES ${CMAKE_SOURCE_DIR}/src/${DIR}/*.cu)

    if(CU_FILES)
        # 可执行文件名称 = 子目录名称
        add_executable(${DIR} ${CU_FILES})
        set_target_properties(${DIR} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        message(STATUS "Added CUDA sample: ${DIR}")
    endif()
endforeach()
