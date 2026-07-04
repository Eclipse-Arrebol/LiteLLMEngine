cmake_minimum_required(VERSION 3.18)

project(LiteLLMEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB_RECURSE CPP_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/cuda/*.cu
)

add_executable(LiteLLMEngine
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(LiteLLMEngine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(LiteLLMEngine PRIVATE
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(LiteLLMEngine PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)