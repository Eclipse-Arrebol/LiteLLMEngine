cmake_minimum_required(VERSION 3.18)

project(LiteLLMEngine LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
    ${CMAKE_SOURCE_DIR}/cuda/*.cu
)

list(FILTER ENGINE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_library(litellm_engine_core STATIC
    ${ENGINE_SOURCES}
)

target_include_directories(litellm_engine_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(litellm_engine_core PUBLIC
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(litellm_engine_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_executable(LiteLLMEngine
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(LiteLLMEngine PRIVATE
    litellm_engine_core
)

set_target_properties(LiteLLMEngine PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

enable_testing()

add_executable(test_tensor
    ${CMAKE_SOURCE_DIR}/tests/test_tensor.cpp
)
target_link_libraries(test_tensor PRIVATE
    litellm_engine_core
)
add_test(NAME test_tensor COMMAND test_tensor)


add_executable(test_add
    ${CMAKE_SOURCE_DIR}/tests/test_add.cpp
)
target_link_libraries(test_add PRIVATE
    litellm_engine_core
)
add_test(NAME test_add COMMAND test_add)


add_executable(test_layer_skeleton
    ${CMAKE_SOURCE_DIR}/tests/test_layer_skeleton.cpp
)
target_link_libraries(test_layer_skeleton PRIVATE
    litellm_engine_core
)
add_test(NAME test_layer_skeleton COMMAND test_layer_skeleton)