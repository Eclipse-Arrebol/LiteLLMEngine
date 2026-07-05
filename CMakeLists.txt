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


add_executable(test_model_skeleton
    ${CMAKE_SOURCE_DIR}/tests/test_model_skeleton.cpp
)
target_link_libraries(test_model_skeleton PRIVATE
    litellm_engine_core
)
add_test(NAME test_model_skeleton COMMAND test_model_skeleton)


add_executable(test_weight_loading_skeleton
    ${CMAKE_SOURCE_DIR}/tests/test_weight_loading_skeleton.cpp
)
target_link_libraries(test_weight_loading_skeleton PRIVATE
    litellm_engine_core
)
add_test(NAME test_weight_loading_skeleton COMMAND test_weight_loading_skeleton)


add_executable(test_weight_loader
    ${CMAKE_SOURCE_DIR}/tests/test_weight_loader.cpp
)
target_link_libraries(test_weight_loader PRIVATE
    litellm_engine_core
)
add_test(NAME test_weight_loader COMMAND test_weight_loader)


add_executable(test_real_qwen3_weight_loading
    ${CMAKE_SOURCE_DIR}/tests/test_real_qwen3_weight_loading.cpp
)
target_link_libraries(test_real_qwen3_weight_loading PRIVATE
    litellm_engine_core
)
add_test(NAME test_real_qwen3_weight_loading COMMAND test_real_qwen3_weight_loading)


add_executable(test_embedding tests/test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE litellm_engine_core)
add_test(NAME test_embedding COMMAND test_embedding)


add_executable(test_rms_norm tests/test_rms_norm.cpp)
target_link_libraries(test_rms_norm PRIVATE litellm_engine_core)
add_test(NAME test_rms_norm COMMAND test_rms_norm)


add_executable(test_linear tests/test_linear.cpp)
target_link_libraries(test_linear PRIVATE litellm_engine_core)
add_test(NAME test_linear COMMAND test_linear)


add_executable(test_silu_and_mul tests/test_silu_and_mul.cpp)
target_link_libraries(test_silu_and_mul PRIVATE litellm_engine_core)
add_test(NAME test_silu_and_mul COMMAND test_silu_and_mul)


add_executable(test_qwen3_mlp tests/test_qwen3_mlp.cpp)
target_link_libraries(test_qwen3_mlp PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_mlp COMMAND test_qwen3_mlp)


add_executable(test_rotary tests/test_rotary.cpp)
target_link_libraries(test_rotary PRIVATE litellm_engine_core)
add_test(NAME test_rotary COMMAND test_rotary)