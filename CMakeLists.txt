cmake_minimum_required(VERSION 3.18)

project(LiteLLMEngine LANGUAGES CXX CUDA)

add_subdirectory(3rdparty/tokenizers-cpp)

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
    tokenizers_cpp
)

set_target_properties(litellm_engine_core PROPERTIES
    CUDA_STANDARD 20
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


add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE litellm_engine_core)
add_test(NAME test_attention COMMAND test_attention)


add_executable(test_copy tests/test_copy.cpp)
target_link_libraries(test_copy PRIVATE litellm_engine_core)
add_test(NAME test_copy COMMAND test_copy)


add_executable(test_qwen3_attention tests/test_qwen3_attention.cpp)
target_link_libraries(test_qwen3_attention PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_attention COMMAND test_qwen3_attention)


add_executable(test_qwen3_decoder_layer tests/test_qwen3_decoder_layer.cpp)
target_link_libraries(test_qwen3_decoder_layer PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_decoder_layer COMMAND test_qwen3_decoder_layer)


add_executable(test_qwen3_model tests/test_qwen3_model.cpp)
target_link_libraries(test_qwen3_model PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_model COMMAND test_qwen3_model)


add_executable(test_qwen3_causal_lm tests/test_qwen3_causal_lm.cpp)
target_link_libraries(test_qwen3_causal_lm PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_causal_lm COMMAND test_qwen3_causal_lm)


add_executable(test_argmax tests/test_argmax.cpp)
target_link_libraries(test_argmax PRIVATE litellm_engine_core)
add_test(NAME test_argmax COMMAND test_argmax)


add_executable(test_generation tests/test_generation.cpp)
target_link_libraries(test_generation PRIVATE litellm_engine_core)
add_test(NAME test_generation COMMAND test_generation)


add_executable(test_token_ids tests/test_token_ids.cpp)
target_link_libraries(test_token_ids PRIVATE litellm_engine_core)
add_test(NAME test_token_ids COMMAND test_token_ids)


add_executable(test_tensor_memory tests/test_tensor_memory.cpp)
target_link_libraries(test_tensor_memory PRIVATE litellm_engine_core)
add_test(NAME test_tensor_memory COMMAND test_tensor_memory)


add_executable(test_kv_cache tests/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_kv_cache COMMAND test_kv_cache)


add_executable(test_kv_cache_update tests/test_kv_cache_update.cpp)
target_link_libraries(test_kv_cache_update PRIVATE litellm_engine_core)
add_test(NAME test_kv_cache_update COMMAND test_kv_cache_update)


add_executable(test_qwen3_kv_cache_write tests/test_qwen3_kv_cache_write.cpp)
target_link_libraries(test_qwen3_kv_cache_write PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_kv_cache_write COMMAND test_qwen3_kv_cache_write)


add_executable(test_attention_kv_cache tests/test_attention_kv_cache.cpp)
target_link_libraries(test_attention_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_attention_kv_cache COMMAND test_attention_kv_cache)


add_executable(test_generation_kv_cache tests/test_generation_kv_cache.cpp)
target_link_libraries(test_generation_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_generation_kv_cache COMMAND test_generation_kv_cache)


add_executable(test_request_manager tests/test_request_manager.cpp)
target_link_libraries(test_request_manager PRIVATE litellm_engine_core)
add_test(NAME test_request_manager COMMAND test_request_manager)


add_executable(test_paged_kv_cache tests/test_paged_kv_cache.cpp)
target_link_libraries(test_paged_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_paged_kv_cache COMMAND test_paged_kv_cache)



add_executable(test_block_table_manager tests/test_block_table_manager.cpp)
target_link_libraries(test_block_table_manager PRIVATE litellm_engine_core)
add_test(NAME test_block_table_manager COMMAND test_block_table_manager)


add_executable(test_request_block_table tests/test_request_block_table.cpp)
target_link_libraries(test_request_block_table PRIVATE litellm_engine_core)
add_test(NAME test_request_block_table COMMAND test_request_block_table)


add_executable(test_paged_attention tests/test_paged_attention.cpp)
target_link_libraries(test_paged_attention PRIVATE litellm_engine_core)
add_test(NAME test_paged_attention COMMAND test_paged_attention)


add_executable(test_paged_attention_cuda tests/test_paged_attention_cuda.cpp)
target_link_libraries(test_paged_attention_cuda PRIVATE litellm_engine_core)
add_test(NAME test_paged_attention_cuda COMMAND test_paged_attention_cuda)


add_executable(test_qwen3_attention_paged_kv_cache tests/test_qwen3_attention_paged_kv_cache.cpp)
target_link_libraries(test_qwen3_attention_paged_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_qwen3_attention_paged_kv_cache COMMAND test_qwen3_attention_paged_kv_cache)



add_executable(test_generate_paged_kv_cache tests/test_generate_paged_kv_cache.cpp)
target_link_libraries(test_generate_paged_kv_cache PRIVATE litellm_engine_core)
add_test(NAME test_generate_paged_kv_cache COMMAND test_generate_paged_kv_cache)


add_executable(test_paged_generate_engine tests/test_paged_generate_engine.cpp)
target_link_libraries(test_paged_generate_engine PRIVATE litellm_engine_core)
add_test(NAME test_paged_generate_engine COMMAND test_paged_generate_engine)


add_executable(test_real_qwen3_paged_generate_engine tests/test_real_qwen3_paged_generate_engine.cpp)
target_link_libraries(test_real_qwen3_paged_generate_engine PRIVATE litellm_engine_core)
add_test(NAME test_real_qwen3_paged_generate_engine COMMAND test_real_qwen3_paged_generate_engine)