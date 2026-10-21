# Inject a Wllm validation target into the locked upstream build without modifying it.
# CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE is evaluated at project(), before upstream targets.
get_filename_component(WLLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(wllm-vulkan "${WLLM_ROOT}/src/main.cpp")
target_link_libraries(wllm-vulkan PRIVATE llama-common psapi dxgi)
target_include_directories(wllm-vulkan PRIVATE "${WLLM_ROOT}/vendor/llama.cpp/vendor" "${WLLM_ROOT}/vendor/llama.cpp/src")
set_target_properties(wllm-vulkan PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(wllm-vulkan PRIVATE WLLM_VULKAN=1 WLLM_AVX512=0 WLLM_CPU_TARGET="avx2"
    WLLM_LLAMA_COMMIT="f2918cabbffe8abf8e2a90c1c089085fa116c2cf")
if(MSVC)
    target_compile_options(wllm-vulkan PRIVATE /utf-8 /W4)
endif()
