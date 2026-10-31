cmake_minimum_required(VERSION 3.14)
project(igbundle_native_check C CXX)

find_package(Git REQUIRED)
set(EXPECTED_REVISION "067b8d7af3f4775b1ad39470d1860696a8a7b3a3")
if(NOT IS_DIRECTORY "${LLAMA_CPP_SOURCE}")
    message(FATAL_ERROR "LLAMA_CPP_SOURCE must name the pinned llama.cpp checkout")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE NATIVE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE REVISION_STATUS
)
if(NOT REVISION_STATUS EQUAL 0 OR NOT NATIVE_REVISION STREQUAL EXPECTED_REVISION)
    message(FATAL_ERROR "Native check requires llama.cpp ${EXPECTED_REVISION}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_SOURCE}" diff --quiet HEAD --
    RESULT_VARIABLE DIRTY_STATUS
)
if(NOT DIRTY_STATUS EQUAL 0)
    message(FATAL_ERROR "Native checkout has tracked modifications")
endif()

add_subdirectory("${LLAMA_CPP_SOURCE}" llama.cpp)
add_executable(native-qwen2 qwen2_logits.cpp)
target_compile_features(native-qwen2 PRIVATE cxx_std_17)
target_compile_definitions(native-qwen2 PRIVATE IGBUNDLE_LLAMA_REVISION="${EXPECTED_REVISION}")
target_link_libraries(native-qwen2 PRIVATE llama)
set_target_properties(native-qwen2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
