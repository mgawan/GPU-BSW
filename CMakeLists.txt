cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(sw_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "")

#find_package(timemory REQUIRED COMPONENTS headers cuda cupti compile-options arch)
find_package(OpenMP REQUIRED)

add_subdirectory(submodules/alignment_boilerplate EXCLUDE_FROM_ALL)

set(sources
  src/driver.cpp
)

set(test_sources
  unittest/test_blockShuffleReduce_with_index.cu
  unittest/test_main.cpp
  unittest/test_sequence_process.cu
  unittest/test_warpReduceMax_with_index.cu
)


set_source_files_properties(${sources} ${test_sources} evaluation/main.cpp
PROPERTIES
  LANGUAGE CUDA
  LINKER_LANGUAGE CUDA
)

add_library(gpu_bsw ${sources})
target_include_directories(gpu_bsw PUBLIC include)
target_link_libraries(gpu_bsw PUBLIC albp OpenMP::OpenMP_CXX)
#target_link_libraries(program_gpu PUBLIC timemory OpenMP::OpenMP_CXX)
target_compile_options(gpu_bsw PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    # $<$<COMPILE_LANGUAGE:CUDA>:-G>
)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if(CMAKE_CUDA_COMPILER AND "CUDA" IN_LIST LANGUAGES)
    target_compile_options(gpu_bsw PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
endif()

add_executable(program_gpu evaluation/main.cpp)
target_link_libraries(program_gpu PRIVATE gpu_bsw)

add_executable(unittest ${test_sources})
target_link_libraries(unittest PRIVATE gpu_bsw)

# add_executable(test unittest/test.cu)
# target_link_libraries(test PRIVATE gpu_bsw)


add_custom_target(
    regression_tests ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/regression_test
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/regression_test/* ${CMAKE_BINARY_DIR}/regression_test
)

add_custom_target(
    test-data ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test-data
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test-data/* ${CMAKE_BINARY_DIR}/test-data
)
