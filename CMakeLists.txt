cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(sw_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "")
set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "")

#find_package(timemory REQUIRED COMPONENTS headers cuda cupti compile-options arch)
find_package(OpenMP REQUIRED)

set(GPU_BSW_SOURCES
  src/alignments.cpp
  src/driver.cpp
  src/kernel.cpp
  src/utils.cpp
)

set_source_files_properties(${GPU_BSW_SOURCES} PROPERTIES
  LANGUAGE CUDA
  LINKER_LANGUAGE CUDA
)

add_library(gpu_bsw ${GPU_BSW_SOURCES})
target_compile_options(gpu_bsw PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
target_include_directories(gpu_bsw PUBLIC include)

add_executable(program_gpu evaluation/main.cpp)
target_link_libraries(program_gpu PRIVATE gpu_bsw OpenMP::OpenMP_CXX) #timemory
