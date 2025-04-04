cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(benchmark, VERSION 0.1.0 LANGUAGES CXX)

# Make this optional
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/home/diaz/miniforge3/envs/cuda/bin/nvcc")
set(CUDAToolkit_ROOT "/home/diaz/miniforge3/envs/cuda")
set(CUDA_TOOLKIT_ROOT_DIR "/home/diaz/miniforge3/envs/cuda")
enable_language(CUDA)

# verbose
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add dependencies
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include(CPM.cmake)

CPMAddPackage(
    NAME cxxopts
    GITHUB_REPOSITORY jarro2783/cxxopts
    VERSION 3.2.0
    OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)

CPMAddPackage(
    NAME matio-cpp
    GIT_TAG v0.2.6
    VERSION 0.2.6
    GITHUB_REPOSITORY ami-iit/matio-cpp
    SOURCE_DIR ${LIB_DIR}/matio-cpp
)

CPMAddPackage("gh:gabime/spdlog@1.8.2")
# CPMAddPackage(
#     NAME Fastor
#     GIT_TAG V0.6.4
#     VERSION 0.6.4
#     GITHUB_REPOSITORY romeric/Fastor
#     SOURCE_DIR ${LIB_DIR}/Fastor
# )

# ##### Benchmark ######
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)

set(EXE_NAME "benchmark_sequence")

add_executable(
    ${EXE_NAME}
)

target_sources(
    ${EXE_NAME}
    PRIVATE
    main.cpp
)

set(
    RELEASE_COMPILE_OPTIONS
    "$<$<CONFIG:Release>:-O3;-mfma;-march=native;-DNDEBUG;-DEIGEN_USE_BLAS;-ffast-math;-finline-limit=1000000;-ffp-contract=fast>"
)

target_compile_options(
    ${EXE_NAME}
    PRIVATE

    # "-DFASTOR_ENABLE_RUNTIME_CHECKS=1"
    # "-DFASTOR_DONT_ALIGN=1"
    "${RELEASE_COMPILE_OPTIONS}"
)

target_link_libraries(
    ${EXE_NAME}
    PUBLIC
    matioCpp::matioCpp
    cxxopts
    spdlog
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    BLAS::BLAS
)

set_target_properties(${EXE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)


### CUDA ###

find_package(CUDAToolkit REQUIRED)

# Add the CUDA test executable
add_library(cuda_kernel SHARED cuda_kernel.cu)
set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_kernel PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(cuda_kernel PRIVATE -O3 --use_fast_math)

# set variables
set(EXE_NAME "benchmark_sequence_cuda")

add_executable(
    ${EXE_NAME}
)

target_sources(
    ${EXE_NAME}
    PRIVATE
    main_cuda.cpp
)

set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(
    ${EXE_NAME}
    PRIVATE
    cxxopts
    spdlog
    CUDA::cudart
    matioCpp::matioCpp
    cuda_kernel
)

### CUDA STATS ###

add_executable(cuda_stats stats.cpp)
target_link_libraries(cuda_stats CUDA::cudart)
set_target_properties(cuda_stats PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
