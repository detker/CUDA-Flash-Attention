cmake_minimum_required(VERSION 3.18)
project(FlashAttention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CUDAToolkit_ROOT)
    if(EXISTS "/usr/local/cuda-12.6")
        set(CUDAToolkit_ROOT "/usr/local/cuda-12.6")
    elseif(EXISTS "/opt/cuda")
        set(CUDAToolkit_ROOT "/opt/cuda")
    elseif(EXISTS "/usr/lib/cuda")
        set(CUDAToolkit_ROOT "/usr/lib/cuda")
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES  "${PROJECT_SOURCE_DIR}/kernels/*.cu")
file(GLOB_RECURSE C_SOURCES  "${PROJECT_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/kernels/*.cuh")

add_executable(${PROJECT_NAME} ${C_SOURCES} ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/kernels
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda 
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    >
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    dl
    pthread
    rt
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 61
)
