cmake_minimum_required (VERSION 3.21)

project (TornadoCK LANGUAGES CXX)

# libtornado-ck: library-task kernels for HIP devices built from AMD Composable Kernel (CK), with a
# C ABI that tornado-cudnn and tornado-cutlass bind through java.lang.foreign. It holds the fused
# FP16 attention kernels (CK tile FMHA) behind the cuDNN SDPA functions and the GEMM kernels (CK
# device operations) behind the CUTLASS functions.
#
# Without ROCm, or on a generator without HIP language support (Visual Studio), the build skips this
# library with a warning; those functions then report the task as unsupported on HIP devices. With
# TORNADO_CK_REQUIRED set, a skipped build fails instead.
macro(tornado_ck_skip reason)
    if(NOT "$ENV{TORNADO_CK_REQUIRED}" STREQUAL "" AND NOT "$ENV{TORNADO_CK_REQUIRED}" STREQUAL "0")
        message(FATAL_ERROR "TornadoVM CK: ${reason} (TORNADO_CK_REQUIRED is set, so this fails the build.)")
    endif()
    message(WARNING "TornadoVM CK: ${reason} Skipping libtornado-ck (the fused SDPA path on HIP devices will report it at runtime).")
    return()
endmacro()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    tornado_ck_skip("the ${CMAKE_GENERATOR} generator cannot compile HIP sources.")
endif()

set(ROCM_ROOT "")
foreach(variable ROCM_PATH HIP_PATH ROCM_HOME)
    if(NOT ROCM_ROOT AND DEFINED ENV{${variable}} AND EXISTS "$ENV{${variable}}")
        set(ROCM_ROOT "$ENV{${variable}}")
    endif()
endforeach()
if(NOT ROCM_ROOT AND EXISTS "/opt/rocm")
    set(ROCM_ROOT "/opt/rocm")
endif()
if(NOT ROCM_ROOT)
    tornado_ck_skip("no ROCm found (looked at ROCM_PATH, HIP_PATH, ROCM_HOME and /opt/rocm).")
endif()
file(TO_CMAKE_PATH "${ROCM_ROOT}" ROCM_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

if(NOT CMAKE_HIP_COMPILER)
    find_program(TORNADO_HIP_CLANG clang++ HINTS "${ROCM_ROOT}/llvm/bin" "${ROCM_ROOT}/lib/llvm/bin" NO_DEFAULT_PATH)
    if(NOT TORNADO_HIP_CLANG)
        tornado_ck_skip("no ROCm clang++ under ${ROCM_ROOT}.")
    endif()
    set(CMAKE_HIP_COMPILER "${TORNADO_HIP_CLANG}")
endif()

# The GPU architectures to build kernels for: TORNADO_CK_GPU_TARGETS (semicolon or comma separated)
# or every architecture the CK attention kernels support. A GPU outside the list reports SDPA as
# unsupported.
if(DEFINED ENV{TORNADO_CK_GPU_TARGETS} AND NOT "$ENV{TORNADO_CK_GPU_TARGETS}" STREQUAL "")
    string(REPLACE "," ";" TORNADO_CK_GPU_TARGETS "$ENV{TORNADO_CK_GPU_TARGETS}")
else()
    set(TORNADO_CK_GPU_TARGETS gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1151 gfx1200 gfx1201)
endif()
set(CMAKE_HIP_ARCHITECTURES ${TORNADO_CK_GPU_TARGETS})
enable_language(HIP)
find_package(hip REQUIRED CONFIG)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# CK is header-only plus a code generator for the attention kernels; fetch just those two
# directories of a pinned commit (the whole repository is large).
set(CK_COMMIT 9abc8908235e8a62460f81d4e54866586b3cda9d)
set(CK_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/composable_kernel-src")
if(NOT EXISTS "${CK_SOURCE_DIR}/.tornado-${CK_COMMIT}")
    find_package(Git REQUIRED)
    file(REMOVE_RECURSE "${CK_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${CK_SOURCE_DIR}")

    macro(ck_git)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} ${ARGN}
            WORKING_DIRECTORY "${CK_SOURCE_DIR}"
            RESULT_VARIABLE _ck_git_rc
        )
        if(NOT _ck_git_rc EQUAL 0)
            message(FATAL_ERROR "Composable Kernel sparse checkout failed: git ${ARGN}")
        endif()
    endmacro()

    ck_git(init -q)
    ck_git(remote add origin https://github.com/ROCm/composable_kernel.git)
    ck_git(sparse-checkout init --cone)
    ck_git(sparse-checkout set include example/ck_tile/01_fmha)
    ck_git(fetch --depth 1 origin ${CK_COMMIT})
    ck_git(checkout -q FETCH_HEAD)
    file(TOUCH "${CK_SOURCE_DIR}/.tornado-${CK_COMMIT}")
endif()
set(CK_FMHA_DIR "${CK_SOURCE_DIR}/example/ck_tile/01_fmha")

# Generate only the forward kernels the SDPA functions call: FP16, batch mode, no bias, no
# dropout, no logits soft-cap, no LSE output, no quantization scales, no sink; with and without
# the causal mask.
string(REPLACE ";" "," CK_TARGETS_ARG "${TORNADO_CK_GPU_TARGETS}")
set(CK_FMHA_GEN_DIR "${CMAKE_BINARY_DIR}/fmha")
file(REMOVE_RECURSE "${CK_FMHA_GEN_DIR}")
file(MAKE_DIRECTORY "${CK_FMHA_GEN_DIR}")
execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CK_FMHA_DIR}/generate.py"
        --targets ${CK_TARGETS_ARG}
        --api fwd
        --optdim 32,64,128,256
        --filter "*fp16_batch*_nlogits_nbias_*_nlse_ndropout_nskip_nqscale*_nsink"
        --output_dir "${CK_FMHA_GEN_DIR}"
    RESULT_VARIABLE _ck_gen_rc
)
if(NOT _ck_gen_rc EQUAL 0)
    message(FATAL_ERROR "Composable Kernel attention code generation failed")
endif()
file(GLOB CK_FMHA_SOURCES "${CK_FMHA_GEN_DIR}/*.cpp")

# CK's build generates ck/config.h; enable what the GEMM operations here use.
set(CK_ENABLE_FP16 ON)
set(CK_ENABLE_BF16 ON)
set(CK_ENABLE_FP32 ON)
set(CK_ENABLE_DL_KERNELS ON)
set(CK_USE_XDL ON)
set(CK_USE_WMMA ON)
configure_file("${CK_SOURCE_DIR}/include/ck/config.h.in" "${CMAKE_BINARY_DIR}/ck-config/ck/config.h")

# One translation unit per GEMM type and fused activation, so they compile in parallel.
set(SOURCE_FILES
    "source/tornado-ck-sdpa.cpp"
    "source/tornado-ck-gemm.cpp"
    "source/tornado-ck-gemm-f32.cpp"
    "source/tornado-ck-gemm-f16.cpp"
    "source/tornado-ck-gemm-bf16.cpp"
    "source/tornado-ck-gemm-bias-relu.cpp"
    "source/tornado-ck-gemm-bias-gelu.cpp"
    "source/tornado-ck-gemm-bias-silu.cpp"
    "source/tornado-ck-gemm-bias-sigmoid.cpp"
    "source/tornado-ck-gemm-bias-tanh.cpp"
    "source/tornado-ck-gemm-bias-hardswish.cpp"
    ${CK_FMHA_SOURCES})
set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE HIP)
add_library(tornado-ck SHARED ${SOURCE_FILES})

set_target_properties(tornado-ck PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
target_include_directories(tornado-ck PRIVATE source "${CMAKE_BINARY_DIR}/ck-config" "${CK_SOURCE_DIR}/include" "${CK_FMHA_DIR}")
# The compile options of CK's own builds; the attention sources also take those of its FMHA example
# (flushing denormals is confined to them).
target_compile_options(tornado-ck PRIVATE
    -O3
    -Wno-undefined-func-template
    -Wno-float-equal
    "SHELL:-mllvm -enable-post-misched=0"
    "SHELL:-mllvm -amdgpu-early-inline-all=true"
    "SHELL:-mllvm -amdgpu-function-calls=false")
set_source_files_properties("source/tornado-ck-sdpa.cpp" ${CK_FMHA_SOURCES} PROPERTIES COMPILE_OPTIONS
    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero;-DCK_TILE_FMHA_FWD_SPLITKV_API=0;-DCK_TILE_FMHA_FWD_APPENDKV_API=0;-DCK_TILE_FMHA_FWD_PAGEDKV_API=0;-DCK_TILE_USE_OCP_FP8")
target_link_libraries(tornado-ck PRIVATE hip::host)

# CK's newer unified tile framework needs the host pass to know the GPU target, which CK takes as
# the first of CK_CMAKE_GPU_TARGET_IDS for every pass and so breaks a build that mixes gfx9 and
# gfx11 targets. Its legacy WarpGemm path (what CK itself uses for gfx1250) has no such dependence.
target_compile_definitions(tornado-ck PRIVATE USE_NEW_UNIFIED_FRAMEWORK=0)

list(LENGTH CK_FMHA_SOURCES CK_FMHA_COUNT)
message(STATUS "TornadoVM CK: ROCm=${ROCM_ROOT}; targets=${TORNADO_CK_GPU_TARGETS}; compiler=${CMAKE_HIP_COMPILER}; ${CK_FMHA_COUNT} attention sources")
