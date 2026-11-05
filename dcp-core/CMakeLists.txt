cmake_minimum_required(VERSION 3.20)
project(dcp_core VERSION 1.0.0 LANGUAGES CXX C)

# C++20 required for concepts, modules, std::span, etc.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build types
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer")

# Platform-specific flags
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_compile_options(/MP /arch:AVX2 /fp:fast)
else()
    add_compile_options(-mavx2 -mfma -mpopcnt)
endif()

# Warnings as errors for core
add_compile_options(
    -Wall -Wextra -Wpedantic -Werror
    -Wno-unused-parameter
    -Wno-missing-field-initializers
)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
find_package(Threads REQUIRED)

# ONNX Runtime (optional, for Demucs)
option(DCP_WITH_ONNX "Build with ONNX Runtime support for Demucs" OFF)
if(DCP_WITH_ONNX)
    find_package(onnxruntime REQUIRED)
    add_definitions(-DDCP_WITH_ONNX=1)
endif()

# Python bindings (optional)
option(DCP_WITH_PYTHON "Build Python bindings via pybind11" OFF)
if(DCP_WITH_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(pybind11 REQUIRED)
endif()

# SIMD intrinsics detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512)
check_cxx_compiler_flag("-mneon" HAVE_NEON)
if(HAVE_AVX2)
    add_definitions(-DHAVE_AVX2=1)
endif()
if(HAVE_AVX512)
    add_definitions(-DHAVE_AVX512=1)
endif()
if(HAVE_NEON)
    add_definitions(-DHAVE_NEON=1)
endif()

# Source files
set(DCP_CORE_SOURCES
    src/dcp.cpp
    src/dsp/zdf_svf.cpp
    src/dsp/fdm_reverb.cpp
    src/dsp/phase_vocoder.cpp
    src/dsp/sidechain.cpp
    src/dsp/oversample.cpp
    src/dsp/dither.cpp
    src/dsp/utils.cpp
    src/musical/beat_tracker.cpp
    src/musical/phrase_analyzer.cpp
    src/musical/key_detector.cpp
    src/musical/groove_analyzer.cpp
    src/stems/dsp_splitter_v2.cpp
    src/stems/chain.cpp
    src/composer/archetype.cpp
    src/composer/harmonic.cpp
    src/composer/drum_groove.cpp
    src/composer/vocal_intelligence.cpp
    src/composer/generative.cpp
    src/composer/master_bus.cpp
    src/composer/song_c.cpp
    src/pipeline/graph.cpp
    src/pipeline/automation.cpp
    src/pipeline/clip_guard.cpp
)

# SIMD optimized sources (compiled separately for multi-versioning)
if(HAVE_AVX2)
    set(DCP_AVX2_SOURCES src/dsp/zdf_svf_avx2.cpp)
endif()
if(HAVE_NEON)
    set(DCP_NEON_SOURCES src/dsp/zdf_svf_neon.cpp)
endif()

# Static library
add_library(dcp_core STATIC ${DCP_CORE_SOURCES})
target_include_directories(dcp_core PUBLIC include PRIVATE src)
target_link_libraries(dcp_core PUBLIC Threads::Threads)

# AVX2 optimized library (separate compilation unit)
if(HAVE_AVX2)
    add_library(dcp_avx2 STATIC ${DCP_AVX2_SOURCES})
    target_compile_options(dcp_avx2 PRIVATE -mavx2 -mfma)
    target_link_libraries(dcp_avx2 PUBLIC dcp_core)
endif()

# NEON optimized library
if(HAVE_NEON)
    add_library(dcp_neon STATIC ${DCP_NEON_SOURCES})
    target_compile_options(dcp_neon PRIVATE -mneon)
    target_link_libraries(dcp_neon PUBLIC dcp_core)
endif()

# ONNX Runtime wrapper
if(DCP_WITH_ONNX)
    add_library(dcp_onnx STATIC src/stems/demucs_onnx.cpp)
    target_include_directories(dcp_onnx PRIVATE src)
    target_link_libraries(dcp_onnx PUBLIC dcp_core onnxruntime::onnxruntime)
endif()

# Shared library for dynamic linking
add_library(dcp_shared SHARED ${DCP_CORE_SOURCES})
target_include_directories(dcp_shared PUBLIC include PRIVATE src)
target_link_libraries(dcp_shared PUBLIC Threads::Threads)
set_target_properties(dcp_shared PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Install targets
install(TARGETS dcp_core dcp_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES include/dcp/dcp.h include/dcp/dcp_types.h include/dcp/dcp_config.h include/dcp/dcp_errors.h
    DESTINATION include/dcp
)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Python bindings
if(DCP_WITH_PYTHON)
    add_subdirectory(python)
endif()