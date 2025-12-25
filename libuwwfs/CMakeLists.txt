cmake_minimum_required(VERSION 3.15)
project(uwwfs CXX)

# Fix LTO Policy Warning

cmake_policy(SET CMP0069 NEW)

# Set C++ Standard

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure Release if nothing specified

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(uwwfs SHARED
libuwwfs.cpp
fwFit_ComplexLS_1r2star.cpp
fwFit_MagnLS_1r2star.cpp
fwFit_MixedLS_1r2star.cpp
fwFit_VarproLS_1r2star.cpp
fast_wfs.cpp
)

target_include_directories(uwwfs PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

# Position-independent code for shared libs (portable alternative to -fPIC)

set_target_properties(uwwfs PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Interprocedural optimization (LTO) where available

set_property(TARGET uwwfs PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# Common compile-time definitions

target_compile_definitions(uwwfs PRIVATE
NDEBUG # disable asserts
EIGEN_NO_DEBUG # disable Eigen’s internal checks
EIGEN_DONT_PARALLELIZE # force Eigen single-thread
)

# Per-compiler optimization flags (assuming GCC/Clang based on build logs)

# MSVC flags are in the doc if needed

target_compile_options(uwwfs PRIVATE
\-O3
\-march=native # Use scanner CPU features
\-ffast-math
\-fno-math-errno
\-fno-trapping-math
$<$<CONFIG:Release>:-flto> # FIXED: Removed quotes
)

# LTO and aggressive linker opts (only in Release)

target_link_options(uwwfs PRIVATE
$<$<CONFIG:Release>:-flto> # FIXED: Removed quotes
$<$<CONFIG:Release>:-Wl,-O3> # FIXED: Removed quotes

# Optional: strip at link time

# $<$<CONFIG:Release>:-s>
)

# The destination is now 'pycsemri'

install(TARGETS uwwfs
LIBRARY DESTINATION pycsemri
ARCHIVE DESTINATION pycsemri
)