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

# --- Shared Library ---
add_library(uwwfs SHARED
    libuwwfs.cpp
    fwFit_ComplexLS_1r2star.cpp
    fwFit_MagnLS_1r2star.cpp
    fwFit_MixedLS_1r2star.cpp
    fwFit_VarproLS_1r2star.cpp
    fast_wfs.cpp
)

# --- Static Library ---
# We create a second target for the static build
add_library(uwwfs_static STATIC
    libuwwfs.cpp
    fwFit_ComplexLS_1r2star.cpp
    fwFit_MagnLS_1r2star.cpp
    fwFit_MixedLS_1r2star.cpp
    fwFit_VarproLS_1r2star.cpp
    fast_wfs.cpp
)

# Optional: Rename the output file of the static lib to "libuwwfs.a"
# instead of "libuwwfs_static.a" if you prefer them to have the same base name.
set_target_properties(uwwfs_static PROPERTIES OUTPUT_NAME "uwwfs")

# --- Configuration (Applied to both) ---

foreach(target uwwfs uwwfs_static)
    target_include_directories(${target} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${eigen_SOURCE_DIR}
    )

    # Position-independent code (PIC) is usually required for shared libs,
    # but often good for static libs too if they are linked into shared libs later.
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Interprocedural optimization (LTO)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

    # Common compile-time definitions
    target_compile_definitions(${target} PRIVATE
        NDEBUG                  # disable asserts
        EIGEN_NO_DEBUG          # disable Eigen’s internal checks
        EIGEN_DONT_PARALLELIZE  # force Eigen single-thread
    )

    # Per-compiler optimization flags
    target_compile_options(${target} PRIVATE
        -O3
        -march=native
        -ffast-math
        -fno-math-errno
        -fno-trapping-math
        $<$<CONFIG:Release>:-flto>
    )

    # LTO and aggressive linker opts (Release only)
    target_link_options(${target} PRIVATE
        $<$<CONFIG:Release>:-flto>
        $<$<CONFIG:Release>:-Wl,-O3>
    )
endforeach()

# --- Install ---

install(TARGETS uwwfs uwwfs_static
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)