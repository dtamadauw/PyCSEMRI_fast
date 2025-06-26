# --- libdecomposeGivenFieldMapAndDampings/CMakeLists.txt ---

add_library(decomposeGivenFieldMapAndDampings SHARED
    decomposeGivenFieldMapAndDampings.cpp
)

target_include_directories(decomposeGivenFieldMapAndDampings PUBLIC
    ${CMAKE_SOURCE_DIR}/include/eigen
)

target_compile_options(decomposeGivenFieldMapAndDampings PRIVATE -O3 -Wall -fPIC)

# Install rule for scikit-build-core to pick up the binary
install(TARGETS decomposeGivenFieldMapAndDampings
    LIBRARY DESTINATION package
    ARCHIVE DESTINATION package
)
