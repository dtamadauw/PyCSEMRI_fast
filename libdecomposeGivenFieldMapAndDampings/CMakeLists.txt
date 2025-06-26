# --- libdecomposeGivenFieldMapAndDampings/CMakeLists.txt ---

add_library(decomposeGivenFieldMapAndDampings SHARED
    decomposeGivenFieldMapAndDampings.cpp
)

# Tell this library to use the downloaded Eigen
target_link_libraries(decomposeGivenFieldMapAndDampings PRIVATE eigen)

target_compile_options(decomposeGivenFieldMapAndDampings PRIVATE -O3 -Wall -fPIC)

install(TARGETS decomposeGivenFieldMapAndDampings LIBRARY DESTINATION package ARCHIVE DESTINATION package)

