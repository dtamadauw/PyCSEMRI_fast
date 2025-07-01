# --- libdecomposeGivenFieldMapAndDampings/CMakeLists.txt ---

add_library(decomposeGivenFieldMapAndDampings SHARED
    decomposeGivenFieldMapAndDampings.cpp
)


target_include_directories(decomposeGivenFieldMapAndDampings PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_compile_options(decomposeGivenFieldMapAndDampings PRIVATE -O3 -Wall -fPIC)

# The destination is now 'pycsemri'
install(TARGETS decomposeGivenFieldMapAndDampings
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)

