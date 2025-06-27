# --- libdecomposeGivenFieldMapAndDampings/CMakeLists.txt ---

add_library(decomposeGivenFieldMapAndDampings SHARED
    decomposeGivenFieldMapAndDampings.cpp
)
target_link_libraries(decomposeGivenFieldMapAndDampings PRIVATE eigen)
target_compile_options(decomposeGivenFieldMapAndDampings PRIVATE -O3 -Wall -fPIC)

# The destination is now 'pycsemri'
install(TARGETS decomposeGivenFieldMapAndDampings
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)

