# --- libuwwfs/CMakeLists.txt ---

add_library(uwwfs SHARED
    libuwwfs.cpp
    fwFit_ComplexLS_1r2star.cpp
    fwFit_MagnLS_1r2star.cpp
    fwFit_MixedLS_1r2star.cpp
)

target_include_directories(uwwfs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Eigen
)

target_compile_options(uwwfs PRIVATE -O3 -s -fPIC)
target_compile_definitions(uwwfs PRIVATE NDEBUG)

# Install rule for scikit-build-core to pick up the binary
install(TARGETS uwwfs
    LIBRARY DESTINATION package
    ARCHIVE DESTINATION package
)
