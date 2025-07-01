# --- libuwwfs/CMakeLists.txt ---

add_library(uwwfs SHARED
    libuwwfs.cpp
    fwFit_ComplexLS_1r2star.cpp
    fwFit_MagnLS_1r2star.cpp
    fwFit_MixedLS_1r2star.cpp
)

target_include_directories(uwwfs PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_compile_options(uwwfs PRIVATE -O3 -s -fPIC)
target_compile_definitions(uwwfs PRIVATE NDEBUG)

# The destination is now 'pycsemri'
install(TARGETS uwwfs
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)


