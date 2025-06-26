# --- libuwwfs/CMakeLists.txt ---

add_library(uwwfs SHARED
    libuwwfs.cpp
    fwFit_ComplexLS_1r2star.cpp
    fwFit_MagnLS_1r2star.cpp
    fwFit_MixedLS_1r2star.cpp
)

# Tell this library to use the downloaded Eigen
target_link_libraries(uwwfs PRIVATE eigen)

# We still need the other include path for lsqcpp.hpp
target_include_directories(uwwfs PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(uwwfs PRIVATE -O3 -s -fPIC)
target_compile_definitions(uwwfs PRIVATE NDEBUG)

install(TARGETS uwwfs LIBRARY DESTINATION package ARCHIVE DESTINATION package)

