# libgraphCutIterations/CMakeLists.txt

add_library(graphCutIterations SHARED
    graphCutIterations.cpp
    createExpansionGraphVARPRO_fast.cpp
    findLocalMinima.cpp
)

target_include_directories(graphCutIterations PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
    ${boost_SOURCE_DIR}
)

# Set compile options for this specific target.
target_compile_options(graphCutIterations PRIVATE -O3 -fPIC -w)

# The destination for the final library file.
install(TARGETS graphCutIterations
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)

